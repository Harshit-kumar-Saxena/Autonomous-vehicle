"""
PID-based velocity controller.

Converts high-level Twist2D commands into wheel velocities using
differential drive kinematics, with optional PID feedback control.
"""

from dataclasses import dataclass, field
from typing import Optional
import time

from ..core.interfaces import ControllerPlugin
from ..core.types import Twist2D, WheelVelocities, RobotState
from ..robot_model.kinematics import DiffDriveKinematics
from ..robot_model.robot_params import RobotParams
from ..config import PIDGains
from ..core.logging_config import get_logger

logger = get_logger("controls.pid")


@dataclass
class PIDState:
    """PID controller state for one axis."""
    integral: float = 0.0
    last_error: float = 0.0
    last_time: float = field(default_factory=time.time)
    
    def reset(self):
        """Reset PID state."""
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = time.time()


class PID:
    """
    Single-axis PID controller.
    
    Implements the standard PID control law with anti-windup.
    """
    
    def __init__(self, gains: PIDGains):
        """
        Initialize PID controller.
        
        Args:
            gains: PID gains (kp, ki, kd, windup_limit)
        """
        self.kp = gains.kp
        self.ki = gains.ki
        self.kd = gains.kd
        self.windup_limit = gains.windup_limit
        self._state = PIDState()
    
    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID output.
        
        Args:
            error: Current error (setpoint - measurement)
            dt: Time since last update
            
        Returns:
            Control output
        """
        if dt <= 0:
            return self.kp * error
        
        # Proportional term
        p_term = self.kp * error
        
        # Integral term with anti-windup
        self._state.integral += error * dt
        self._state.integral = max(
            -self.windup_limit,
            min(self.windup_limit, self._state.integral)
        )
        i_term = self.ki * self._state.integral
        
        # Derivative term
        d_error = (error - self._state.last_error) / dt
        d_term = self.kd * d_error
        
        # Update state
        self._state.last_error = error
        
        return p_term + i_term + d_term
    
    def reset(self):
        """Reset PID state."""
        self._state.reset()


class PIDController(ControllerPlugin):
    """
    PID-based velocity controller for differential drive.
    
    Converts Twist2D commands to wheel velocities using:
    1. Inverse kinematics to get target wheel velocities
    2. Optional PID feedback to correct velocity errors
    3. Acceleration limiting for smooth motion
    """
    
    def __init__(
        self,
        kinematics: DiffDriveKinematics,
        robot_params: RobotParams,
        pid_linear: Optional[PIDGains] = None,
        pid_angular: Optional[PIDGains] = None,
        use_feedback: bool = False
    ):
        """
        Initialize PID controller.
        
        Args:
            kinematics: Differential drive kinematics calculator
            robot_params: Robot physical parameters
            pid_linear: PID gains for linear velocity (None = feedforward only)
            pid_angular: PID gains for angular velocity (None = feedforward only)
            use_feedback: Whether to use feedback control (requires encoder data)
        """
        self.kinematics = kinematics
        self.robot_params = robot_params
        self.use_feedback = use_feedback
        
        # Create PID controllers if gains provided
        self._pid_linear = PID(pid_linear) if pid_linear else None
        self._pid_angular = PID(pid_angular) if pid_angular else None
        
        # State for acceleration limiting
        self._last_wheel_cmd = WheelVelocities()
        self._last_time = time.time()
    
    @property
    def name(self) -> str:
        return "pid_controller"
    
    def compute_wheel_velocities(
        self,
        cmd_vel: Twist2D,
        robot_state: RobotState,
        dt: float
    ) -> WheelVelocities:
        """
        Compute wheel velocities to achieve commanded velocity.
        
        Args:
            cmd_vel: Desired robot velocity
            robot_state: Current robot state
            dt: Time since last update
            
        Returns:
            Target wheel velocities
        """
        # Clamp commanded velocity to limits
        cmd_vel = cmd_vel.clamp(
            self.robot_params.max_linear_vel,
            self.robot_params.max_angular_vel
        )
        
        # Use inverse kinematics to get target wheel velocities
        target_wheels = self.kinematics.inverse(cmd_vel)
        
        # Apply feedback control if enabled and we have encoder data
        if self.use_feedback and robot_state.wheel_velocities is not None:
            target_wheels = self._apply_feedback(
                target_wheels, 
                robot_state.wheel_velocities,
                dt
            )
        
        # Apply acceleration limiting
        target_wheels = self._apply_acceleration_limit(target_wheels, dt)
        
        # Clamp to max wheel velocity
        target_wheels = self.kinematics.clamp_wheel_velocities(
            target_wheels,
            self.robot_params.max_wheel_vel,
            preserve_ratio=True
        )
        
        # Store for next iteration
        self._last_wheel_cmd = target_wheels
        
        return target_wheels
    
    def _apply_feedback(
        self,
        target: WheelVelocities,
        actual: WheelVelocities,
        dt: float
    ) -> WheelVelocities:
        """
        Apply PID feedback correction.
        
        Args:
            target: Target wheel velocities
            actual: Measured wheel velocities
            dt: Time step
            
        Returns:
            Corrected wheel velocities
        """
        if self._pid_linear is None and self._pid_angular is None:
            return target
        
        # Convert to robot velocity for error calculation
        target_twist = self.kinematics.forward(target)
        actual_twist = self.kinematics.forward(actual)
        
        # Compute errors
        linear_error = target_twist.linear - actual_twist.linear
        angular_error = target_twist.angular - actual_twist.angular
        
        # Apply PID correction
        linear_correction = 0.0
        angular_correction = 0.0
        
        if self._pid_linear:
            linear_correction = self._pid_linear.compute(linear_error, dt)
        if self._pid_angular:
            angular_correction = self._pid_angular.compute(angular_error, dt)
        
        # Add corrections to target
        corrected_twist = Twist2D(
            linear=target_twist.linear + linear_correction,
            angular=target_twist.angular + angular_correction
        )
        
        # Convert back to wheel velocities
        return self.kinematics.inverse(corrected_twist)
    
    def _apply_acceleration_limit(
        self,
        target: WheelVelocities,
        dt: float
    ) -> WheelVelocities:
        """
        Apply acceleration limiting to prevent jerky motion.
        
        Args:
            target: Desired wheel velocities
            dt: Time step
            
        Returns:
            Acceleration-limited wheel velocities
        """
        if dt <= 0:
            return target
        
        max_delta = self.robot_params.max_wheel_accel * dt
        
        # Limit change in each wheel velocity
        left_delta = target.left - self._last_wheel_cmd.left
        right_delta = target.right - self._last_wheel_cmd.right
        
        left_delta = max(-max_delta, min(max_delta, left_delta))
        right_delta = max(-max_delta, min(max_delta, right_delta))
        
        return WheelVelocities(
            left=self._last_wheel_cmd.left + left_delta,
            right=self._last_wheel_cmd.right + right_delta
        )
    
    def reset(self):
        """Reset controller state."""
        if self._pid_linear:
            self._pid_linear.reset()
        if self._pid_angular:
            self._pid_angular.reset()
        self._last_wheel_cmd = WheelVelocities()
        logger.debug("PID controller reset")
