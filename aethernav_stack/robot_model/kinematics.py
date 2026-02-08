"""
Differential drive kinematics.

Provides forward and inverse kinematics for a two-wheeled differential
drive robot, converting between robot body velocities and individual
wheel velocities.
"""

import numpy as np
from dataclasses import dataclass

from ..core.types import Pose2D, Twist2D, WheelVelocities
from ..core.logging_config import get_logger

logger = get_logger("kinematics")


class DiffDriveKinematics:
    """
    Differential drive kinematics calculator.
    
    Uses the standard differential drive model:
    - v = (v_r + v_l) / 2           (linear velocity)
    - ω = (v_r - v_l) / L           (angular velocity)
    
    Where:
    - v_r, v_l are right/left wheel linear velocities
    - L is the track width (distance between wheels)
    """
    
    def __init__(self, wheel_radius: float, track_width: float):
        """
        Initialize kinematics calculator.
        
        Args:
            wheel_radius: Radius of drive wheels (meters)
            track_width: Distance between wheel centers (meters)
        """
        if wheel_radius <= 0:
            raise ValueError(f"wheel_radius must be positive, got {wheel_radius}")
        if track_width <= 0:
            raise ValueError(f"track_width must be positive, got {track_width}")
            
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        
        logger.debug(
            f"Initialized DiffDriveKinematics: "
            f"wheel_radius={wheel_radius:.4f}m, track_width={track_width:.4f}m"
        )
    
    def inverse(self, twist: Twist2D) -> WheelVelocities:
        """
        Convert robot velocity to wheel angular velocities (inverse kinematics).
        
        Given desired linear and angular velocity of the robot, compute
        the required angular velocity of each wheel.
        
        Args:
            twist: Desired robot velocity (linear m/s, angular rad/s)
            
        Returns:
            WheelVelocities: Angular velocity of left and right wheels (rad/s)
        """
        v = twist.linear       # Linear velocity (m/s)
        omega = twist.angular  # Angular velocity (rad/s)
        
        # Compute wheel linear velocities
        # v_l = v - ω * L/2
        # v_r = v + ω * L/2
        v_left = v - (omega * self.track_width / 2.0)
        v_right = v + (omega * self.track_width / 2.0)
        
        # Convert to angular velocities: ω_wheel = v_linear / r
        omega_left = v_left / self.wheel_radius
        omega_right = v_right / self.wheel_radius
        
        return WheelVelocities(left=omega_left, right=omega_right)
    
    def forward(self, wheels: WheelVelocities) -> Twist2D:
        """
        Convert wheel velocities to robot velocity (forward kinematics).
        
        Given wheel angular velocities, compute the resulting robot
        linear and angular velocity.
        
        Args:
            wheels: Angular velocity of left and right wheels (rad/s)
            
        Returns:
            Twist2D: Robot linear (m/s) and angular (rad/s) velocity
        """
        # Convert angular to linear wheel velocities
        v_left = wheels.left * self.wheel_radius
        v_right = wheels.right * self.wheel_radius
        
        # Compute robot velocities
        # v = (v_r + v_l) / 2
        # ω = (v_r - v_l) / L
        linear = (v_right + v_left) / 2.0
        angular = (v_right - v_left) / self.track_width
        
        return Twist2D(linear=linear, angular=angular)
    
    def compute_pose_delta(
        self,
        wheels: WheelVelocities,
        dt: float,
        current_theta: float = 0.0
    ) -> Pose2D:
        """
        Compute pose change from wheel velocities over a time step.
        
        Used for odometry integration. Assumes constant velocity during dt.
        
        Args:
            wheels: Wheel angular velocities during the interval
            dt: Time step (seconds)
            current_theta: Current robot orientation (radians)
            
        Returns:
            Pose2D: Change in pose (dx, dy, dtheta)
        """
        if dt <= 0:
            return Pose2D()
        
        # Get robot velocity from wheel velocities
        twist = self.forward(wheels)
        
        # Compute pose delta using arc approximation
        # For small dt, use simple integration
        v = twist.linear
        omega = twist.angular
        
        if abs(omega) < 1e-6:
            # Straight line motion
            dx = v * dt * np.cos(current_theta)
            dy = v * dt * np.sin(current_theta)
            dtheta = 0.0
        else:
            # Arc motion - use exact integration
            # Robot moves along an arc of radius r = v/omega
            dtheta = omega * dt
            
            # Displacement along the arc
            dx = (v / omega) * (np.sin(current_theta + dtheta) - np.sin(current_theta))
            dy = (v / omega) * (-np.cos(current_theta + dtheta) + np.cos(current_theta))
        
        return Pose2D(x=dx, y=dy, theta=dtheta)
    
    def clamp_wheel_velocities(
        self,
        wheels: WheelVelocities,
        max_wheel_vel: float,
        preserve_ratio: bool = True
    ) -> WheelVelocities:
        """
        Clamp wheel velocities to maximum, optionally preserving the ratio.
        
        Args:
            wheels: Input wheel velocities
            max_wheel_vel: Maximum allowed wheel angular velocity (rad/s)
            preserve_ratio: If True, scale both wheels to preserve turning ratio
            
        Returns:
            WheelVelocities: Clamped wheel velocities
        """
        if preserve_ratio:
            # Scale both wheels proportionally if either exceeds max
            max_current = max(abs(wheels.left), abs(wheels.right))
            if max_current > max_wheel_vel:
                scale = max_wheel_vel / max_current
                return WheelVelocities(
                    left=wheels.left * scale,
                    right=wheels.right * scale
                )
            return wheels
        else:
            # Clamp each wheel independently
            return wheels.clamp(max_wheel_vel)
    
    def compute_turn_radius(self, twist: Twist2D) -> float:
        """
        Compute instantaneous turn radius for the given velocity.
        
        Args:
            twist: Robot velocity
            
        Returns:
            Turn radius in meters. Returns float('inf') for straight motion.
        """
        if abs(twist.angular) < 1e-6:
            return float('inf')
        return abs(twist.linear / twist.angular)
