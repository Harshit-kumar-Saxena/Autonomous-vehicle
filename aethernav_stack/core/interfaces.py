"""
Abstract base classes for plugin architecture.

These interfaces define the contracts that planners and controllers
must implement, enabling easy extensibility and swapping of algorithms.
"""

from abc import ABC, abstractmethod
from typing import Optional
from .types import Pose2D, Twist2D, WheelVelocities, SegmentationResult, RobotState


class PlannerPlugin(ABC):
    """
    Abstract base class for motion planners.
    
    Planners convert perception data (e.g., lane segmentation) into
    velocity commands (Twist2D) for the robot.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the planner's identifier."""
        pass
    
    @abstractmethod
    def compute_velocity(
        self,
        perception_data: SegmentationResult,
        robot_state: RobotState
    ) -> Twist2D:
        """
        Compute velocity command based on perception and current state.
        
        Args:
            perception_data: Current segmentation/perception result
            robot_state: Current robot pose, velocity, and sensor data
            
        Returns:
            Twist2D: Desired linear and angular velocity
        """
        pass
    
    def reset(self) -> None:
        """Reset any internal state (e.g., error accumulators)."""
        pass


class ControllerPlugin(ABC):
    """
    Abstract base class for low-level controllers.
    
    Controllers convert high-level velocity commands (Twist2D) into
    individual wheel velocities, optionally using feedback for
    closed-loop control.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the controller's identifier."""
        pass
    
    @abstractmethod
    def compute_wheel_velocities(
        self,
        cmd_vel: Twist2D,
        robot_state: RobotState,
        dt: float
    ) -> WheelVelocities:
        """
        Compute wheel velocities to achieve the commanded velocity.
        
        Args:
            cmd_vel: Desired robot velocity (linear, angular)
            robot_state: Current robot state including wheel feedback
            dt: Time since last control update (seconds)
            
        Returns:
            WheelVelocities: Target angular velocities for each wheel
        """
        pass
    
    def reset(self) -> None:
        """Reset any internal state (e.g., PID integrators)."""
        pass


class HardwarePlugin(ABC):
    """
    Abstract base class for hardware interfaces.
    
    Abstracts the underlying hardware communication (motors, sensors)
    to allow testing with mock implementations.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to hardware. Returns True on success."""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Cleanly disconnect from hardware."""
        pass
    
    @abstractmethod
    def write_wheel_velocities(self, wheels: WheelVelocities) -> bool:
        """Send wheel velocity commands to motors."""
        pass
    
    @abstractmethod
    def read_wheel_velocities(self) -> Optional[WheelVelocities]:
        """Read current wheel velocities from encoders (if available)."""
        pass
    
    @abstractmethod
    def read_imu_yaw(self) -> Optional[float]:
        """Read yaw angle from IMU (if available)."""
        pass
    
    @abstractmethod
    def emergency_stop(self) -> None:
        """Immediately stop all motors."""
        pass
    
    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        pass
