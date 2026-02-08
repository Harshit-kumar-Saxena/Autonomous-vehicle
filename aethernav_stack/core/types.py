"""
Core data types used across the AetherNav stack.

These dataclasses provide a consistent, typed interface for passing data
between layers of the navigation stack.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import time


@dataclass
class Pose2D:
    """
    2D pose in the robot's odometry frame.
    
    Attributes:
        x: Position along X-axis (meters)
        y: Position along Y-axis (meters)
        theta: Orientation (radians, counter-clockwise from X-axis)
    """
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    
    def __add__(self, other: "Pose2D") -> "Pose2D":
        """Add two poses (for integrating deltas)."""
        return Pose2D(
            x=self.x + other.x,
            y=self.y + other.y,
            theta=self._normalize_angle(self.theta + other.theta)
        )
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, theta]."""
        return np.array([self.x, self.y, self.theta])


@dataclass
class Twist2D:
    """
    2D velocity command (linear and angular).
    
    Attributes:
        linear: Linear velocity (m/s, positive = forward)
        angular: Angular velocity (rad/s, positive = counter-clockwise)
    """
    linear: float = 0.0
    angular: float = 0.0
    
    def is_zero(self, tolerance: float = 1e-6) -> bool:
        """Check if velocity is effectively zero."""
        return abs(self.linear) < tolerance and abs(self.angular) < tolerance
    
    def clamp(self, max_linear: float, max_angular: float) -> "Twist2D":
        """Return a new Twist2D clamped to the given limits."""
        return Twist2D(
            linear=np.clip(self.linear, -max_linear, max_linear),
            angular=np.clip(self.angular, -max_angular, max_angular)
        )


@dataclass
class WheelVelocities:
    """
    Individual wheel angular velocities for differential drive.
    
    Attributes:
        left: Left wheel angular velocity (rad/s)
        right: Right wheel angular velocity (rad/s)
    """
    left: float = 0.0
    right: float = 0.0
    
    def clamp(self, max_vel: float) -> "WheelVelocities":
        """Return new WheelVelocities clamped to max angular velocity."""
        return WheelVelocities(
            left=np.clip(self.left, -max_vel, max_vel),
            right=np.clip(self.right, -max_vel, max_vel)
        )
    
    def to_pwm(self, max_wheel_vel: float) -> tuple[int, int]:
        """
        Convert angular velocities to PWM values (0-255).
        
        Uses linear mapping: PWM = |velocity| / max_vel * 255
        Returns (left_pwm, right_pwm) as integers.
        """
        left_pwm = int(abs(self.left) / max_wheel_vel * 255)
        right_pwm = int(abs(self.right) / max_wheel_vel * 255)
        return (
            np.clip(left_pwm, 0, 255),
            np.clip(right_pwm, 0, 255)
        )
    
    def get_directions(self) -> tuple[int, int]:
        """Get motor directions (1 = forward, 0 = backward)."""
        return (
            1 if self.left >= 0 else 0,
            1 if self.right >= 0 else 0
        )


@dataclass
class SegmentationResult:
    """
    Output from the segmentation engine.
    
    Attributes:
        mask: Raw segmentation mask (H x W probability or binary)
        centerline_points: List of (x, y) centerline points from bottom to top
        lane_center_normalized: Lane center as fraction of image width
                                (0.0 = left edge, 1.0 = right edge)
        lane_detected: Whether a valid lane was detected
        confidence: Confidence score (0.0 to 1.0)
        timestamp: Time when frame was captured (seconds since epoch)
    """
    mask: np.ndarray
    centerline_points: list = field(default_factory=list)
    lane_center_normalized: float = 0.5
    lane_detected: bool = False
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    @property
    def lane_error(self) -> float:
        """
        Signed error from center (-0.5 to 0.5).
        Negative = lane center is to the left
        Positive = lane center is to the right
        """
        return self.lane_center_normalized - 0.5


@dataclass 
class RobotState:
    """
    Complete robot state at a given time.
    
    Combines pose, velocity, and sensor data for use by planners
    and controllers.
    """
    pose: Pose2D = field(default_factory=Pose2D)
    twist: Twist2D = field(default_factory=Twist2D)
    wheel_velocities: WheelVelocities = field(default_factory=WheelVelocities)
    imu_yaw: Optional[float] = None  # From IMU if available
    timestamp: float = field(default_factory=time.time)
    
    def update_pose(self, delta: Pose2D) -> None:
        """Update pose by adding a delta."""
        self.pose = self.pose + delta
        self.timestamp = time.time()
