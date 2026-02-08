"""
Odometry computation for differential drive.

Integrates wheel velocities and optional IMU data to estimate
the robot's pose over time.
"""

import time
from typing import Optional
from dataclasses import dataclass, field

from ..core.types import Pose2D, Twist2D, WheelVelocities, RobotState
from ..robot_model.kinematics import DiffDriveKinematics
from ..core.logging_config import get_logger

logger = get_logger("controls.odometry")


@dataclass
class OdometryState:
    """Internal odometry state."""
    pose: Pose2D = field(default_factory=Pose2D)
    twist: Twist2D = field(default_factory=Twist2D)
    last_update_time: float = field(default_factory=time.time)
    total_distance: float = 0.0
    
    
class Odometry:
    """
    Odometry estimator for differential drive robots.
    
    Integrates wheel velocities to estimate robot pose. Optionally
    fuses IMU yaw data for improved heading estimation.
    """
    
    def __init__(
        self,
        kinematics: DiffDriveKinematics,
        use_imu_yaw: bool = True,
        imu_yaw_weight: float = 0.8
    ):
        """
        Initialize odometry estimator.
        
        Args:
            kinematics: Differential drive kinematics calculator
            use_imu_yaw: Whether to use IMU yaw for heading (if available)
            imu_yaw_weight: Weight for IMU yaw vs integrated yaw (0-1)
        """
        self.kinematics = kinematics
        self.use_imu_yaw = use_imu_yaw
        self.imu_yaw_weight = imu_yaw_weight
        
        self._state = OdometryState()
        self._imu_yaw_offset: Optional[float] = None
    
    def update(
        self,
        wheel_velocities: WheelVelocities,
        imu_yaw: Optional[float] = None,
        timestamp: Optional[float] = None
    ) -> RobotState:
        """
        Update odometry with new wheel velocity data.
        
        Args:
            wheel_velocities: Current wheel angular velocities
            imu_yaw: Optional IMU yaw angle (radians)
            timestamp: Measurement timestamp (defaults to current time)
            
        Returns:
            Updated robot state
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Calculate time delta
        dt = timestamp - self._state.last_update_time
        
        if dt <= 0:
            # No time has passed, return current state
            return self._get_robot_state()
        
        # Cap dt to prevent huge jumps on resume
        dt = min(dt, 0.5)
        
        # Compute robot velocity from wheel velocities
        self._state.twist = self.kinematics.forward(wheel_velocities)
        
        # Compute pose delta using kinematics
        pose_delta = self.kinematics.compute_pose_delta(
            wheel_velocities,
            dt,
            self._state.pose.theta
        )
        
        # Update pose
        self._state.pose = self._state.pose + pose_delta
        
        # Fuse IMU yaw if available and enabled
        if self.use_imu_yaw and imu_yaw is not None:
            self._fuse_imu_yaw(imu_yaw)
        
        # Update total distance traveled
        self._state.total_distance += abs(self._state.twist.linear * dt)
        
        # Update timestamp
        self._state.last_update_time = timestamp
        
        return self._get_robot_state()
    
    def _fuse_imu_yaw(self, imu_yaw: float) -> None:
        """
        Fuse IMU yaw measurement with integrated yaw.
        
        Uses complementary filter: weighted average of IMU and integrated yaw.
        """
        # Initialize offset on first IMU reading
        if self._imu_yaw_offset is None:
            self._imu_yaw_offset = self._state.pose.theta - imu_yaw
            return
        
        # Correct IMU yaw with offset
        corrected_imu_yaw = imu_yaw + self._imu_yaw_offset
        
        # Complementary filter
        integrated_yaw = self._state.pose.theta
        fused_yaw = (
            self.imu_yaw_weight * corrected_imu_yaw +
            (1.0 - self.imu_yaw_weight) * integrated_yaw
        )
        
        # Update pose with fused yaw
        self._state.pose = Pose2D(
            x=self._state.pose.x,
            y=self._state.pose.y,
            theta=Pose2D._normalize_angle(fused_yaw)
        )
    
    def _get_robot_state(self) -> RobotState:
        """Get current robot state."""
        return RobotState(
            pose=Pose2D(
                x=self._state.pose.x,
                y=self._state.pose.y,
                theta=self._state.pose.theta
            ),
            twist=Twist2D(
                linear=self._state.twist.linear,
                angular=self._state.twist.angular
            ),
            timestamp=self._state.last_update_time
        )
    
    def reset(self, initial_pose: Optional[Pose2D] = None) -> None:
        """
        Reset odometry to a known pose.
        
        Args:
            initial_pose: Starting pose (defaults to origin)
        """
        if initial_pose is None:
            initial_pose = Pose2D()
        
        self._state = OdometryState(pose=initial_pose)
        self._imu_yaw_offset = None
        
        logger.info(
            f"Odometry reset to ({initial_pose.x:.3f}, {initial_pose.y:.3f}, "
            f"{initial_pose.theta:.3f})"
        )
    
    @property
    def pose(self) -> Pose2D:
        """Get current pose estimate."""
        return self._state.pose
    
    @property
    def twist(self) -> Twist2D:
        """Get current velocity estimate."""
        return self._state.twist
    
    @property
    def total_distance(self) -> float:
        """Get total distance traveled since reset."""
        return self._state.total_distance
    
    def get_status(self) -> dict:
        """Get odometry status for telemetry."""
        return {
            "x": self._state.pose.x,
            "y": self._state.pose.y,
            "theta": self._state.pose.theta,
            "linear_vel": self._state.twist.linear,
            "angular_vel": self._state.twist.angular,
            "total_distance": self._state.total_distance,
        }
