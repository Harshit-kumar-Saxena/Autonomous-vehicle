"""
Robot physical parameters.

Provides a structured interface to robot physical properties like
dimensions, velocity limits, and motor constraints.
"""

from dataclasses import dataclass
from typing import Optional

from ..config import RobotConfig
from ..core.logging_config import get_logger

logger = get_logger("robot_params")


@dataclass
class RobotParams:
    """
    Complete robot physical parameters.
    
    All dimensions in meters, velocities in m/s or rad/s.
    """
    # Physical dimensions
    wheel_radius: float = 0.05      # meters
    track_width: float = 0.20       # meters
    
    # Velocity limits
    max_linear_vel: float = 0.5     # m/s
    max_angular_vel: float = 1.0    # rad/s
    max_wheel_vel: float = 10.0     # rad/s (angular velocity of wheel)
    max_wheel_accel: float = 5.0    # rad/s² (angular acceleration of wheel)
    
    # Optional physical properties
    name: str = "AetherNav Bot"
    mass: Optional[float] = None    # kg
    wheel_base: Optional[float] = None  # meters (for Ackermann, not used for diff drive)
    
    @classmethod
    def from_config(cls, config: RobotConfig) -> "RobotParams":
        """Create RobotParams from configuration."""
        return cls(
            wheel_radius=config.wheel_radius,
            track_width=config.track_width,
            max_linear_vel=config.max_linear_vel,
            max_angular_vel=config.max_angular_vel,
            max_wheel_vel=config.max_wheel_vel,
            max_wheel_accel=config.max_wheel_accel,
            name=config.name,
        )
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate that all parameters are physically reasonable."""
        errors = []
        
        if self.wheel_radius <= 0:
            errors.append(f"wheel_radius must be positive, got {self.wheel_radius}")
        if self.track_width <= 0:
            errors.append(f"track_width must be positive, got {self.track_width}")
        if self.max_linear_vel <= 0:
            errors.append(f"max_linear_vel must be positive, got {self.max_linear_vel}")
        if self.max_angular_vel <= 0:
            errors.append(f"max_angular_vel must be positive, got {self.max_angular_vel}")
        if self.max_wheel_vel <= 0:
            errors.append(f"max_wheel_vel must be positive, got {self.max_wheel_vel}")
        if self.max_wheel_accel <= 0:
            errors.append(f"max_wheel_accel must be positive, got {self.max_wheel_accel}")
        
        if errors:
            error_msg = "Invalid robot parameters:\n  " + "\n  ".join(errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check consistency: max_wheel_vel should be able to achieve max_linear_vel
        max_linear_from_wheels = self.max_wheel_vel * self.wheel_radius
        if max_linear_from_wheels < self.max_linear_vel:
            logger.warning(
                f"max_wheel_vel ({self.max_wheel_vel} rad/s) may be too low to achieve "
                f"max_linear_vel ({self.max_linear_vel} m/s). "
                f"Wheel can only provide {max_linear_from_wheels:.3f} m/s."
            )
    
    @property
    def min_turn_radius(self) -> float:
        """
        Minimum turn radius the robot can achieve.
        
        At max angular velocity and max linear velocity:
        R = v / omega
        """
        return self.max_linear_vel / self.max_angular_vel
    
    @property
    def wheel_circumference(self) -> float:
        """Circumference of drive wheel."""
        import math
        return 2.0 * math.pi * self.wheel_radius
    
    def describe(self) -> str:
        """Return a human-readable description of the robot."""
        return (
            f"Robot: {self.name}\n"
            f"  Wheel radius: {self.wheel_radius * 100:.1f} cm\n"
            f"  Track width: {self.track_width * 100:.1f} cm\n"
            f"  Max linear vel: {self.max_linear_vel:.2f} m/s\n"
            f"  Max angular vel: {self.max_angular_vel:.2f} rad/s\n"
            f"  Max wheel vel: {self.max_wheel_vel:.2f} rad/s\n"
            f"  Min turn radius: {self.min_turn_radius:.3f} m"
        )
