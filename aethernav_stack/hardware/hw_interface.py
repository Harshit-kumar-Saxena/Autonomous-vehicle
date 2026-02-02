"""
Hardware interface wrapping AetherNav SDK.

Provides a clean abstraction over the AetherNav hardware SDK,
implementing the HardwarePlugin interface for use with the
navigation stack.
"""

import sys
from pathlib import Path
from typing import Optional
import time

from ..core.interfaces import HardwarePlugin
from ..core.types import WheelVelocities
from ..robot_model.robot_params import RobotParams
from ..config import HardwareConfig
from ..core.logging_config import get_logger, RateLimitedLogger

logger = get_logger("hardware")

# Add athernav_hardware_interface to path
SDK_PATH = Path(__file__).parent.parent.parent.parent / "athernav_hardware_interface"
if SDK_PATH.exists() and str(SDK_PATH) not in sys.path:
    sys.path.insert(0, str(SDK_PATH))

# Try to import athernav_hardware_interface
try:
    from src import HardwareInterface as AetherNavSDK, LogLevel as SDKLogLevel
    SDK_AVAILABLE = True
except ImportError as e:
    SDK_AVAILABLE = False
    logger.warning(f"athernav_hardware_interface not available: {e}")


class AetherNavHardwareInterface(HardwarePlugin):
    """
    Hardware interface using AetherNav SDK.
    
    Wraps the AetherNav SDK to provide motor control and sensor
    reading capabilities for the navigation stack.
    """
    
    def __init__(
        self,
        config: HardwareConfig,
        robot_params: RobotParams
    ):
        """
        Initialize hardware interface.
        
        Args:
            config: Hardware configuration
            robot_params: Robot physical parameters
        """
        self.config = config
        self.robot_params = robot_params
        self._sdk: Optional[AetherNavSDK] = None
        self._connected = False
        self._rate_limited_logger = RateLimitedLogger(logger, min_interval=5.0)
        
        # Track last commands for status
        self._last_wheel_cmd = WheelVelocities()
        self._last_cmd_time = 0.0
    
    def connect(self) -> bool:
        """
        Connect to AetherNav hardware.
        
        Returns:
            True if connection successful
        """
        if not SDK_AVAILABLE:
            logger.warning("AetherNav SDK not available - running in mock mode")
            self._connected = True
            return True
        
        try:
            logger.info(f"Connecting to AetherNav board on {self.config.port}")
            
            self._sdk = AetherNavSDK(
                port=self.config.port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout,
                log_level=SDKLogLevel.INFO,
                auto_reconnect=self.config.auto_reconnect
            )
            
            if not self._sdk.connect():
                logger.error("Failed to connect to AetherNav board")
                return False
            
            # Configure motor limits
            self._configure_motor_limits()
            
            self._connected = True
            logger.info("Successfully connected to AetherNav board")
            return True
            
        except Exception as e:
            logger.error(f"Exception connecting to hardware: {e}")
            self._connected = False
            return False
    
    def _configure_motor_limits(self) -> None:
        """Configure motor speed and acceleration limits on the board."""
        if self._sdk is None:
            return
        
        try:
            # Convert from rad/s to appropriate units for SDK
            # Assuming SDK uses same units
            max_speed = self.robot_params.max_wheel_vel
            max_accel = self.robot_params.max_wheel_accel
            
            self._sdk.set_max_left_motor_speed(max_speed)
            self._sdk.set_max_right_motor_speed(max_speed)
            self._sdk.set_max_left_motor_accel(max_accel)
            self._sdk.set_max_right_motor_accel(max_accel)
            
            logger.debug(
                f"Motor limits configured: max_speed={max_speed:.2f}, "
                f"max_accel={max_accel:.2f}"
            )
        except Exception as e:
            logger.warning(f"Failed to configure motor limits: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from hardware."""
        if self._sdk is not None:
            try:
                self._sdk.stop_motors()
                self._sdk.disconnect()
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        
        self._connected = False
        logger.info("Disconnected from AetherNav board")
    
    @property
    def is_connected(self) -> bool:
        """Check if hardware is connected."""
        return self._connected
    
    def write_wheel_velocities(self, wheels: WheelVelocities) -> bool:
        """
        Send wheel velocity commands to motors.
        
        Args:
            wheels: Target wheel angular velocities (rad/s)
            
        Returns:
            True if command sent successfully
        """
        if not self._connected:
            self._rate_limited_logger.warning("Cannot write - not connected")
            return False
        
        # Convert angular velocity (rad/s) to PWM (0.0 - 1.0)
        left_pwm, right_pwm = wheels.to_pwm(self.robot_params.max_wheel_vel)
        left_pwm_normalized = left_pwm / 255.0
        right_pwm_normalized = right_pwm / 255.0
        
        # Get motor directions
        left_dir, right_dir = wheels.get_directions()
        
        # Track command
        self._last_wheel_cmd = wheels
        self._last_cmd_time = time.time()
        
        # Mock mode
        if not SDK_AVAILABLE or self._sdk is None:
            logger.debug(
                f"Mock motor cmd: L={left_pwm_normalized:.2f} dir={left_dir}, "
                f"R={right_pwm_normalized:.2f} dir={right_dir}"
            )
            return True
        
        try:
            return self._sdk.move_motors(
                left_pwm_normalized,
                right_pwm_normalized,
                left_dir=left_dir,
                right_dir=right_dir
            )
        except Exception as e:
            self._rate_limited_logger.error(f"Failed to write motor command: {e}")
            return False
    
    def read_wheel_velocities(self) -> Optional[WheelVelocities]:
        """
        Read current wheel velocities from encoders.
        
        Returns:
            WheelVelocities if available, None otherwise
        
        Note: Current SDK may not provide encoder feedback.
        Returns None if not available.
        """
        if not self._connected or not SDK_AVAILABLE or self._sdk is None:
            # In mock mode or no encoders, return last commanded velocity
            # This is open-loop estimation
            return self._last_wheel_cmd
        
        try:
            # Get velocity from SDK if available
            vel_x = self._sdk.get_velocity_x()
            if vel_x is not None:
                # SDK provides linear velocity, estimate wheel velocities
                # Assuming symmetric (straight line) motion
                wheel_angular_vel = vel_x / self.robot_params.wheel_radius
                return WheelVelocities(left=wheel_angular_vel, right=wheel_angular_vel)
            
            return None
            
        except Exception as e:
            self._rate_limited_logger.warning(f"Failed to read wheel velocities: {e}")
            return None
    
    def read_imu_yaw(self) -> Optional[float]:
        """
        Read yaw angle from IMU.
        
        Returns:
            Yaw angle in radians, or None if unavailable
        """
        if not self._connected or not SDK_AVAILABLE or self._sdk is None:
            return None
        
        try:
            angles = self._sdk.get_angles()
            if angles is not None:
                # angles is (roll, pitch, yaw) or (x, y, z) in degrees typically
                # Convert to radians
                import math
                yaw_deg = angles[2]  # Assuming Z is yaw
                return math.radians(yaw_deg)
            return None
            
        except Exception as e:
            self._rate_limited_logger.warning(f"Failed to read IMU yaw: {e}")
            return None
    
    def read_accelerometer(self) -> Optional[tuple]:
        """Read accelerometer data (ax, ay, az)."""
        if not self._connected or not SDK_AVAILABLE or self._sdk is None:
            return None
        
        try:
            return self._sdk.get_accelerometer()
        except Exception:
            return None
    
    def read_gyroscope(self) -> Optional[tuple]:
        """Read gyroscope data (gx, gy, gz)."""
        if not self._connected or not SDK_AVAILABLE or self._sdk is None:
            return None
        
        try:
            return self._sdk.get_gyroscope()
        except Exception:
            return None
    
    def emergency_stop(self) -> None:
        """Immediately stop all motors."""
        logger.warning("EMERGENCY STOP triggered")
        
        self._last_wheel_cmd = WheelVelocities()
        
        if SDK_AVAILABLE and self._sdk is not None:
            try:
                # Send stop command multiple times for reliability
                for _ in range(3):
                    self._sdk.stop_motors()
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Emergency stop failed: {e}")
    
    def get_status(self) -> dict:
        """Get hardware status for telemetry."""
        status = {
            "connected": self._connected,
            "sdk_available": SDK_AVAILABLE,
            "last_cmd_left": self._last_wheel_cmd.left,
            "last_cmd_right": self._last_wheel_cmd.right,
            "last_cmd_time": self._last_cmd_time,
        }
        
        if SDK_AVAILABLE and self._sdk is not None:
            try:
                board_status = self._sdk.get_board_status()
                if board_status:
                    status["board"] = board_status
            except Exception:
                pass
        
        return status
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.emergency_stop()
        self.disconnect()
