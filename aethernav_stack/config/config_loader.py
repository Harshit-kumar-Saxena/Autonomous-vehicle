"""
Configuration loader for AetherNav stack.

Loads YAML configuration files and provides typed access to configuration values.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from pathlib import Path
import yaml

from ..core.logging_config import get_logger

logger = get_logger("config")


@dataclass
class PIDGains:
    """PID controller gains."""
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    windup_limit: float = 1.0


@dataclass
class RobotConfig:
    """Robot physical parameters and motion limits."""
    # Physical dimensions
    name: str = "AetherNav Bot"
    wheel_radius: float = 0.05      # meters
    track_width: float = 0.20       # meters
    
    # Velocity limits
    max_linear_vel: float = 0.5     # m/s
    max_angular_vel: float = 1.0    # rad/s
    max_wheel_vel: float = 10.0     # rad/s
    max_wheel_accel: float = 5.0    # rad/s²
    
    # Controller settings
    controller_type: str = "pid"
    pid_linear: PIDGains = field(default_factory=PIDGains)
    pid_angular: PIDGains = field(default_factory=PIDGains)


@dataclass
class CameraConfig:
    """Camera capture settings."""
    device_id: int = 2
    width: int = 640
    height: int = 480
    fps: int = 30
    backend: str = "v4l2"


@dataclass
class SegmentationConfig:
    """Segmentation model settings."""
    engine_path: str = ""
    input_width: int = 512
    input_height: int = 256
    roi_start_ratio: float = 0.7
    confidence_threshold: float = 0.4
    min_lane_pixels: int = 100


@dataclass
class LaneFollowerConfig:
    """Lane following planner settings."""
    cruise_linear_vel: float = 0.3
    min_linear_vel: float = 0.1
    deadband_min: float = 0.42
    deadband_max: float = 0.58
    angular_gain: float = 2.0
    slow_down_on_curve: bool = True
    curve_slow_factor: float = 0.5
    recovery_enabled: bool = True
    hold_last_command_frames: int = 10
    stop_after_lost_frames: int = 30


@dataclass
class HardwareConfig:
    """Hardware interface settings."""
    port: str = "/dev/ttyACM0"
    baudrate: int = 500000
    timeout: float = 1.0
    auto_reconnect: bool = True


@dataclass
class ExecutorConfig:
    """Main executor settings."""
    loop_rate_hz: int = 30
    enable_telemetry: bool = True
    telemetry_log_interval: int = 10
    dry_run: bool = False


@dataclass
class StackConfig:
    """Complete stack configuration."""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    lane_follower: LaneFollowerConfig = field(default_factory=LaneFollowerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    log_level: str = "INFO"


class ConfigLoader:
    """Loads and parses YAML configuration files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config loader.
        
        Args:
            config_dir: Directory containing config files. Defaults to
                        the 'config' directory in the package.
        """
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = Path(config_dir)
    
    def load_robot_config(self, filename: str = "robot_config.yaml") -> RobotConfig:
        """Load robot configuration."""
        config_path = self.config_dir / filename
        data = self._load_yaml(config_path)
        
        robot_data = data.get("robot", {})
        controller_data = data.get("controller", {})
        
        # Parse PID gains
        pid_data = controller_data.get("pid", {})
        pid_linear = PIDGains(**pid_data.get("linear", {}))
        pid_angular = PIDGains(**pid_data.get("angular", {}))
        
        return RobotConfig(
            name=robot_data.get("name", "AetherNav Bot"),
            wheel_radius=robot_data.get("wheel_radius", 0.05),
            track_width=robot_data.get("track_width", 0.20),
            max_linear_vel=robot_data.get("max_linear_vel", 0.5),
            max_angular_vel=robot_data.get("max_angular_vel", 1.0),
            max_wheel_vel=robot_data.get("max_wheel_vel", 10.0),
            max_wheel_accel=robot_data.get("max_wheel_accel", 5.0),
            controller_type=controller_data.get("type", "pid"),
            pid_linear=pid_linear,
            pid_angular=pid_angular,
        )
    
    def load_stack_config(self, filename: str = "stack_config.yaml") -> StackConfig:
        """Load stack configuration."""
        config_path = self.config_dir / filename
        data = self._load_yaml(config_path)
        
        # Parse hardware config
        hw_data = data.get("hardware", {})
        hardware = HardwareConfig(
            port=hw_data.get("port", "/dev/ttyACM0"),
            baudrate=hw_data.get("baudrate", 500000),
            timeout=hw_data.get("timeout", 1.0),
            auto_reconnect=hw_data.get("auto_reconnect", True),
        )
        
        # Parse camera config
        cam_data = data.get("camera", {})
        camera = CameraConfig(
            device_id=cam_data.get("device_id", 2),
            width=cam_data.get("width", 640),
            height=cam_data.get("height", 480),
            fps=cam_data.get("fps", 30),
            backend=cam_data.get("backend", "v4l2"),
        )
        
        # Parse segmentation config
        seg_data = data.get("segmentation", {})
        segmentation = SegmentationConfig(
            engine_path=seg_data.get("engine_path", ""),
            input_width=seg_data.get("input_width", 512),
            input_height=seg_data.get("input_height", 256),
            roi_start_ratio=seg_data.get("roi_start_ratio", 0.7),
            confidence_threshold=seg_data.get("confidence_threshold", 0.4),
            min_lane_pixels=seg_data.get("min_lane_pixels", 100),
        )
        
        # Parse lane follower config
        planner_data = data.get("planner", {})
        lf_data = planner_data.get("lane_follower", {})
        recovery_data = lf_data.get("recovery", {})
        lane_follower = LaneFollowerConfig(
            cruise_linear_vel=lf_data.get("cruise_linear_vel", 0.3),
            min_linear_vel=lf_data.get("min_linear_vel", 0.1),
            deadband_min=lf_data.get("deadband_min", 0.42),
            deadband_max=lf_data.get("deadband_max", 0.58),
            angular_gain=lf_data.get("angular_gain", 2.0),
            slow_down_on_curve=lf_data.get("slow_down_on_curve", True),
            curve_slow_factor=lf_data.get("curve_slow_factor", 0.5),
            recovery_enabled=recovery_data.get("enabled", True),
            hold_last_command_frames=recovery_data.get("hold_last_command_frames", 10),
            stop_after_lost_frames=recovery_data.get("stop_after_lost_frames", 30),
        )
        
        # Parse executor config
        exec_data = data.get("executor", {})
        executor = ExecutorConfig(
            loop_rate_hz=exec_data.get("loop_rate_hz", 30),
            enable_telemetry=exec_data.get("enable_telemetry", True),
            telemetry_log_interval=exec_data.get("telemetry_log_interval", 10),
            dry_run=exec_data.get("dry_run", False),
        )
        
        # Parse logging
        log_data = data.get("logging", {})
        log_level = log_data.get("level", "INFO")
        
        return StackConfig(
            hardware=hardware,
            camera=camera,
            segmentation=segmentation,
            lane_follower=lane_follower,
            executor=executor,
            log_level=log_level,
        )
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file and return its contents as a dictionary."""
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return {}
        
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                return data if data else {}
        except Exception as e:
            logger.error(f"Failed to load config from {path}: {e}")
            return {}
