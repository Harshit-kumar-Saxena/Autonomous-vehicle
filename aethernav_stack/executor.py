"""
Main executor for the AetherNav navigation stack.

Orchestrates all layers of the navigation stack:
1. Capture camera frame
2. Run segmentation
3. Compute planner output (Twist2D)
4. Run controller → WheelVelocities
5. Update odometry
6. Write to hardware
7. Log telemetry
"""

import signal
import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from .core.types import RobotState, WheelVelocities
from .core.logging_config import configure_logging, get_logger, LogLevel
from .config import ConfigLoader, StackConfig, RobotConfig
from .robot_model import DiffDriveKinematics, RobotParams
from .perception import CameraManager, SegmentationEngine
from .planner import LaneFollowerPlanner
from .controls import PIDController, Odometry
from .hardware import AetherNavHardwareInterface

logger = get_logger("executor")


@dataclass
class ExecutorStats:
    """Runtime statistics for the executor."""
    frame_count: int = 0
    start_time: float = 0.0
    last_loop_time: float = 0.0
    avg_loop_time_ms: float = 0.0
    
    def update(self, loop_time: float) -> None:
        """Update statistics with latest loop time."""
        self.frame_count += 1
        # Exponential moving average
        alpha = 0.1
        self.avg_loop_time_ms = (
            alpha * (loop_time * 1000) + 
            (1 - alpha) * self.avg_loop_time_ms
        )


class Executor:
    """
    Main control loop orchestrator.
    
    Initializes all layers and runs the main perception-planning-control
    loop at the configured rate.
    """
    
    def __init__(
        self,
        stack_config: StackConfig,
        robot_config: RobotConfig,
        dry_run: bool = False
    ):
        """
        Initialize executor with configuration.
        
        Args:
            stack_config: System configuration
            robot_config: Robot configuration
            dry_run: If True, don't send commands to hardware
        """
        self.stack_config = stack_config
        self.robot_config = robot_config
        self.dry_run = dry_run or stack_config.executor.dry_run
        
        # State
        self._running = False
        self._stats = ExecutorStats()
        self._robot_state = RobotState()
        
        # Initialize layers
        self._init_robot_model()
        self._init_perception()
        self._init_planner()
        self._init_controls()
        self._init_hardware()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_robot_model(self) -> None:
        """Initialize robot model and kinematics."""
        self.robot_params = RobotParams.from_config(self.robot_config)
        self.kinematics = DiffDriveKinematics(
            wheel_radius=self.robot_params.wheel_radius,
            track_width=self.robot_params.track_width
        )
        logger.info(f"Robot model initialized:\n{self.robot_params.describe()}")
    
    def _init_perception(self) -> None:
        """Initialize camera and segmentation."""
        self.camera = CameraManager(self.stack_config.camera)
        self.segmentation = SegmentationEngine(self.stack_config.segmentation)
    
    def _init_planner(self) -> None:
        """Initialize motion planner."""
        self.planner = LaneFollowerPlanner(self.stack_config.lane_follower)
        logger.info(f"Planner initialized: {self.planner.name}")
    
    def _init_controls(self) -> None:
        """Initialize controller and odometry."""
        self.controller = PIDController(
            kinematics=self.kinematics,
            robot_params=self.robot_params,
            pid_linear=self.robot_config.pid_linear,
            pid_angular=self.robot_config.pid_angular,
            use_feedback=False  # No encoder feedback by default
        )
        
        self.odometry = Odometry(
            kinematics=self.kinematics,
            use_imu_yaw=True
        )
        logger.info(f"Controller initialized: {self.controller.name}")
    
    def _init_hardware(self) -> None:
        """Initialize hardware interface."""
        self.hardware = AetherNavHardwareInterface(
            config=self.stack_config.hardware,
            robot_params=self.robot_params
        )
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self._running = False
    
    def start(self) -> bool:
        """
        Start the executor (initialize hardware and perception).
        
        Returns:
            True if startup successful
        """
        logger.info("Starting AetherNav executor...")
        
        # Initialize segmentation engine
        if not self.segmentation.initialize():
            logger.error("Failed to initialize segmentation engine")
            return False
        
        # Open camera
        if not self.camera.open():
            logger.error("Failed to open camera")
            return False
        
        # Connect to hardware
        if not self.dry_run:
            if not self.hardware.connect():
                logger.error("Failed to connect to hardware")
                return False
        else:
            logger.info("DRY RUN mode - hardware commands will not be sent")
        
        self._stats.start_time = time.time()
        logger.info("Executor started successfully")
        return True
    
    def stop(self) -> None:
        """Stop the executor and cleanup resources."""
        logger.info("Stopping executor...")
        
        # Stop motors
        if not self.dry_run:
            self.hardware.emergency_stop()
            self.hardware.disconnect()
        
        # Release camera
        self.camera.close()
        
        # Cleanup segmentation
        self.segmentation.cleanup()
        
        logger.info("Executor stopped")
    
    def run(self) -> None:
        """
        Run the main control loop.
        
        Loops at the configured rate until stopped.
        """
        if not self.start():
            logger.error("Failed to start executor")
            return
        
        self._running = True
        loop_period = 1.0 / self.stack_config.executor.loop_rate_hz
        
        logger.info("=" * 50)
        logger.info("   AETHERNAV AUTONOMOUS MODE ACTIVE")
        logger.info(f"   Loop rate: {self.stack_config.executor.loop_rate_hz} Hz")
        logger.info(f"   Dry run: {self.dry_run}")
        logger.info("=" * 50)
        
        try:
            while self._running:
                loop_start = time.time()
                
                # Execute one control cycle
                self._control_cycle()
                
                # Calculate timing
                loop_time = time.time() - loop_start
                self._stats.update(loop_time)
                
                # Log telemetry periodically
                if (self._stats.frame_count % 
                    self.stack_config.executor.telemetry_log_interval == 0):
                    self._log_telemetry()
                
                # Sleep to maintain loop rate
                sleep_time = loop_period - loop_time
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
        except Exception as e:
            logger.error(f"Exception in control loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _control_cycle(self) -> None:
        """Execute one iteration of the control loop."""
        # 1. Capture frame
        frame = self.camera.read()
        if frame is None or not frame.is_valid:
            return
        
        # 2. Run segmentation
        seg_result = self.segmentation.infer(frame.image)
        
        # 3. Compute planner output
        cmd_vel = self.planner.compute_velocity(seg_result, self._robot_state)
        
        # 4. Run controller
        dt = time.time() - self._stats.last_loop_time if self._stats.last_loop_time > 0 else 0.033
        self._stats.last_loop_time = time.time()
        
        wheel_velocities = self.controller.compute_wheel_velocities(
            cmd_vel, 
            self._robot_state,
            dt
        )
        
        # 5. Write to hardware
        if not self.dry_run:
            self.hardware.write_wheel_velocities(wheel_velocities)
        
        # 6. Update odometry
        # Read feedback from hardware if available
        feedback_wheels = self.hardware.read_wheel_velocities() if not self.dry_run else wheel_velocities
        imu_yaw = self.hardware.read_imu_yaw() if not self.dry_run else None
        
        self._robot_state = self.odometry.update(
            wheel_velocities=feedback_wheels or wheel_velocities,
            imu_yaw=imu_yaw
        )
        
        # Update robot state with commanded velocities
        self._robot_state.wheel_velocities = wheel_velocities
    
    def _log_telemetry(self) -> None:
        """Log telemetry data."""
        fps = self.camera.fps
        odom = self.odometry.get_status()
        planner_status = self.planner.get_status()
        
        logger.info(
            f"[Frame {self._stats.frame_count}] "
            f"FPS: {fps:.1f} | "
            f"Loop: {self._stats.avg_loop_time_ms:.1f}ms | "
            f"Pose: ({odom['x']:.2f}, {odom['y']:.2f}, {odom['theta']:.2f}) | "
            f"V: {odom['linear_vel']:.2f} m/s | "
            f"Lane: {planner_status['consecutive_detected']} frames"
        )


def main():
    """Entry point for the executor."""
    parser = argparse.ArgumentParser(
        description="AetherNav Autonomous Vehicle Stack"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Path to configuration directory"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without sending commands to hardware"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = getattr(LogLevel, args.log_level)
    configure_logging(level=log_level)
    
    # Load configuration
    config_dir = Path(args.config_dir) if args.config_dir else None
    loader = ConfigLoader(config_dir)
    
    try:
        stack_config = loader.load_stack_config()
        robot_config = loader.load_robot_config()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Create and run executor
    executor = Executor(
        stack_config=stack_config,
        robot_config=robot_config,
        dry_run=args.dry_run
    )
    
    executor.run()


if __name__ == "__main__":
    main()
