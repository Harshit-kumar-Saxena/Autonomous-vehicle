"""
Threaded executor for the AetherNav navigation stack.

Runs a multi-threaded pipeline:
- CameraThread: Continuous frame capture
- PerceptionThread: Segmentation inference
- ControlThread: Planning + Control at fixed rate

This is the recommended executor for production use.
"""

import signal
import sys
import time
import argparse
from pathlib import Path
from typing import Optional

from .core.types import RobotState
from .core.logging_config import configure_logging, get_logger, LogLevel
from .core.threading_utils import LatestHolder
from .core.pipeline import (
    CameraThread, 
    PerceptionThread, 
    ControlThread,
    FrameData,
    PerceptionData
)
from .config import ConfigLoader, StackConfig, RobotConfig
from .robot_model import DiffDriveKinematics, RobotParams
from .perception import CameraManager, SegmentationEngine
from .planner import LaneFollowerPlanner
from .controls import PIDController, Odometry
from .hardware import AetherNavHardwareInterface

logger = get_logger("executor")


class ThreadedExecutor:
    """
    Multi-threaded executor with pipelined processing.
    
    Architecture:
        Camera ──► FrameHolder ──► Perception ──► ResultHolder ──► Control
           │                            │                             │
         30 Hz                       ~15 Hz                        30 Hz
    
    Benefits:
    - Camera capture doesn't block inference
    - Inference doesn't block control loop
    - Always uses latest available data
    """
    
    def __init__(
        self,
        stack_config: StackConfig,
        robot_config: RobotConfig,
        dry_run: bool = False
    ):
        self.stack_config = stack_config
        self.robot_config = robot_config
        self.dry_run = dry_run or stack_config.executor.dry_run
        
        # Pipeline data holders
        self._frame_holder = LatestHolder[FrameData]()
        self._perception_holder = LatestHolder[PerceptionData]()
        
        # Thread references
        self._camera_thread: Optional[CameraThread] = None
        self._perception_thread: Optional[PerceptionThread] = None
        self._control_thread: Optional[ControlThread] = None
        
        # Initialize components
        self._init_components()
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _init_components(self) -> None:
        """Initialize all stack components."""
        # Robot model
        self.robot_params = RobotParams.from_config(self.robot_config)
        self.kinematics = DiffDriveKinematics(
            wheel_radius=self.robot_params.wheel_radius,
            track_width=self.robot_params.track_width
        )
        
        # Perception
        self.camera = CameraManager(self.stack_config.camera)
        self.segmentation = SegmentationEngine(self.stack_config.segmentation)
        
        # Planning & Control
        self.planner = LaneFollowerPlanner(self.stack_config.lane_follower)
        self.controller = PIDController(
            kinematics=self.kinematics,
            robot_params=self.robot_params,
            pid_linear=self.robot_config.pid_linear,
            pid_angular=self.robot_config.pid_angular,
            use_feedback=False
        )
        self.odometry = Odometry(kinematics=self.kinematics, use_imu_yaw=True)
        
        # Hardware
        self.hardware = AetherNavHardwareInterface(
            config=self.stack_config.hardware,
            robot_params=self.robot_params
        )
        
        logger.info(f"Components initialized:\n{self.robot_params.describe()}")
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig}, stopping...")
        self.stop()
    
    def start(self) -> bool:
        """Initialize and start all pipeline threads."""
        logger.info("Starting threaded executor...")
        
        # Initialize segmentation engine
        if not self.segmentation.initialize():
            logger.error("Failed to initialize segmentation")
            return False
        
        # Open camera
        if not self.camera.open():
            logger.error("Failed to open camera")
            return False
        
        # Connect hardware
        if not self.dry_run:
            if not self.hardware.connect():
                logger.error("Failed to connect hardware")
                return False
        else:
            logger.info("DRY RUN mode - hardware disabled")
        
        # Create and start threads
        self._camera_thread = CameraThread(
            camera=self.camera,
            output=self._frame_holder,
            target_fps=self.stack_config.camera.fps
        )
        
        self._perception_thread = PerceptionThread(
            segmentation=self.segmentation,
            input_frames=self._frame_holder,
            output=self._perception_holder
        )
        
        self._control_thread = ControlThread(
            planner=self.planner,
            controller=self.controller,
            odometry=self.odometry,
            hardware=self.hardware,
            perception_input=self._perception_holder,
            robot_params=self.robot_params,
            loop_rate_hz=self.stack_config.executor.loop_rate_hz,
            dry_run=self.dry_run
        )
        
        # Start threads
        self._camera_thread.start()
        self._perception_thread.start()
        self._control_thread.start()
        
        logger.info("All pipeline threads started")
        return True
    
    def stop(self) -> None:
        """Stop all threads and cleanup."""
        logger.info("Stopping pipeline threads...")
        
        # Stop threads in reverse order
        if self._control_thread:
            self._control_thread.stop()
        if self._perception_thread:
            self._perception_thread.stop()
        if self._camera_thread:
            self._camera_thread.stop()
        
        # Wait for threads to finish
        if self._camera_thread:
            self._camera_thread.join(timeout=2.0)
        if self._perception_thread:
            self._perception_thread.join(timeout=2.0)
        if self._control_thread:
            self._control_thread.join(timeout=2.0)
        
        # Cleanup resources
        self.camera.close()
        self.segmentation.cleanup()
        if not self.dry_run:
            self.hardware.disconnect()
        
        logger.info("Executor stopped")
    
    def run(self) -> None:
        """Run the executor until stopped."""
        if not self.start():
            logger.error("Failed to start")
            return
        
        logger.info("=" * 50)
        logger.info("   AETHERNAV THREADED PIPELINE ACTIVE")
        logger.info(f"   Camera: {self.stack_config.camera.fps} Hz")
        logger.info(f"   Control: {self.stack_config.executor.loop_rate_hz} Hz")
        logger.info(f"   Dry run: {self.dry_run}")
        logger.info("=" * 50)
        
        try:
            # Main thread just monitors and logs telemetry
            start_time = time.time()
            last_log_time = start_time
            
            while (self._control_thread and 
                   self._control_thread.is_alive()):
                time.sleep(0.5)
                
                # Log telemetry every few seconds
                now = time.time()
                if now - last_log_time >= 2.0:
                    self._log_telemetry()
                    last_log_time = now
                    
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def _log_telemetry(self) -> None:
        """Log system telemetry."""
        if self._control_thread is None:
            return
        
        state = self._control_thread.robot_state
        count = self._control_thread.control_count
        
        logger.info(
            f"[Cycles: {count}] "
            f"Pose: ({state.pose.x:.2f}, {state.pose.y:.2f}, {state.pose.theta:.2f}) | "
            f"Vel: {state.twist.linear:.2f} m/s | "
            f"Camera FPS: {self.camera.fps:.1f}"
        )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AetherNav Threaded Executor"
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="Config directory path"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run without hardware"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    configure_logging(level=getattr(LogLevel, args.log_level))
    
    # Load config
    loader = ConfigLoader(Path(args.config_dir) if args.config_dir else None)
    
    try:
        stack_config = loader.load_stack_config()
        robot_config = loader.load_robot_config()
    except Exception as e:
        logger.error(f"Config load failed: {e}")
        sys.exit(1)
    
    # Run
    executor = ThreadedExecutor(
        stack_config=stack_config,
        robot_config=robot_config,
        dry_run=args.dry_run
    )
    executor.run()


if __name__ == "__main__":
    main()
