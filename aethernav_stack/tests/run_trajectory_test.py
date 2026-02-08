#!/usr/bin/env python3
"""
Trajectory Test Runner for AetherNav Stack.

Runs the stack with trajectory logging enabled, then generates
analysis and visualizations for smoothness optimization.

Usage:
    python -m aethernav_stack.tests.run_trajectory_test [--duration 30]
"""

import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aethernav_stack.config import ConfigLoader
from aethernav_stack.robot_model import DiffDriveKinematics, RobotParams
from aethernav_stack.perception import CameraManager, SegmentationEngine
from aethernav_stack.planner import LaneFollowerPlanner
from aethernav_stack.controls import PIDController, Odometry
from aethernav_stack.hardware import AetherNavHardwareInterface
from aethernav_stack.core.types import RobotState
from aethernav_stack.core.logging_config import configure_logging, get_logger, LogLevel

from aethernav_stack.tests.trajectory_logger import TrajectoryLogger
from aethernav_stack.tests.trajectory_analyzer import TrajectoryAnalyzer

logger = get_logger("trajectory_test")


class TrajectoryTestRunner:
    """
    Runs the stack with trajectory logging and generates analysis.
    
    Uses single-threaded execution for precise logging.
    """
    
    def __init__(self, duration: float = 30.0, show_viz: bool = True):
        """
        Initialize test runner.
        
        Args:
            duration: Test duration in seconds
            show_viz: Show visualization window during test
        """
        self.duration = duration
        self.show_viz = show_viz
        self._running = False
        
        # Load config
        self.loader = ConfigLoader()
        self.stack_config = self.loader.load_stack_config()
        self.robot_config = self.loader.load_robot_config()
        
        # Initialize components
        self._init_components()
        
        # Trajectory logger
        self.traj_logger = TrajectoryLogger()
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
    
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
        
        # Hardware (mock mode)
        self.hardware = AetherNavHardwareInterface(
            config=self.stack_config.hardware,
            robot_params=self.robot_params
        )
    
    def _signal_handler(self, sig, frame) -> None:
        """Handle Ctrl+C."""
        logger.info("Stopping test...")
        self._running = False
    
    def run(self) -> Path:
        """
        Run the trajectory test.
        
        Returns:
            Path to saved trajectory CSV file
        """
        logger.info("=" * 50)
        logger.info("   TRAJECTORY TEST STARTING")
        logger.info(f"   Duration: {self.duration}s")
        logger.info("=" * 50)
        
        # Initialize
        if not self.segmentation.initialize():
            logger.error("Failed to initialize segmentation")
            return None
        
        if not self.camera.open():
            logger.error("Failed to open camera")
            return None
        
        self.hardware.connect()  # Will use mock mode
        
        # Start logging
        self.traj_logger.start()
        self._running = True
        
        robot_state = RobotState()
        start_time = time.time()
        last_time = start_time
        frame_count = 0
        
        try:
            while self._running and (time.time() - start_time) < self.duration:
                loop_start = time.time()
                
                # 1. Capture frame
                frame = self.camera.read()
                if frame is None or not frame.is_valid:
                    continue
                
                # 2. Run segmentation
                seg_result = self.segmentation.infer(frame.image)
                
                # 3. Compute dt
                now = time.time()
                dt = now - last_time
                last_time = now
                
                # 4. Run planner
                cmd_vel = self.planner.compute_velocity(seg_result, robot_state)
                
                # 5. Run controller
                wheel_vel = self.controller.compute_wheel_velocities(
                    cmd_vel, robot_state, dt
                )
                
                # 6. Log trajectory point
                self.traj_logger.log(cmd_vel, wheel_vel, robot_state, seg_result)
                
                # 7. Update odometry (assuming perfect execution)
                robot_state = self.odometry.update(
                    wheel_velocities=wheel_vel,
                    imu_yaw=None
                )
                robot_state.wheel_velocities = wheel_vel
                
                # 8. Show visualization if enabled
                if self.show_viz:
                    self._show_frame(frame.image, seg_result)
                
                frame_count += 1
                
                # Rate limiting
                elapsed = time.time() - loop_start
                sleep_time = (1.0 / 30) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Progress log
                if frame_count % 90 == 0:
                    elapsed_total = time.time() - start_time
                    logger.info(f"Progress: {elapsed_total:.1f}s / {self.duration:.1f}s")
        
        finally:
            self._running = False
            self.traj_logger.stop()
            
            # Cleanup
            self.camera.close()
            self.segmentation.cleanup()
            self.hardware.disconnect()
            
            if self.show_viz:
                import cv2
                cv2.destroyAllWindows()
        
        # Save trajectory
        csv_path = self.traj_logger.save_csv()
        logger.info(f"Trajectory saved: {csv_path}")
        
        return csv_path
    
    def _show_frame(self, frame, seg_result) -> None:
        """Show visualization during test."""
        import cv2
        import numpy as np
        
        vis = frame.copy()
        h, w = vis.shape[:2]
        
        # Overlay mask
        mask_resized = cv2.resize(
            (seg_result.mask * 255).astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST
        )
        overlay = np.zeros_like(vis)
        overlay[:, :, 1] = mask_resized
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)
        
        # Draw centerline
        if len(seg_result.centerline_points) > 1:
            scale_x = w / seg_result.mask.shape[1]
            scale_y = h / seg_result.mask.shape[0]
            pts = [(int(p[0] * scale_x), int(p[1] * scale_y)) 
                   for p in seg_result.centerline_points]
            for i in range(len(pts) - 1):
                cv2.line(vis, pts[i], pts[i+1], (0, 0, 255), 2)
        
        # Info text
        text = f"Lane: {'YES' if seg_result.lane_detected else 'NO'} | Conf: {seg_result.confidence:.2f}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow("Trajectory Test", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self._running = False


def analyze_trajectory(csv_path: Path, robot_params: RobotParams) -> None:
    """Run full trajectory analysis."""
    logger.info("=" * 50)
    logger.info("   ANALYZING TRAJECTORY")
    logger.info("=" * 50)
    
    # Load trajectory
    traj = TrajectoryLogger.load_csv(csv_path)
    
    # Create analyzer
    analyzer = TrajectoryAnalyzer(
        traj,
        wheel_radius=robot_params.wheel_radius,
        track_width=robot_params.track_width
    )
    
    # Compute metrics
    metrics = analyzer.compute_metrics()
    print("\n" + str(metrics))
    
    # Generate plots
    logs_dir = csv_path.parent
    
    analyzer.plot_trajectory(logs_dir / "trajectory_path.png")
    analyzer.plot_velocity_profile(logs_dir / "velocity_profile.png")
    analyzer.plot_lane_tracking(logs_dir / "lane_tracking.png")
    
    # Save report
    analyzer.save_report(logs_dir / "smoothness_report.txt")
    
    logger.info(f"\nAll outputs saved to: {logs_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AetherNav Trajectory Test")
    parser.add_argument(
        "--duration", type=float, default=30.0,
        help="Test duration in seconds (default: 30)"
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Disable visualization during test"
    )
    parser.add_argument(
        "--analyze-only", type=str, default=None,
        help="Only analyze existing CSV file (skip test run)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    
    args = parser.parse_args()
    
    configure_logging(level=getattr(LogLevel, args.log_level))
    
    # Load robot params for analysis
    loader = ConfigLoader()
    robot_config = loader.load_robot_config()
    robot_params = RobotParams.from_config(robot_config)
    
    if args.analyze_only:
        # Just run analysis on existing file
        csv_path = Path(args.analyze_only)
        if not csv_path.exists():
            logger.error(f"File not found: {csv_path}")
            sys.exit(1)
        analyze_trajectory(csv_path, robot_params)
    else:
        # Run test then analyze
        runner = TrajectoryTestRunner(
            duration=args.duration,
            show_viz=not args.no_viz
        )
        
        csv_path = runner.run()
        
        if csv_path and csv_path.exists():
            analyze_trajectory(csv_path, robot_params)


if __name__ == "__main__":
    main()
