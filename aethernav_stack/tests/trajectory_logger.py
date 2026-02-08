"""
Trajectory Logger for AetherNav Stack.

Logs motion commands, wheel velocities, and perception data during execution
for post-run trajectory analysis and smoothness optimization.
"""

import csv
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple
from datetime import datetime

from ..core.types import WheelVelocities, Twist2D, SegmentationResult, RobotState
from ..core.logging_config import get_logger

logger = get_logger("trajectory_logger")


@dataclass
class TrajectoryPoint:
    """Single logged data point."""
    timestamp: float
    dt: float  # Time since last point
    
    # Commanded velocities
    cmd_linear: float
    cmd_angular: float
    
    # Wheel velocities (rad/s)
    wheel_left: float
    wheel_right: float
    
    # Odometry pose (reconstructed)
    pose_x: float
    pose_y: float
    pose_theta: float
    
    # Perception data
    lane_detected: bool
    lane_center: float  # Normalized 0-1
    confidence: float
    
    # Optional: centerline point count
    centerline_points: int = 0


class TrajectoryLogger:
    """
    Logs motion commands and state during execution.
    
    Usage:
        logger = TrajectoryLogger()
        logger.start()
        
        # In control loop:
        logger.log(cmd_vel, wheel_vel, robot_state, seg_result)
        
        # After run:
        logger.stop()
        logger.save("trajectory.csv")
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize trajectory logger.
        
        Args:
            output_dir: Directory to save logs (default: aethernav_stack/logs)
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "logs"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._points: List[TrajectoryPoint] = []
        self._start_time: float = 0.0
        self._last_time: float = 0.0
        self._running = False
    
    def start(self) -> None:
        """Start logging."""
        self._points = []
        self._start_time = time.time()
        self._last_time = self._start_time
        self._running = True
        logger.info("Trajectory logging started")
    
    def stop(self) -> None:
        """Stop logging."""
        self._running = False
        duration = time.time() - self._start_time
        logger.info(f"Trajectory logging stopped: {len(self._points)} points, {duration:.1f}s")
    
    def log(
        self,
        cmd_vel: Twist2D,
        wheel_vel: WheelVelocities,
        robot_state: RobotState,
        seg_result: Optional[SegmentationResult] = None
    ) -> None:
        """
        Log a single data point.
        
        Args:
            cmd_vel: Commanded twist velocity
            wheel_vel: Commanded wheel velocities
            robot_state: Current robot state (with odometry)
            seg_result: Optional segmentation result
        """
        if not self._running:
            return
        
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        
        # Extract perception data
        if seg_result:
            lane_detected = seg_result.lane_detected
            lane_center = seg_result.lane_center_normalized
            confidence = seg_result.confidence
            centerline_count = len(seg_result.centerline_points)
        else:
            lane_detected = False
            lane_center = 0.5
            confidence = 0.0
            centerline_count = 0
        
        point = TrajectoryPoint(
            timestamp=now - self._start_time,
            dt=dt,
            cmd_linear=cmd_vel.linear,
            cmd_angular=cmd_vel.angular,
            wheel_left=wheel_vel.left,
            wheel_right=wheel_vel.right,
            pose_x=robot_state.pose.x,
            pose_y=robot_state.pose.y,
            pose_theta=robot_state.pose.theta,
            lane_detected=lane_detected,
            lane_center=lane_center,
            confidence=confidence,
            centerline_points=centerline_count,
        )
        
        self._points.append(point)
    
    @property
    def points(self) -> List[TrajectoryPoint]:
        """Get all logged points."""
        return self._points
    
    @property
    def duration(self) -> float:
        """Get total logging duration."""
        if len(self._points) == 0:
            return 0.0
        return self._points[-1].timestamp
    
    def save_csv(self, filename: Optional[str] = None) -> Path:
        """
        Save trajectory to CSV file.
        
        Args:
            filename: Optional filename (default: trajectory_YYYYMMDD_HHMMSS.csv)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.csv"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            if len(self._points) == 0:
                logger.warning("No points to save")
                return filepath
            
            writer = csv.DictWriter(f, fieldnames=asdict(self._points[0]).keys())
            writer.writeheader()
            
            for point in self._points:
                writer.writerow(asdict(point))
        
        logger.info(f"Trajectory saved to {filepath}")
        return filepath
    
    def save_json(self, filename: Optional[str] = None) -> Path:
        """Save trajectory to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trajectory_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        data = {
            "metadata": {
                "start_time": self._start_time,
                "duration": self.duration,
                "point_count": len(self._points),
            },
            "points": [asdict(p) for p in self._points]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Trajectory saved to {filepath}")
        return filepath
    
    @classmethod
    def load_csv(cls, filepath: Path) -> "TrajectoryLogger":
        """Load trajectory from CSV file."""
        instance = cls()
        
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                point = TrajectoryPoint(
                    timestamp=float(row['timestamp']),
                    dt=float(row['dt']),
                    cmd_linear=float(row['cmd_linear']),
                    cmd_angular=float(row['cmd_angular']),
                    wheel_left=float(row['wheel_left']),
                    wheel_right=float(row['wheel_right']),
                    pose_x=float(row['pose_x']),
                    pose_y=float(row['pose_y']),
                    pose_theta=float(row['pose_theta']),
                    lane_detected=row['lane_detected'].lower() == 'true',
                    lane_center=float(row['lane_center']),
                    confidence=float(row['confidence']),
                    centerline_points=int(row['centerline_points']),
                )
                instance._points.append(point)
        
        logger.info(f"Loaded {len(instance._points)} points from {filepath}")
        return instance
