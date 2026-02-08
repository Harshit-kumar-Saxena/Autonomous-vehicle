"""
Trajectory Analyzer for AetherNav Stack.

Reconstructs robot path from logged commands and computes smoothness metrics
for optimization analysis.
"""

import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

from ..core.logging_config import get_logger
from .trajectory_logger import TrajectoryLogger, TrajectoryPoint

logger = get_logger("trajectory_analyzer")


@dataclass
class SmoothnessMetrics:
    """Trajectory smoothness metrics."""
    # Jerk metrics (rate of acceleration change)
    mean_linear_jerk: float  # m/s³
    max_linear_jerk: float
    mean_angular_jerk: float  # rad/s³
    max_angular_jerk: float
    
    # Velocity variance
    linear_vel_variance: float
    angular_vel_variance: float
    
    # Oscillation detection
    angular_zero_crossings: int  # Sign changes in angular velocity
    oscillation_frequency: float  # Hz
    
    # Curvature
    mean_curvature: float  # 1/m
    max_curvature: float
    curvature_rate_mean: float  # 1/m²
    
    # Overall score (lower is smoother)
    smoothness_score: float
    
    def __str__(self) -> str:
        return (
            f"=== Smoothness Report ===\n"
            f"Linear Jerk:   mean={self.mean_linear_jerk:.3f}, max={self.max_linear_jerk:.3f} m/s³\n"
            f"Angular Jerk:  mean={self.mean_angular_jerk:.3f}, max={self.max_angular_jerk:.3f} rad/s³\n"
            f"Linear Vel σ²: {self.linear_vel_variance:.4f}\n"
            f"Angular Vel σ²: {self.angular_vel_variance:.4f}\n"
            f"Oscillations:  {self.angular_zero_crossings} zero-crossings, {self.oscillation_frequency:.2f} Hz\n"
            f"Curvature:     mean={self.mean_curvature:.3f}, max={self.max_curvature:.3f} 1/m\n"
            f"Curvature Rate: {self.curvature_rate_mean:.3f} 1/m²\n"
            f"──────────────────────────\n"
            f"Smoothness Score: {self.smoothness_score:.2f} (lower is better)\n"
        )


class TrajectoryAnalyzer:
    """
    Analyzes trajectory for smoothness and generates visualizations.
    
    Usage:
        analyzer = TrajectoryAnalyzer(trajectory_logger)
        metrics = analyzer.compute_metrics()
        print(metrics)
        analyzer.plot_trajectory("path.png")
        analyzer.plot_velocity_profile("velocity.png")
    """
    
    def __init__(
        self, 
        trajectory: TrajectoryLogger,
        wheel_radius: float = 0.055,
        track_width: float = 0.20
    ):
        """
        Initialize analyzer with trajectory data.
        
        Args:
            trajectory: TrajectoryLogger with recorded data
            wheel_radius: Robot wheel radius (m)
            track_width: Distance between wheels (m)
        """
        self.trajectory = trajectory
        self.wheel_radius = wheel_radius
        self.track_width = track_width
        
        self._points = trajectory.points
        self._metrics: Optional[SmoothnessMetrics] = None
        
        # Computed arrays
        self._timestamps: np.ndarray = None
        self._linear_vel: np.ndarray = None
        self._angular_vel: np.ndarray = None
        self._linear_accel: np.ndarray = None
        self._angular_accel: np.ndarray = None
        self._linear_jerk: np.ndarray = None
        self._angular_jerk: np.ndarray = None
        self._poses: np.ndarray = None  # Nx3 (x, y, theta)
        
        self._compute_derivatives()
    
    def _compute_derivatives(self) -> None:
        """Compute velocity, acceleration, and jerk from logged data."""
        if len(self._points) < 3:
            logger.warning("Not enough points for analysis")
            return
        
        n = len(self._points)
        
        # Extract arrays
        self._timestamps = np.array([p.timestamp for p in self._points])
        self._linear_vel = np.array([p.cmd_linear for p in self._points])
        self._angular_vel = np.array([p.cmd_angular for p in self._points])
        
        # Compute dt array
        dt = np.diff(self._timestamps)
        dt = np.maximum(dt, 1e-6)  # Avoid division by zero
        
        # Acceleration (first derivative of velocity)
        self._linear_accel = np.diff(self._linear_vel) / dt
        self._angular_accel = np.diff(self._angular_vel) / dt
        
        # Jerk (second derivative of velocity)
        if len(dt) > 1:
            dt2 = dt[:-1]
            self._linear_jerk = np.diff(self._linear_accel) / dt2
            self._angular_jerk = np.diff(self._angular_accel) / dt2
        else:
            self._linear_jerk = np.array([0])
            self._angular_jerk = np.array([0])
        
        # Extract odometry poses
        self._poses = np.array([
            [p.pose_x, p.pose_y, p.pose_theta] 
            for p in self._points
        ])
    
    def compute_metrics(self) -> SmoothnessMetrics:
        """Compute all smoothness metrics."""
        if len(self._points) < 3:
            logger.warning("Not enough points for metrics")
            return SmoothnessMetrics(
                mean_linear_jerk=0, max_linear_jerk=0,
                mean_angular_jerk=0, max_angular_jerk=0,
                linear_vel_variance=0, angular_vel_variance=0,
                angular_zero_crossings=0, oscillation_frequency=0,
                mean_curvature=0, max_curvature=0, curvature_rate_mean=0,
                smoothness_score=0
            )
        
        # Jerk metrics
        mean_linear_jerk = np.mean(np.abs(self._linear_jerk))
        max_linear_jerk = np.max(np.abs(self._linear_jerk))
        mean_angular_jerk = np.mean(np.abs(self._angular_jerk))
        max_angular_jerk = np.max(np.abs(self._angular_jerk))
        
        # Velocity variance
        linear_vel_variance = np.var(self._linear_vel)
        angular_vel_variance = np.var(self._angular_vel)
        
        # Oscillation detection (zero crossings in angular velocity)
        angular_sign = np.sign(self._angular_vel)
        zero_crossings = np.sum(np.abs(np.diff(angular_sign)) > 0)
        duration = self._timestamps[-1] - self._timestamps[0]
        oscillation_freq = zero_crossings / (2 * duration) if duration > 0 else 0
        
        # Curvature: κ = ω / v
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.where(
                np.abs(self._linear_vel) > 0.01,
                self._angular_vel / self._linear_vel,
                0
            )
        mean_curvature = np.mean(np.abs(curvature))
        max_curvature = np.max(np.abs(curvature))
        
        # Curvature rate: dκ/dt
        dt = np.diff(self._timestamps)
        dt = np.maximum(dt, 1e-6)
        curvature_rate = np.diff(curvature) / dt
        curvature_rate_mean = np.mean(np.abs(curvature_rate))
        
        # Compute overall smoothness score (weighted sum)
        smoothness_score = (
            1.0 * mean_linear_jerk +
            0.5 * mean_angular_jerk +
            2.0 * oscillation_freq +
            0.5 * curvature_rate_mean
        )
        
        self._metrics = SmoothnessMetrics(
            mean_linear_jerk=mean_linear_jerk,
            max_linear_jerk=max_linear_jerk,
            mean_angular_jerk=mean_angular_jerk,
            max_angular_jerk=max_angular_jerk,
            linear_vel_variance=linear_vel_variance,
            angular_vel_variance=angular_vel_variance,
            angular_zero_crossings=zero_crossings,
            oscillation_frequency=oscillation_freq,
            mean_curvature=mean_curvature,
            max_curvature=max_curvature,
            curvature_rate_mean=curvature_rate_mean,
            smoothness_score=smoothness_score
        )
        
        return self._metrics
    
    def reconstruct_path_from_wheels(self) -> np.ndarray:
        """
        Reconstruct path using forward kinematics from wheel velocities.
        
        This is useful to verify odometry or compute path from raw wheel commands.
        
        Returns:
            Nx3 array of (x, y, theta) poses
        """
        poses = [[0.0, 0.0, 0.0]]  # Start at origin
        
        for i, point in enumerate(self._points[1:], 1):
            dt = point.dt
            prev = poses[-1]
            
            # Differential drive forward kinematics
            v_left = point.wheel_left * self.wheel_radius
            v_right = point.wheel_right * self.wheel_radius
            
            v = (v_left + v_right) / 2.0
            omega = (v_right - v_left) / self.track_width
            
            theta = prev[2]
            
            if abs(omega) < 1e-6:
                # Straight line
                x_new = prev[0] + v * math.cos(theta) * dt
                y_new = prev[1] + v * math.sin(theta) * dt
                theta_new = theta
            else:
                # Arc motion
                r = v / omega
                theta_new = theta + omega * dt
                x_new = prev[0] + r * (math.sin(theta_new) - math.sin(theta))
                y_new = prev[1] - r * (math.cos(theta_new) - math.cos(theta))
            
            poses.append([x_new, y_new, theta_new])
        
        return np.array(poses)
    
    def plot_trajectory(self, output_path: Optional[Path] = None) -> None:
        """
        Plot XY trajectory with direction indicators.
        
        Args:
            output_path: Path to save plot (None = show interactively)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Use odometry poses
        x = self._poses[:, 0]
        y = self._poses[:, 1]
        
        # Color by time
        colors = self._timestamps
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=5)
        plt.colorbar(scatter, label='Time (s)')
        
        # Draw direction arrows every N points
        step = max(1, len(x) // 30)
        for i in range(0, len(x), step):
            theta = self._poses[i, 2]
            ax.arrow(
                x[i], y[i],
                0.05 * math.cos(theta), 0.05 * math.sin(theta),
                head_width=0.02, head_length=0.01, fc='red', ec='red'
            )
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Robot Trajectory')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Mark start and end
        ax.plot(x[0], y[0], 'go', markersize=10, label='Start')
        ax.plot(x[-1], y[-1], 'ro', markersize=10, label='End')
        ax.legend()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Trajectory plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_velocity_profile(self, output_path: Optional[Path] = None) -> None:
        """
        Plot velocity, acceleration, and jerk profiles over time.
        
        Args:
            output_path: Path to save plot (None = show interactively)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        
        t = self._timestamps
        t_accel = t[:-1]
        t_jerk = t[:-2]
        
        # Velocity
        axes[0, 0].plot(t, self._linear_vel, 'b-', label='Linear')
        axes[0, 0].set_ylabel('Velocity (m/s)')
        axes[0, 0].set_title('Linear Velocity')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(t, self._angular_vel, 'r-', label='Angular')
        axes[0, 1].set_ylabel('Velocity (rad/s)')
        axes[0, 1].set_title('Angular Velocity')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Acceleration
        axes[1, 0].plot(t_accel, self._linear_accel, 'b-')
        axes[1, 0].set_ylabel('Accel (m/s²)')
        axes[1, 0].set_title('Linear Acceleration')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(t_accel, self._angular_accel, 'r-')
        axes[1, 1].set_ylabel('Accel (rad/s²)')
        axes[1, 1].set_title('Angular Acceleration')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Jerk
        axes[2, 0].plot(t_jerk, self._linear_jerk, 'b-')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Jerk (m/s³)')
        axes[2, 0].set_title('Linear Jerk')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(t_jerk, self._angular_jerk, 'r-')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Jerk (rad/s³)')
        axes[2, 1].set_title('Angular Jerk')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Velocity profile saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_lane_tracking(self, output_path: Optional[Path] = None) -> None:
        """
        Plot lane detection and tracking performance.
        
        Args:
            output_path: Path to save plot (None = show interactively)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("matplotlib not installed")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        t = self._timestamps
        lane_center = np.array([p.lane_center for p in self._points])
        confidence = np.array([p.confidence for p in self._points])
        lane_detected = np.array([p.lane_detected for p in self._points])
        
        # Lane center (should stay near 0.5 for centered)
        axes[0].plot(t, lane_center, 'b-', linewidth=1)
        axes[0].axhline(y=0.5, color='g', linestyle='--', label='Center')
        axes[0].fill_between(t, 0.42, 0.58, alpha=0.2, color='green', label='Deadband')
        axes[0].set_ylabel('Lane Center')
        axes[0].set_title('Lane Center Position (0=left, 1=right)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Confidence
        axes[1].plot(t, confidence, 'orange', linewidth=1)
        axes[1].set_ylabel('Confidence')
        axes[1].set_title('Detection Confidence')
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3)
        
        # Angular velocity response
        axes[2].plot(t, self._angular_vel, 'r-', linewidth=1)
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Angular Vel (rad/s)')
        axes[2].set_title('Angular Velocity Command')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Lane tracking plot saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_report(self, output_path: Path) -> None:
        """Save full analysis report to text file."""
        if self._metrics is None:
            self.compute_metrics()
        
        with open(output_path, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("  AETHERNAV TRAJECTORY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Duration: {self.trajectory.duration:.2f} seconds\n")
            f.write(f"Data points: {len(self._points)}\n")
            f.write(f"Sample rate: {len(self._points) / self.trajectory.duration:.1f} Hz\n\n")
            
            f.write(str(self._metrics))
            
            f.write("\n\n=== Recommendations ===\n")
            
            if self._metrics.oscillation_frequency > 1.0:
                f.write("HIGH OSCILLATION FREQUENCY detected\n")
                f.write("    → Consider reducing angular_gain in lane_follower config\n")
                f.write("    → Increase deadband to reduce sensitivity\n\n")
            
            if self._metrics.max_angular_jerk > 5.0:
                f.write("HIGH ANGULAR JERK detected\n")
                f.write("    → Add angular velocity rate limiting\n")
                f.write("    → Increase temporal_alpha for smoother centerline\n\n")
            
            if self._metrics.mean_curvature > 2.0:
                f.write("HIGH CURVATURE detected\n")
                f.write("    → Check if track has sharp turns\n")
                f.write("    → Consider lowering cruise_linear_vel\n\n")
        
        logger.info(f"Report saved to {output_path}")
