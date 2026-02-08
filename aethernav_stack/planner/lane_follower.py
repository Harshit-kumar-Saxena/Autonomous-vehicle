"""
Lane following planner.

Converts lane segmentation data into velocity commands to keep the
robot centered in the detected lane.
"""

from typing import Optional
from dataclasses import dataclass

from ..core.interfaces import PlannerPlugin
from ..core.types import Twist2D, SegmentationResult, RobotState
from ..config import LaneFollowerConfig
from ..core.logging_config import get_logger

logger = get_logger("planner.lane_follower")


@dataclass
class LaneFollowerState:
    """Internal state for lane follower."""
    last_valid_command: Twist2D
    lost_lane_frames: int = 0
    consecutive_lane_detected: int = 0


class LaneFollowerPlanner(PlannerPlugin):
    """
    Lane-following planner.
    
    Computes velocity commands to follow detected lanes:
    - When lane center is within deadband → go straight
    - When lane is offset → apply angular velocity proportional to offset
    - When lane is lost → gradually stop or hold last command
    """
    
    def __init__(self, config: LaneFollowerConfig):
        """
        Initialize lane follower.
        
        Args:
            config: Lane follower configuration
        """
        self.config = config
        self._state = LaneFollowerState(
            last_valid_command=Twist2D(linear=0.0, angular=0.0)
        )
    
    @property
    def name(self) -> str:
        return "lane_follower"
    
    def compute_velocity(
        self,
        perception_data: SegmentationResult,
        robot_state: RobotState
    ) -> Twist2D:
        """
        Compute velocity command based on lane detection.
        
        Args:
            perception_data: Lane segmentation result
            robot_state: Current robot state
            
        Returns:
            Twist2D velocity command
        """
        if not perception_data.lane_detected:
            return self._handle_lane_lost()
        
        # Reset lost counter and update detected counter
        self._state.lost_lane_frames = 0
        self._state.consecutive_lane_detected += 1
        
        # Get lane error (negative = left, positive = right)
        lane_error = perception_data.lane_error
        
        # Check if within deadband (go straight)
        if self._is_within_deadband(perception_data.lane_center_normalized):
            cmd = Twist2D(
                linear=self.config.cruise_linear_vel,
                angular=0.0
            )
        else:
            # Compute angular velocity proportional to error
            angular = -lane_error * self.config.angular_gain
            
            # Optionally reduce linear velocity during turns
            if self.config.slow_down_on_curve:
                turn_factor = 1.0 - abs(lane_error) * (1.0 - self.config.curve_slow_factor) * 2
                turn_factor = max(self.config.curve_slow_factor, turn_factor)
                linear = self.config.cruise_linear_vel * turn_factor
            else:
                linear = self.config.cruise_linear_vel
            
            # Ensure minimum linear velocity
            linear = max(linear, self.config.min_linear_vel)
            
            cmd = Twist2D(linear=linear, angular=angular)
        
        # Store as last valid command
        self._state.last_valid_command = cmd
        
        return cmd
    
    def _is_within_deadband(self, lane_center_normalized: float) -> bool:
        """Check if lane center is within the straight-ahead deadband."""
        return (
            self.config.deadband_min <= 
            lane_center_normalized <= 
            self.config.deadband_max
        )
    
    def _handle_lane_lost(self) -> Twist2D:
        """
        Handle case when lane is not detected.
        
        Implements recovery behavior:
        1. Hold last command for a few frames
        2. Gradually slow down
        3. Stop after timeout
        """
        self._state.lost_lane_frames += 1
        self._state.consecutive_lane_detected = 0
        
        if not self.config.recovery_enabled:
            # No recovery - stop immediately
            logger.debug("Lane lost - stopping (recovery disabled)")
            return Twist2D()
        
        if self._state.lost_lane_frames <= self.config.hold_last_command_frames:
            # Hold last valid command
            logger.debug(
                f"Lane lost - holding last command "
                f"({self._state.lost_lane_frames}/{self.config.hold_last_command_frames})"
            )
            return self._state.last_valid_command
        
        elif self._state.lost_lane_frames <= self.config.stop_after_lost_frames:
            # Gradually slow down
            frames_into_slowdown = (
                self._state.lost_lane_frames - self.config.hold_last_command_frames
            )
            slowdown_duration = (
                self.config.stop_after_lost_frames - self.config.hold_last_command_frames
            )
            
            if slowdown_duration > 0:
                progress = frames_into_slowdown / slowdown_duration
                scale = 1.0 - progress
            else:
                scale = 0.0
            
            logger.debug(f"Lane lost - slowing down (scale={scale:.2f})")
            
            return Twist2D(
                linear=self._state.last_valid_command.linear * scale,
                angular=self._state.last_valid_command.angular * scale
            )
        else:
            # Full stop
            logger.debug("Lane lost - stopped")
            return Twist2D()
    
    def reset(self) -> None:
        """Reset planner state."""
        self._state = LaneFollowerState(
            last_valid_command=Twist2D()
        )
        logger.debug("Lane follower reset")
    
    def get_status(self) -> dict:
        """Get current planner status for telemetry."""
        return {
            "planner": self.name,
            "lost_lane_frames": self._state.lost_lane_frames,
            "consecutive_detected": self._state.consecutive_lane_detected,
            "last_cmd_linear": self._state.last_valid_command.linear,
            "last_cmd_angular": self._state.last_valid_command.angular,
        }
