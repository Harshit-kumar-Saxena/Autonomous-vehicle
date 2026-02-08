"""
Pipeline components for threaded execution.

Clean, single-responsibility classes for each pipeline stage.
"""

import cv2
import time
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from .threading_utils import StoppableThread, LatestHolder
from .types import SegmentationResult, RobotState
from .logging_config import get_logger

logger = get_logger("pipeline")


# ============================================================
# Data containers passed between pipeline stages
# ============================================================

@dataclass
class FrameData:
    """Container for camera frame data."""
    image: any  # np.ndarray
    timestamp: float
    frame_number: int


@dataclass
class PerceptionData:
    """Container for perception output."""
    segmentation: SegmentationResult
    timestamp: float


# ============================================================
# Pipeline Stage: Camera Capture
# ============================================================

class CameraThread(StoppableThread):
    """
    Continuously captures frames from camera.
    
    Publishes latest frame to output holder. Old frames are
    automatically discarded if consumer is slower.
    """
    
    def __init__(
        self,
        camera,  # CameraManager
        output: LatestHolder[FrameData],
        target_fps: int = 30
    ):
        super().__init__(name="CameraThread")
        self.camera = camera
        self.output = output
        self.target_fps = target_fps
        self._frame_count = 0
    
    def on_start(self) -> None:
        logger.info("Camera thread started")
    
    def run_loop(self) -> None:
        frame = self.camera.read()
        
        if frame is not None and frame.is_valid:
            self._frame_count += 1
            self.output.put(FrameData(
                image=frame.image,
                timestamp=frame.timestamp,
                frame_number=self._frame_count
            ))
        
        # Rate limiting (camera may already be rate-limited)
        self.sleep_or_stop(1.0 / self.target_fps)
    
    def on_stop(self) -> None:
        logger.info(f"Camera thread stopped (captured {self._frame_count} frames)")


# ============================================================
# Pipeline Stage: Perception (Segmentation)
# ============================================================

class PerceptionThread(StoppableThread):
    """
    Runs segmentation on latest camera frame.
    
    Reads from frame holder, runs inference, publishes result.
    Skips old frames to always process the freshest data.
    """
    
    def __init__(
        self,
        segmentation,  # SegmentationEngine
        input_frames: LatestHolder[FrameData],
        output: LatestHolder[PerceptionData],
        show_visualization: bool = False
    ):
        super().__init__(name="PerceptionThread")
        self.segmentation = segmentation
        self.input_frames = input_frames
        self.output = output
        self.show_visualization = show_visualization
        self._inference_count = 0
        self._last_frame = None  # Store frame for visualization
    
    def on_start(self) -> None:
        logger.info(f"Perception thread started (visualization={self.show_visualization})")
    
    def run_loop(self) -> None:
        # Wait for a frame (with timeout to check for stop)
        frame_data = self.input_frames.get(timeout=0.1)
        
        if frame_data is None:
            return  # No frame available, loop again
        
        self._last_frame = frame_data.image
        
        # Run inference
        seg_result = self.segmentation.infer(frame_data.image)
        self._inference_count += 1
        
        # Publish result
        self.output.put(PerceptionData(
            segmentation=seg_result,
            timestamp=time.time()
        ))
        
        # Show visualization if enabled
        if self.show_visualization:
            self._show_visualization(seg_result)
    
    def _show_visualization(self, seg_result: SegmentationResult) -> None:
        """Display segmentation visualization with cv2.imshow."""
        if self._last_frame is None:
            return
        
        # Create visualization frame
        vis_frame = self._last_frame.copy()
        
        # Resize mask to match frame size
        h, w = vis_frame.shape[:2]
        mask_resized = cv2.resize(
            (seg_result.mask * 255).astype(np.uint8), 
            (w, h), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Create colored overlay
        overlay = np.zeros_like(vis_frame)
        overlay[:, :, 1] = mask_resized  # Green channel for road
        
        # Blend with original frame
        alpha = 0.4
        vis_frame = cv2.addWeighted(vis_frame, 1 - alpha, overlay, alpha, 0)
        
        # Draw centerline if available
        if len(seg_result.centerline_points) > 1:
            # Scale centerline points to frame size
            scale_x = w / seg_result.mask.shape[1]
            scale_y = h / seg_result.mask.shape[0]
            
            scaled_points = [
                (int(pt[0] * scale_x), int(pt[1] * scale_y)) 
                for pt in seg_result.centerline_points
            ]
            
            # Draw centerline as polyline
            for i in range(len(scaled_points) - 1):
                cv2.line(
                    vis_frame, 
                    scaled_points[i], 
                    scaled_points[i + 1], 
                    (0, 0, 255),  # Red
                    thickness=3
                )
            
            # Draw lane center indicator at bottom
            lane_center_x = int(seg_result.lane_center_normalized * w)
            cv2.circle(vis_frame, (lane_center_x, h - 30), 10, (255, 255, 0), -1)
            cv2.line(vis_frame, (w // 2, h - 50), (w // 2, h - 10), (255, 255, 255), 2)
        
        # Add info text
        info_text = f"Lane: {'YES' if seg_result.lane_detected else 'NO'} | Conf: {seg_result.confidence:.2f}"
        cv2.putText(
            vis_frame, info_text, (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        
        cv2.imshow("AetherNav Segmentation", vis_frame)
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            logger.info("Quit requested via visualization window")
            self.stop()
    
    def on_stop(self) -> None:
        if self.show_visualization:
            cv2.destroyAllWindows()
        logger.info(f"Perception thread stopped ({self._inference_count} inferences)")


# ============================================================
# Pipeline Stage: Control Loop
# ============================================================

class ControlThread(StoppableThread):
    """
    Runs planning and control at fixed rate.
    
    Uses latest perception data for planning, but maintains
    consistent control rate even if perception is slow.
    """
    
    def __init__(
        self,
        planner,
        controller,
        odometry,
        hardware,
        perception_input: LatestHolder[PerceptionData],
        robot_params,
        loop_rate_hz: int = 30,
        dry_run: bool = False
    ):
        super().__init__(name="ControlThread")
        self.planner = planner
        self.controller = controller
        self.odometry = odometry
        self.hardware = hardware
        self.perception_input = perception_input
        self.robot_params = robot_params
        self.loop_rate_hz = loop_rate_hz
        self.dry_run = dry_run
        
        self._robot_state = RobotState()
        self._last_perception: Optional[PerceptionData] = None
        self._control_count = 0
        self._last_time = time.time()
    
    def on_start(self) -> None:
        logger.info(f"Control thread started @ {self.loop_rate_hz} Hz")
    
    def run_loop(self) -> None:
        loop_start = time.time()
        
        # Get latest perception (non-blocking)
        perception = self.perception_input.get_nowait()
        if perception is not None:
            self._last_perception = perception
        
        # Skip if no perception data yet
        if self._last_perception is None:
            self.sleep_or_stop(0.01)
            return
        
        # Calculate dt
        now = time.time()
        dt = now - self._last_time
        self._last_time = now
        
        # 1. Run planner
        cmd_vel = self.planner.compute_velocity(
            self._last_perception.segmentation,
            self._robot_state
        )
        
        # 2. Run controller
        wheel_velocities = self.controller.compute_wheel_velocities(
            cmd_vel,
            self._robot_state,
            dt
        )
        
        # 3. Send to hardware
        if not self.dry_run:
            self.hardware.write_wheel_velocities(wheel_velocities)
        
        # 4. Update odometry
        feedback = self.hardware.read_wheel_velocities() if not self.dry_run else wheel_velocities
        imu_yaw = self.hardware.read_imu_yaw() if not self.dry_run else None
        
        self._robot_state = self.odometry.update(
            wheel_velocities=feedback or wheel_velocities,
            imu_yaw=imu_yaw
        )
        self._robot_state.wheel_velocities = wheel_velocities
        
        self._control_count += 1
        
        # Maintain loop rate
        elapsed = time.time() - loop_start
        sleep_time = (1.0 / self.loop_rate_hz) - elapsed
        if sleep_time > 0:
            self.sleep_or_stop(sleep_time)
    
    def on_stop(self) -> None:
        # Emergency stop on shutdown
        if not self.dry_run:
            self.hardware.emergency_stop()
        logger.info(f"Control thread stopped ({self._control_count} cycles)")
    
    @property
    def robot_state(self) -> RobotState:
        return self._robot_state
    
    @property
    def control_count(self) -> int:
        return self._control_count
