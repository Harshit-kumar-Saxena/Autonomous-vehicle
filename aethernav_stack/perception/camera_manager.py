"""
Camera manager for frame capture and validation.

Handles camera initialization, frame capture, and automatic recovery
on camera disconnection.
"""

import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum

from ..config import CameraConfig
from ..core.logging_config import get_logger, RateLimitedLogger

logger = get_logger("camera")


class CameraBackend(Enum):
    """Supported camera backends."""
    V4L2 = cv2.CAP_V4L2
    GSTREAMER = cv2.CAP_GSTREAMER
    AUTO = cv2.CAP_ANY


@dataclass
class CapturedFrame:
    """Container for a captured frame with metadata."""
    image: np.ndarray
    timestamp: float
    frame_number: int
    width: int
    height: int
    
    @property
    def is_valid(self) -> bool:
        """Check if frame is valid."""
        return self.image is not None and self.image.size > 0


class CameraManager:
    """
    Manages camera capture with automatic reconnection.
    
    Features:
    - Automatic retry on connection failure
    - Frame validation
    - FPS tracking
    - Graceful error handling
    """
    
    def __init__(self, config: CameraConfig):
        """
        Initialize camera manager.
        
        Args:
            config: Camera configuration
        """
        self.config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_count = 0
        self._last_capture_time = 0.0
        self._fps_tracker = _FPSTracker()
        self._rate_limited_logger = RateLimitedLogger(logger, min_interval=5.0)
        self._using_video_file = config.use_video_file
        self._video_loop = True  # Loop video when reaching end
        
        # Determine backend
        backend_map = {
            "v4l2": CameraBackend.V4L2,
            "gstreamer": CameraBackend.GSTREAMER,
            "auto": CameraBackend.AUTO,
        }
        self._backend = backend_map.get(
            config.backend.lower(), 
            CameraBackend.AUTO
        )
    
    def open(self) -> bool:
        """
        Open the camera or video file.
        
        Returns:
            True if camera/video opened successfully
        """
        if self._cap is not None and self._cap.isOpened():
            logger.warning("Camera already open, closing first")
            self.close()
        
        try:
            # Video file mode
            if self._using_video_file:
                import os
                if not self.config.video_path or not os.path.exists(self.config.video_path):
                    logger.error(f"Video file not found: {self.config.video_path}")
                    return False
                
                logger.info(f"Opening video file: {self.config.video_path}")
                self._cap = cv2.VideoCapture(self.config.video_path)
                
                if not self._cap.isOpened():
                    logger.error(f"Failed to open video file: {self.config.video_path}")
                    return False
                
                # Get video properties
                actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                logger.info(
                    f"Video opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS, "
                    f"{total_frames} frames (loop={self._video_loop})"
                )
                
                self._frame_count = 0
                return True
            
            # Live camera mode
            logger.info(
                f"Opening camera {self.config.device_id} with backend "
                f"{self._backend.name} ({self.config.width}x{self.config.height})"
            )
            
            self._cap = cv2.VideoCapture(
                self.config.device_id, 
                self._backend.value
            )
            
            if not self._cap.isOpened():
                logger.error(f"Failed to open camera {self.config.device_id}")
                return False
            
            # Configure camera
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Verify settings (they may not be exactly as requested)
            actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(
                f"Camera opened: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
            )
            
            self._frame_count = 0
            return True
            
        except Exception as e:
            logger.error(f"Exception opening camera/video: {e}")
            return False
    
    def close(self) -> None:
        """Release the camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed")
    
    def read(self) -> Optional[CapturedFrame]:
        """
        Capture a frame from the camera or video file.
        
        Returns:
            CapturedFrame if successful, None otherwise
        """
        if self._cap is None or not self._cap.isOpened():
            self._rate_limited_logger.warning("Camera not open, attempting to reopen")
            if not self.open():
                return None
        
        try:
            ret, frame = self._cap.read()
            
            # Handle video loop
            if not ret and self._using_video_file and self._video_loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if ret:
                    logger.debug("Video looped to beginning")
            
            if not ret or frame is None:
                self._rate_limited_logger.warning("Failed to read frame from camera/video")
                return None
            
            timestamp = time.time()
            self._frame_count += 1
            self._fps_tracker.update(timestamp)
            
            return CapturedFrame(
                image=frame,
                timestamp=timestamp,
                frame_number=self._frame_count,
                width=frame.shape[1],
                height=frame.shape[0]
            )
            
        except Exception as e:
            self._rate_limited_logger.error(f"Exception reading frame: {e}")
            return None
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open."""
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def fps(self) -> float:
        """Get current capture FPS."""
        return self._fps_tracker.fps
    
    @property
    def frame_count(self) -> int:
        """Get total frames captured."""
        return self._frame_count
    
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class _FPSTracker:
    """Tracks frames per second using a sliding window."""
    
    def __init__(self, window_size: int = 30):
        self._window_size = window_size
        self._timestamps: list[float] = []
        self._fps = 0.0
    
    def update(self, timestamp: float) -> None:
        """Add a timestamp and update FPS."""
        self._timestamps.append(timestamp)
        
        # Keep only recent timestamps
        if len(self._timestamps) > self._window_size:
            self._timestamps.pop(0)
        
        # Calculate FPS from window
        if len(self._timestamps) >= 2:
            duration = self._timestamps[-1] - self._timestamps[0]
            if duration > 0:
                self._fps = (len(self._timestamps) - 1) / duration
    
    @property
    def fps(self) -> float:
        return self._fps
