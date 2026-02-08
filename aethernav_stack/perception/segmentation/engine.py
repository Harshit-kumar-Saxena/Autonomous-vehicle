"""
Main segmentation engine with configurable backends.

Handles model loading, inference, and post-processing for road/lane
segmentation with mask filtering and centerline extraction.
"""

import cv2
import time
import numpy as np
from typing import Optional, List, Tuple

from .base import InferenceBackend
from .tensorrt_backend import TensorRTBackend
from .onnx_backend import ONNXBackend
from .mock_backend import MockBackend

from ...config import SegmentationConfig
from ...core.types import SegmentationResult
from ...core.logging_config import get_logger
from ..mask_filtering import clean_mask_from_config
from ..centerline_extraction import extract_centerline_from_config
from ..temporal_smoothing import TemporalSmoother

logger = get_logger("segmentation")


class SegmentationEngine:
    """
    Road segmentation engine with configurable backends.
    
    Supports TensorRT (Jetson/NVIDIA), ONNX Runtime (cross-platform),
    and mock mode for testing. Includes mask filtering and centerline
    extraction.
    """
    
    def __init__(self, config: SegmentationConfig):
        """
        Initialize segmentation engine.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        self._backend: Optional[InferenceBackend] = None
        self._temporal_smoother = TemporalSmoother.from_config(config)
        self._prev_centerline: List[Tuple[int, int]] = []
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize inference backend based on config.
        
        Returns:
            True if initialization successful
        """
        backend_type = self.config.inference_backend.lower()
        
        # Try to initialize the configured backend
        if backend_type == "tensorrt":
            self._backend = TensorRTBackend(self.config.engine_path)
            if not self._backend.initialize():
                logger.warning("TensorRT init failed, trying ONNX fallback")
                self._backend = ONNXBackend(self.config.onnx_path)
                if not self._backend.initialize():
                    logger.warning("ONNX init failed, using mock mode")
                    self._backend = MockBackend(
                        self.config.input_height, 
                        self.config.input_width
                    )
                    self._backend.initialize()
        
        elif backend_type == "onnx":
            self._backend = ONNXBackend(self.config.onnx_path)
            if not self._backend.initialize():
                logger.warning("ONNX init failed, trying TensorRT fallback")
                self._backend = TensorRTBackend(self.config.engine_path)
                if not self._backend.initialize():
                    logger.warning("TensorRT init failed, using mock mode")
                    self._backend = MockBackend(
                        self.config.input_height, 
                        self.config.input_width
                    )
                    self._backend.initialize()
        
        else:
            logger.warning(f"Unknown backend '{backend_type}', using mock mode")
            self._backend = MockBackend(
                self.config.input_height, 
                self.config.input_width
            )
            self._backend.initialize()
        
        self._initialized = self._backend.is_initialized
        return self._initialized
    
    def infer(self, frame: np.ndarray) -> SegmentationResult:
        """
        Run segmentation inference on a frame.
        
        Args:
            frame: BGR image from camera (H x W x 3)
            
        Returns:
            SegmentationResult with lane detection info and centerline
        """
        timestamp = time.time()
        
        if not self._initialized:
            logger.warning("Engine not initialized, returning empty result")
            return self._empty_result(timestamp)
        
        try:
            # Preprocess
            input_blob = self._preprocess(frame)
            
            # Run inference
            raw_output = self._backend.infer(input_blob)
            
            # Post-process (includes mask cleaning and centerline extraction)
            return self._postprocess(raw_output, timestamp)
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return self._empty_result(timestamp)
    
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for model input.
        
        Args:
            frame: BGR image (H x W x 3)
            
        Returns:
            Preprocessed tensor (C x H x W), normalized to [0, 1]
        """
        resized = cv2.resize(
            frame, 
            (self.config.input_width, self.config.input_height)
        )
        
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        blob = normalized.transpose(2, 0, 1)
        
        return blob
    
    def _postprocess(self, raw_output: np.ndarray, timestamp: float) -> SegmentationResult:
        """
        Post-process model output with mask filtering and centerline extraction.
        
        Args:
            raw_output: Model output (flattened probabilities)
            timestamp: Capture timestamp
            
        Returns:
            SegmentationResult with centerline
        """
        # Reshape to 2D mask
        mask = raw_output.reshape(self.config.input_height, self.config.input_width)
        
        # Apply sigmoid if needed (for raw logits)
        if mask.min() < 0 or mask.max() > 1:
            mask = 1 / (1 + np.exp(-mask))
        
        # Threshold to binary
        binary_mask = (mask > self.config.confidence_threshold).astype(np.uint8)
        
        # Clean mask with morphological operations
        clean_mask = clean_mask_from_config(binary_mask, self.config)
        
        # Extract centerline
        centerline_raw, debug_info = extract_centerline_from_config(
            clean_mask, 
            self.config,
            self._prev_centerline
        )
        
        # Apply temporal smoothing
        centerline_smooth = self._temporal_smoother.smooth(centerline_raw)
        self._prev_centerline = centerline_smooth
        
        # Calculate lane center from centerline
        if len(centerline_smooth) > 0:
            # Use bottom portion of centerline for lane center calculation
            bottom_points = centerline_smooth[:max(1, len(centerline_smooth)//3)]
            center_x = np.mean([p[0] for p in bottom_points])
            lane_center_normalized = center_x / self.config.input_width
            lane_detected = True
            confidence = min(1.0, debug_info['valid_rows'] / 20)
        else:
            lane_center_normalized = 0.5
            lane_detected = False
            confidence = 0.0
        
        return SegmentationResult(
            mask=mask,
            centerline_points=centerline_smooth,
            lane_center_normalized=lane_center_normalized,
            lane_detected=lane_detected,
            confidence=confidence,
            timestamp=timestamp
        )
    
    def _empty_result(self, timestamp: float) -> SegmentationResult:
        """Return an empty result for error cases."""
        return SegmentationResult(
            mask=np.zeros((self.config.input_height, self.config.input_width)),
            centerline_points=[],
            lane_center_normalized=0.5,
            lane_detected=False,
            confidence=0.0,
            timestamp=timestamp
        )
    
    def cleanup(self) -> None:
        """Release resources."""
        if self._backend:
            self._backend.cleanup()
        self._initialized = False
        logger.info("Segmentation engine cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized
    
    @property
    def backend_name(self) -> str:
        """Get the name of the active backend."""
        if self._backend is None:
            return "none"
        return self._backend.__class__.__name__
