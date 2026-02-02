"""
TensorRT-based segmentation engine.

Handles model loading, inference, and post-processing for road/lane
segmentation using TensorRT for optimized inference on Jetson.
"""

import numpy as np
import cv2
import time
import os
from typing import Optional, Tuple
from dataclasses import dataclass

from ..config import SegmentationConfig
from ..core.types import SegmentationResult
from ..core.logging_config import get_logger

logger = get_logger("segmentation")

# TensorRT imports (may fail on non-Jetson systems)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT/PyCUDA not available - segmentation will use mock mode")


class SegmentationEngine:
    """
    TensorRT-based road segmentation engine.
    
    Loads a TensorRT engine file and performs inference to detect
    road/lane regions in camera frames.
    """
    
    def __init__(self, config: SegmentationConfig):
        """
        Initialize segmentation engine.
        
        Args:
            config: Segmentation configuration
        """
        self.config = config
        self._engine = None
        self._context = None
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Load TensorRT engine and allocate buffers.
        
        Returns:
            True if initialization successful
        """
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available, running in mock mode")
            self._initialized = True
            return True
        
        if not os.path.exists(self.config.engine_path):
            logger.error(f"Engine file not found: {self.config.engine_path}")
            return False
        
        try:
            logger.info(f"Loading TensorRT engine: {self.config.engine_path}")
            
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            with open(self.config.engine_path, "rb") as f:
                runtime = trt.Runtime(trt_logger)
                self._engine = runtime.deserialize_cuda_engine(f.read())
            
            if self._engine is None:
                logger.error("Failed to deserialize TensorRT engine")
                return False
            
            self._context = self._engine.create_execution_context()
            self._stream = cuda.Stream()
            
            # Allocate buffers
            self._inputs = []
            self._outputs = []
            self._bindings = []
            
            for i in range(self._engine.num_bindings):
                shape = self._engine.get_binding_shape(i)
                size = trt.volume(shape)
                dtype = trt.nptype(self._engine.get_binding_dtype(i))
                
                # Allocate host and device memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self._bindings.append(int(dev_mem))
                
                if self._engine.binding_is_input(i):
                    self._inputs.append({'host': host_mem, 'dev': dev_mem, 'shape': shape})
                else:
                    self._outputs.append({'host': host_mem, 'dev': dev_mem, 'shape': shape})
            
            logger.info(
                f"Engine loaded: {len(self._inputs)} inputs, {len(self._outputs)} outputs"
            )
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT engine: {e}")
            return False
    
    def infer(self, frame: np.ndarray) -> SegmentationResult:
        """
        Run segmentation inference on a frame.
        
        Args:
            frame: BGR image from camera (H x W x 3)
            
        Returns:
            SegmentationResult with lane detection info
        """
        timestamp = time.time()
        
        if not self._initialized:
            logger.warning("Engine not initialized, returning empty result")
            return self._empty_result(timestamp)
        
        # Mock mode for testing without TensorRT
        if not TRT_AVAILABLE:
            return self._mock_inference(frame, timestamp)
        
        try:
            # Preprocess
            input_blob = self._preprocess(frame)
            
            # Copy input to GPU
            np.copyto(self._inputs[0]['host'], input_blob.ravel())
            cuda.memcpy_htod_async(
                self._inputs[0]['dev'], 
                self._inputs[0]['host'], 
                self._stream
            )
            
            # Run inference
            self._context.execute_async_v2(
                bindings=self._bindings, 
                stream_handle=self._stream.handle
            )
            
            # Copy output from GPU
            cuda.memcpy_dtoh_async(
                self._outputs[0]['host'], 
                self._outputs[0]['dev'], 
                self._stream
            )
            self._stream.synchronize()
            
            # Post-process
            raw_output = self._outputs[0]['host']
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
        # Resize to model input dimensions
        resized = cv2.resize(
            frame, 
            (self.config.input_width, self.config.input_height)
        )
        
        # Convert BGR to RGB and normalize
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to channels-first (C x H x W)
        blob = normalized.transpose(2, 0, 1)
        
        return blob
    
    def _postprocess(self, raw_output: np.ndarray, timestamp: float) -> SegmentationResult:
        """
        Post-process model output to extract lane information.
        
        Args:
            raw_output: Model output (flattened probabilities)
            timestamp: Capture timestamp
            
        Returns:
            SegmentationResult with lane detection
        """
        # Reshape to 2D mask
        mask = raw_output.reshape(self.config.input_height, self.config.input_width)
        
        # Extract ROI (bottom portion of image)
        start_row = int(self.config.input_height * self.config.roi_start_ratio)
        roi = mask[start_row:, :]
        
        # Find lane pixels above threshold
        lane_pixels = np.where(roi > self.config.confidence_threshold)
        y_indices, x_indices = lane_pixels
        
        # Check if enough pixels detected
        if len(x_indices) < self.config.min_lane_pixels:
            return SegmentationResult(
                mask=mask,
                lane_center_normalized=0.5,
                lane_detected=False,
                confidence=0.0,
                timestamp=timestamp
            )
        
        # Calculate lane center
        center_x = np.mean(x_indices)
        lane_center_normalized = center_x / self.config.input_width
        
        # Calculate confidence as fraction of high-probability pixels
        total_roi_pixels = roi.size
        lane_pixels_count = len(x_indices)
        confidence = min(1.0, lane_pixels_count / (total_roi_pixels * 0.3))
        
        return SegmentationResult(
            mask=mask,
            lane_center_normalized=lane_center_normalized,
            lane_detected=True,
            confidence=confidence,
            timestamp=timestamp
        )
    
    def _mock_inference(self, frame: np.ndarray, timestamp: float) -> SegmentationResult:
        """
        Mock inference for testing without TensorRT.
        
        Generates a simple synthetic lane detection based on frame center.
        """
        # Create a simple mock mask
        h, w = self.config.input_height, self.config.input_width
        mask = np.zeros((h, w), dtype=np.float32)
        
        # Create a fake lane in the center
        center_x = w // 2
        lane_width = w // 4
        mask[int(h * 0.5):, center_x - lane_width:center_x + lane_width] = 0.8
        
        return SegmentationResult(
            mask=mask,
            lane_center_normalized=0.5,
            lane_detected=True,
            confidence=0.8,
            timestamp=timestamp
        )
    
    def _empty_result(self, timestamp: float) -> SegmentationResult:
        """Return an empty result for error cases."""
        return SegmentationResult(
            mask=np.zeros((self.config.input_height, self.config.input_width)),
            lane_center_normalized=0.5,
            lane_detected=False,
            confidence=0.0,
            timestamp=timestamp
        )
    
    def cleanup(self) -> None:
        """Release resources."""
        # PyCUDA handles cleanup automatically
        self._initialized = False
        logger.info("Segmentation engine cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        """Check if engine is initialized."""
        return self._initialized
