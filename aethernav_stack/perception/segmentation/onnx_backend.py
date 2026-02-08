"""
ONNX Runtime inference backend for cross-platform support.
"""

import os
import numpy as np

from .base import InferenceBackend
from ...core.logging_config import get_logger

logger = get_logger("segmentation.onnx")

# ONNX Runtime imports
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.info("ONNX Runtime not available")


class ONNXBackend(InferenceBackend):
    """ONNX Runtime inference backend for cross-platform support."""
    
    def __init__(self, onnx_path: str):
        self.onnx_path = onnx_path
        self._session = None
        self._input_name = None
        self._output_name = None
        self._initialized = False
    
    def initialize(self) -> bool:
        if not ONNX_AVAILABLE:
            logger.warning("ONNX Runtime not available")
            return False
        
        if not os.path.exists(self.onnx_path):
            logger.error(f"ONNX model not found: {self.onnx_path}")
            return False
        
        try:
            logger.info(f"Loading ONNX model: {self.onnx_path}")
            
            # Try GPU providers first, fall back to CPU
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            self._session = ort.InferenceSession(self.onnx_path, providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            
            logger.info(f"ONNX model loaded with providers: {self._session.get_providers()}")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ONNX Runtime: {e}")
            return False
    
    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("ONNX backend not initialized")
        
        # ONNX expects NCHW format with batch dimension
        if input_blob.ndim == 3:
            input_blob = np.expand_dims(input_blob, axis=0)
        
        output = self._session.run([self._output_name], {self._input_name: input_blob})[0]
        return output.ravel()
    
    def cleanup(self) -> None:
        self._session = None
        self._initialized = False
        logger.info("ONNX backend cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
