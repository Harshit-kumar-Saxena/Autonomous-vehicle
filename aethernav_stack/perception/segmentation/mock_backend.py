"""
Mock inference backend for testing without actual models.
"""

import numpy as np

from .base import InferenceBackend
from ...core.logging_config import get_logger

logger = get_logger("segmentation.mock")


class MockBackend(InferenceBackend):
    """Mock backend for testing without actual models."""
    
    def __init__(self, input_height: int, input_width: int):
        self.input_height = input_height
        self.input_width = input_width
        self._initialized = False
    
    def initialize(self) -> bool:
        logger.info("Mock backend initialized (no actual inference)")
        self._initialized = True
        return True
    
    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        # Generate a simple mock output (fake road in center)
        h, w = self.input_height, self.input_width
        mock_output = np.zeros((h, w), dtype=np.float32)
        
        center_x = w // 2
        lane_width = w // 4
        mock_output[int(h * 0.5):, center_x - lane_width:center_x + lane_width] = 0.8
        
        return mock_output.ravel()
    
    def cleanup(self) -> None:
        self._initialized = False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
