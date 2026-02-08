"""
Abstract base class for inference backends.
"""

import numpy as np
from abc import ABC, abstractmethod


class InferenceBackend(ABC):
    """Abstract base class for inference backends."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend. Returns True if successful."""
        pass
    
    @abstractmethod
    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        """Run inference on preprocessed input. Returns raw output."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release resources."""
        pass
    
    @property
    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if backend is ready."""
        pass
