"""
Segmentation module for road/lane detection.

Provides inference backends (TensorRT, ONNX, Mock) and the main
SegmentationEngine for processing camera frames.
"""

from .base import InferenceBackend
from .tensorrt_backend import TensorRTBackend
from .onnx_backend import ONNXBackend
from .mock_backend import MockBackend
from .engine import SegmentationEngine

__all__ = [
    "InferenceBackend",
    "TensorRTBackend",
    "ONNXBackend",
    "MockBackend",
    "SegmentationEngine",
]
