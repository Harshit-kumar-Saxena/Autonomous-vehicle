"""
TensorRT inference backend for Jetson/NVIDIA GPUs.
"""

import os
import numpy as np

from .base import InferenceBackend
from ...core.logging_config import get_logger

logger = get_logger("segmentation.tensorrt")

# TensorRT imports (may fail on non-Jetson systems)
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.info("TensorRT/PyCUDA not available")


class TensorRTBackend(InferenceBackend):
    """TensorRT inference backend for Jetson/NVIDIA GPUs."""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self._engine = None
        self._context = None
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = None
        self._initialized = False
    
    def initialize(self) -> bool:
        if not TRT_AVAILABLE:
            logger.warning("TensorRT not available")
            return False
        
        if not os.path.exists(self.engine_path):
            logger.error(f"TensorRT engine not found: {self.engine_path}")
            return False
        
        try:
            logger.info(f"Loading TensorRT engine: {self.engine_path}")
            
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            with open(self.engine_path, "rb") as f:
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
                
                host_mem = cuda.pagelocked_empty(size, dtype)
                dev_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self._bindings.append(int(dev_mem))
                
                if self._engine.binding_is_input(i):
                    self._inputs.append({'host': host_mem, 'dev': dev_mem, 'shape': shape})
                else:
                    self._outputs.append({'host': host_mem, 'dev': dev_mem, 'shape': shape})
            
            logger.info(f"TensorRT engine loaded: {len(self._inputs)} inputs, {len(self._outputs)} outputs")
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize TensorRT: {e}")
            return False
    
    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        if not self._initialized:
            raise RuntimeError("TensorRT backend not initialized")
        
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
        
        return self._outputs[0]['host'].copy()
    
    def cleanup(self) -> None:
        self._initialized = False
        logger.info("TensorRT backend cleaned up")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
