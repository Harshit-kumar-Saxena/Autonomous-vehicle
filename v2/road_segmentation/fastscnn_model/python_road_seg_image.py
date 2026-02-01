import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "fastscnn_road_fp16.engine"
IMG_SIZE = (512, 256)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine(ENGINE_PATH)
context = engine.create_execution_context()

# Allocate buffers
input_shape = (1, 3, 256, 512)
output_shape = (1, 1, 256, 512)

d_input = cuda.mem_alloc(np.prod(input_shape) * np.float32().nbytes)
d_output = cuda.mem_alloc(np.prod(output_shape) * np.float32().nbytes)

bindings = [int(d_input), int(d_output)]
stream = cuda.Stream()

def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def infer(img):
    inp = preprocess(img)
    out = np.empty(output_shape, dtype=np.float32)

    cuda.memcpy_htod_async(d_input, inp, stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(out, d_output, stream)
    stream.synchronize()

    mask = (1 / (1 + np.exp(-out)) > 0.5).astype(np.uint8) * 255
    return mask[0, 0]

# -------- RUN --------
img = cv2.imread("test.jpg")
mask = infer(img)
mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

cv2.imshow("Road Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
