import cv2
import numpy as np
import onnxruntime as ort

# ---------------- CONFIG ----------------
ONNX_PATH = "fastscnn_road.onnx"
IMG_W, IMG_H = 512, 256
THRESH = 0.4
CAMERA_ID = 2
# ----------------------------------------

# Load ONNX model
session = ort.InferenceSession(
    ONNX_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def preprocess(frame):
    img = cv2.resize(frame, (IMG_W, IMG_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(pred, original_shape):
    pred = 1 / (1 + np.exp(-pred))
    mask = (pred > THRESH).astype(np.uint8) * 255
    mask = mask[0, 0]
    return cv2.resize(mask, (original_shape[1], original_shape[0]))

# Open camera
cap = cv2.VideoCapture(CAMERA_ID)

assert cap.isOpened(), "Camera not opened!"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    inp = preprocess(frame)
    pred = session.run([output_name], {input_name: inp})[0]
    mask = postprocess(pred, frame.shape)

    # Overlay
    overlay = frame.copy()
    overlay[mask == 255] = (0, 255, 0)

    cv2.imshow("Road Mask", mask)
    cv2.imshow("Road Segmentation", overlay)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
