import sys 
import cv2
import numpy as np
from ultralytics import YOLO
import torch

# Load YOLO model once (global)
MODEL_PATH = "/home/harshit/coding/av/model/best.pt"
CONF_THRES = 0.4
IOU_THRES = 0.45
IMG_SIZE = 640

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
model = YOLO(MODEL_PATH)

def get_device():
    """Return the device being used for inference"""
    return device

def find_hurdle_center(frame, draw=True):
    """
    Detect hurdle using YOLO and return the center coordinates.
    
    Args:
        frame: Input frame from camera
        draw: Whether to draw bounding box on frame
    
    Returns:
        center: Tuple (x, y) of hurdle center, or None if no detection
    """
    center = None
    
    # YOLO inference
    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        iou=IOU_THRES,
        imgsz=IMG_SIZE,
        device=device,
        verbose=False
    )
    
    # Process detections
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            continue
        
        # Get the first (or largest) detection
        # If multiple hurdles, you can modify this logic
        best_box = None
        best_area = 0
        
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            
            # Keep the largest detection
            if area > best_area:
                best_area = area
                best_box = box
        
        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])
            conf = float(best_box.conf[0])
            cls = int(best_box.cls[0])
            label = result.names[cls]
            
            # Calculate center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            center = (center_x, center_y)
            
            if draw:
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                
                # Draw label
                text = f"{label} {conf:.2f}"
                cv2.putText(
                    frame,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
    
    return center