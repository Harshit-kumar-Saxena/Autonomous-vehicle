import sys 
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

import triangulation as tri
import yolo_detection as yolo

# -------------------------------
# STEREO CAMERA SETUP
# -------------------------------
print("[INFO] Opening cameras...")
cap_right = cv2.VideoCapture(4, cv2.CAP_V4L2)
cap_left  = cv2.VideoCapture(2, cv2.CAP_V4L2)

# Check if cameras opened successfully
if not cap_right.isOpened():
    print("[ERROR] Cannot open right camera (ID: 4)")
    exit()
if not cap_left.isOpened():
    print("[ERROR] Cannot open left camera (ID: 2)")
    exit()

print("[INFO] Cameras opened successfully")

# -------------------------------
# CALIBRATION PARAMETERS
# -------------------------------
B = 15                # Distance between cameras (cm)
f = 6                # Camera lens focal length (mm)
alpha = 95.0        # Camera field of view in horizontal plane (degrees)

# -------------------------------
# FPS TRACKING
# -------------------------------
prev_time = 0

print(f"[INFO] Using device: {yolo.get_device()}")
print(f"[INFO] Model loaded from: {yolo.MODEL_PATH}")
print("[INFO] Starting stereo hurdle detection...")
print("[INFO] Press 'q' to quit")
print("-" * 50)

# -------------------------------
# MAIN LOOP
# -------------------------------
count = -1
while True:
    count += 1
    ret_right, frame_right = cap_right.read()
    ret_left, frame_left = cap_left.read()
    
    if not ret_right or not ret_left:
        print("[ERROR] Cannot read from cameras")
        break
    
    # Detect hurdle center in both frames
    center_right = yolo.find_hurdle_center(frame_right, draw=True)
    center_left = yolo.find_hurdle_center(frame_left, draw=True)
    
    # Check if hurdle detected in both frames
    if center_right is None or center_left is None:
        cv2.putText(frame_right, "TRACKING LOST", (75, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_left, "TRACKING LOST", (75, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Calculate depth using triangulation
        depth = tri.find_depth(center_right, center_left, 
                               frame_right, frame_left, B, f, alpha)
        
        cv2.putText(frame_right, "TRACKING", (75, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame_left, "TRACKING", (75, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        
        cv2.putText(frame_right, f"Distance: {round(depth, 2)} cm", (200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        cv2.putText(frame_left, f"Distance: {round(depth, 2)} cm", (200, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 252, 0), 2)
        
        print(f"Depth: {depth:.2f} cm")
    
    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time
    
    cv2.putText(frame_right, f"FPS: {int(fps)}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(frame_left, f"FPS: {int(fps)}", (20, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
    # Display frames
    cv2.imshow("Right Camera - Hurdle Detection", frame_right)
    cv2.imshow("Left Camera - Hurdle Detection", frame_left)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# CLEANUP
# -------------------------------
cap_right.release()
cap_left.release()
cv2.destroyAllWindows()
print("[INFO] Program terminated")