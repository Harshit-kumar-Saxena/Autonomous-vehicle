"""
STEREO HURDLE DETECTION WITH RECTIFICATION
==========================================
This script uses calibrated stereo cameras to detect hurdles and measure depth.

THEORY:
-------
1. Loads stereo calibration parameters
2. Rectifies both camera images (aligns epipolar lines horizontally)
3. Detects hurdle in both rectified images using YOLO
4. Computes disparity from center positions
5. Converts disparity to real-world depth using triangulation

WHY RECTIFICATION?
------------------
After rectification:
- Epipolar lines are horizontal
- Corresponding points have same Y coordinate
- Only X coordinate differs (disparity)
- Depth = (baseline × focal_length) / disparity

Without rectification, disparity calculation would be incorrect!

USAGE:
------
1. First run stereo_calibration.py to create calibration file
2. Run this script
3. Show hurdle to both cameras
4. Real-time depth measurement displayed

DEPTH CALCULATION:
------------------
For rectified images:
    disparity = x_left - x_right
    depth_cm = (baseline_cm × focal_pixels) / disparity
    
Where focal_pixels comes from calibrated projection matrix P1[0,0]
"""

import sys 
import cv2
import numpy as np
import time
import torch
from ultralytics import YOLO

import yolo_detection as yolo

# ==============================================================================
# LOAD CALIBRATION DATA
# ==============================================================================

print("\n" + "="*70)
print("STEREO HURDLE DETECTION - RECTIFIED")
print("="*70)

try:
    print("\n[INFO] Loading calibration data...")
    calib = np.load('stereo_calibration.npz')
    
    # Camera intrinsics
    K1 = calib['K1']  # Left camera matrix
    D1 = calib['D1']  # Left distortion coefficients
    K2 = calib['K2']  # Right camera matrix
    D2 = calib['D2']  # Right distortion coefficients
    
    # Stereo parameters
    R = calib['R']   # Rotation matrix
    T = calib['T']   # Translation vector
    
    # Rectification parameters
    R1 = calib['R1']  # Left rectification rotation
    R2 = calib['R2']  # Right rectification rotation
    P1 = calib['P1']  # Left rectified projection matrix
    P2 = calib['P2']  # Right rectified projection matrix
    Q = calib['Q']    # Disparity-to-depth mapping matrix
    
    img_shape = tuple(calib['img_shape'])
    
    print("  ✓ Calibration loaded successfully")
    print(f"  Image size: {img_shape}")
    print(f"  Baseline: {abs(T[0][0]):.2f} cm")
    print(f"  Left focal length: {P1[0,0]:.2f} pixels")
    print(f"  Right focal length: {P2[0,0]:.2f} pixels")
    
except FileNotFoundError:
    print("\n[ERROR] Calibration file not found!")
    print("Please run 'stereo_calibration.py' first to calibrate your cameras.")
    sys.exit(1)

# ==============================================================================
# COMPUTE RECTIFICATION MAPS
# ==============================================================================

print("\n[INFO] Computing rectification maps...")

# These maps transform distorted images to rectified images
# initUndistortRectifyMap computes the mapping for every pixel
map1_left, map2_left = cv2.initUndistortRectifyMap(
    K1, D1, R1, P1, img_shape, cv2.CV_32FC1)

map1_right, map2_right = cv2.initUndistortRectifyMap(
    K2, D2, R2, P2, img_shape, cv2.CV_32FC1)

print("  ✓ Rectification maps computed")

# ==============================================================================
# DEPTH CALCULATION FUNCTION
# ==============================================================================

def calculate_depth_rectified(center_left, center_right, P1, Q):
    """
    Calculate depth from rectified image coordinates.
    
    THEORY:
    -------
    In rectified images, corresponding points have same y-coordinate.
    Disparity is simply the difference in x-coordinates:
        d = x_left - x_right
    
    Depth is calculated using the projection matrix P1:
        Z = (baseline × focal_length) / disparity
        Z = (baseline × P1[0,0]) / d
    
    Where:
        - baseline = distance between cameras (from P2)
        - P1[0,0] = focal length in rectified coordinates
        - d = disparity in pixels
    
    Alternative using Q matrix:
        The Q matrix maps (x, y, disparity) to (X, Y, Z):
        [X]       [1  0    0      -cx          ]   [x]
        [Y]   =   [0  1    0      -cy          ] × [y]
        [Z]       [0  0    0       f           ]   [d]
        [W]       [0  0  -1/Tx  (cx-cx')/Tx    ]   [1]
        
        Then: X=X/W, Y=Y/W, Z=Z/W gives real-world coordinates
    
    Args:
        center_left: (x, y) in left rectified image
        center_right: (x, y) in right rectified image
        P1: Left projection matrix
        Q: Disparity-to-depth matrix
    
    Returns:
        depth: Distance in cm
    """
    x_left, y_left = center_left
    x_right, y_right = center_right
    
    # Calculate disparity (in rectified images, this is just x difference)
    disparity = x_left - x_right
    
    if disparity <= 0:
        print("  [WARNING] Invalid disparity (<=0), object too far or misdetected")
        return None
    
    # Method 1: Using projection matrices
    # baseline is extracted from P2[0,3] and P1[0,0]
    # P2[0,3] = -fx * baseline
    baseline = -P2[0, 3] / P1[0, 0]  # in cm (if T was in cm)
    focal_length = P1[0, 0]  # in pixels
    
    depth = (baseline * focal_length) / disparity
    
    # Method 2: Using Q matrix (alternative, should give same result)
    # point_3d = cv2.perspectiveTransform(
    #     np.array([[[x_left, y_left, disparity]]], dtype=np.float32), Q)
    # depth_alt = point_3d[0][0][2]
    
    return abs(depth)

# ==============================================================================
# CAMERA SETUP
# ==============================================================================

LEFT_CAMERA_ID = 2
RIGHT_CAMERA_ID = 4

print("\n[INFO] Opening cameras...")
cap_left = cv2.VideoCapture(LEFT_CAMERA_ID, cv2.CAP_V4L2)
cap_right = cv2.VideoCapture(RIGHT_CAMERA_ID, cv2.CAP_V4L2)

if not cap_left.isOpened():
    print(f"[ERROR] Cannot open left camera (ID: {LEFT_CAMERA_ID})")
    sys.exit(1)
if not cap_right.isOpened():
    print(f"[ERROR] Cannot open right camera (ID: {RIGHT_CAMERA_ID})")
    sys.exit(1)

print("  ✓ Cameras opened successfully")

# ==============================================================================
# YOLO MODEL INFO
# ==============================================================================

print(f"\n[INFO] YOLO device: {yolo.get_device()}")
print(f"[INFO] YOLO model: {yolo.MODEL_PATH}")

print("\n" + "="*70)
print("Starting real-time depth measurement...")
print("Press 'q' to quit")
print("="*70 + "\n")

# ==============================================================================
# MAIN LOOP
# ==============================================================================

prev_time = time.time()
frame_count = 0

try:
    while True:
        frame_count += 1
        
        # Capture frames
        ret_left, frame_left_raw = cap_left.read()
        ret_right, frame_right_raw = cap_right.read()
        
        if not ret_left or not ret_right:
            print("[ERROR] Cannot read from cameras")
            break
        
        # ======================================================================
        # RECTIFY IMAGES
        # ======================================================================
        # This is the KEY step - transforms raw distorted images to rectified
        # images where epipolar lines are horizontal
        
        frame_left = cv2.remap(frame_left_raw, map1_left, map2_left, 
                               cv2.INTER_LINEAR)
        frame_right = cv2.remap(frame_right_raw, map1_right, map2_right, 
                                cv2.INTER_LINEAR)
        
        # ======================================================================
        # DETECT HURDLE IN RECTIFIED IMAGES
        # ======================================================================
        
        center_left = yolo.find_hurdle_center(frame_left, draw=True)
        center_right = yolo.find_hurdle_center(frame_right, draw=True)
        
        # ======================================================================
        # CALCULATE DEPTH
        # ======================================================================
        
        if center_left is None or center_right is None:
            # No detection in one or both cameras
            cv2.putText(frame_left, "TRACKING LOST", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame_right, "TRACKING LOST", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        else:
            # Both cameras detected hurdle
            depth = calculate_depth_rectified(center_left, center_right, P1, Q)
            
            if depth is not None and depth > 0:
                # Valid depth measurement
                cv2.putText(frame_left, "TRACKING", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame_right, "TRACKING", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                # Display depth
                depth_text = f"Depth: {depth:.1f} cm ({depth/100:.2f} m)"
                cv2.putText(frame_left, depth_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame_right, depth_text, (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Display disparity info
                disparity = center_left[0] - center_right[0]
                disp_text = f"Disparity: {disparity:.1f} px"
                cv2.putText(frame_left, disp_text, (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display coordinates
                coord_text = f"L:({center_left[0]},{center_left[1]}) R:({center_right[0]},{center_right[1]})"
                cv2.putText(frame_left, coord_text, (20, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                depth -= 30
                # Print to console every 10 frames
                if frame_count % 10 == 0:
                    print(f"Depth: {depth:6.1f} cm | Disparity: {disparity:5.1f} px | "
                          f"Left: {center_left} | Right: {center_right}")
            else:
                cv2.putText(frame_left, "INVALID DISPARITY", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
                cv2.putText(frame_right, "INVALID DISPARITY", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # ======================================================================
        # DRAW EPIPOLAR LINES (for visualization)
        # ======================================================================
        # In rectified images, epipolar lines are horizontal
        # Draw a few horizontal lines to show rectification worked
        
        if frame_count % 30 == 0:  # Only draw occasionally to avoid clutter
            for y in range(100, frame_left.shape[0], 100):
                cv2.line(frame_left, (0, y), (frame_left.shape[1], y), 
                        (255, 0, 0), 1)
                cv2.line(frame_right, (0, y), (frame_right.shape[1], y), 
                        (255, 0, 0), 1)
        
        # ======================================================================
        # FPS CALCULATION
        # ======================================================================
        
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        
        cv2.putText(frame_left, f"FPS: {int(fps)}", (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        cv2.putText(frame_right, f"FPS: {int(fps)}", (20, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # ======================================================================
        # DISPLAY
        # ======================================================================
        
        cv2.imshow("Left Camera - RECTIFIED", frame_left)
        cv2.imshow("Right Camera - RECTIFIED", frame_right)
        
        # Exit on 'q'findChessboardCorners
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n[INFO] Interrupted by user")

# ==============================================================================
# CLEANUP
# ==============================================================================

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print("\n[INFO] Program terminated")
print("="*70)