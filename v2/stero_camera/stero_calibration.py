"""
STEREO CAMERA CALIBRATION
=========================
This script calibrates your stereo camera setup using a chessboard pattern.

THEORY:
-------
1. Captures images of chessboard from both cameras simultaneously
2. Detects corners in each image
3. Computes camera intrinsics (K, D) for each camera
4. Computes stereo extrinsics (R, T) - rotation and translation between cameras
5. Computes rectification parameters to align epipolar lines horizontally
6. Saves all calibration data to file

USAGE:
------
1. Print a 7x9 chessboard with 2.5cm squares
2. Run this script
3. Show the chessboard to both cameras from different angles
4. Press SPACE to capture image pair (capture 20-30 pairs)
5. Press 'q' when done capturing
6. Script will calibrate and save parameters

OUTPUT:
-------
Creates 'stereo_calibration.npz' containing:
- K1, D1: Left camera intrinsics
- K2, D2: Right camera intrinsics  
- R, T: Rotation and translation between cameras
- E, F: Essential and fundamental matrices
- R1, R2, P1, P2, Q: Rectification parameters
"""

import cv2
import numpy as np
import glob
import os

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Chessboard dimensions (INNER corners)
CHESSBOARD_SIZE = (8,6)  # (columns, rows) of inner corners
SQUARE_SIZE = 2.5  # Size of each square in cm

# Camera IDs
LEFT_CAMERA_ID = 6
RIGHT_CAMERA_ID = 4

# Calibration settings
MIN_IMAGES = 20  # Minimum number of image pairs needed
CAPTURE_FOLDER = "calibration_images"

# ==============================================================================
# PREPARE OBJECT POINTS
# ==============================================================================

def prepare_object_points():
    """
    Create 3D coordinates of chessboard corners in real world space.
    
    For a 7x9 chessboard with 2.5cm squares:
    - Corner (0,0) is at (0, 0, 0)
    - Corner (1,0) is at (2.5, 0, 0)
    - Corner (0,1) is at (0, 2.5, 0)
    - All corners lie on Z=0 plane
    
    Returns:
        objp: Array of shape (63, 3) with 3D coordinates
    """
    objp = np.zeros((CHESSBOARD_SIZE[1] * CHESSBOARD_SIZE[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 
                           0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE  # Scale by square size
    return objp

# ==============================================================================
# IMAGE CAPTURE
# ==============================================================================

def capture_calibration_images():
    """
    Capture image pairs from both cameras for calibration.
    
    User shows chessboard to cameras and presses SPACE to capture.
    Captures are saved to disk for later processing.
    """
    print("\n" + "="*70)
    print("STEREO CALIBRATION - IMAGE CAPTURE")
    print("="*70)
    print(f"\n[INFO] Opening cameras...")
    
    cap_left = cv2.VideoCapture(LEFT_CAMERA_ID, cv2.CAP_V4L2)
    cap_right = cv2.VideoCapture(RIGHT_CAMERA_ID, cv2.CAP_V4L2)
    
    if not cap_left.isOpened() or not cap_right.isOpened():
        print("[ERROR] Cannot open cameras!")
        return False
    
    print(f"[INFO] Cameras opened successfully")
    print(f"[INFO] Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners")
    print(f"[INFO] Square size: {SQUARE_SIZE} cm")
    print("\nINSTRUCTIONS:")
    print("  1. Show the chessboard to both cameras")
    print("  2. Press SPACE to capture image pair")
    print(f"  3. Capture at least {MIN_IMAGES} pairs from different angles")
    print("  4. Vary distance, tilt, rotation for best results")
    print("  5. Press 'q' when done\n")
    print("-"*70)
    
    # Create folder for images
    os.makedirs(CAPTURE_FOLDER, exist_ok=True)
    
    count = 0
    objp = prepare_object_points()
    
    while True:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not ret_left or not ret_right:
            print("[ERROR] Cannot read from cameras")
            break
        
        # Convert to grayscale for corner detection
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret_left_corners, corners_left = cv2.findChessboardCorners(
            gray_left, CHESSBOARD_SIZE, None)
        ret_right_corners, corners_right = cv2.findChessboardCorners(
            gray_right, CHESSBOARD_SIZE, None)
        
        # Draw corners if found
        display_left = frame_left.copy()
        display_right = frame_right.copy()
        
        if ret_left_corners:
            cv2.drawChessboardCorners(display_left, CHESSBOARD_SIZE, 
                                     corners_left, ret_left_corners)
            cv2.putText(display_left, "Pattern Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_left, "No Pattern", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if ret_right_corners:
            cv2.drawChessboardCorners(display_right, CHESSBOARD_SIZE, 
                                     corners_right, ret_right_corners)
            cv2.putText(display_right, "Pattern Detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_right, "No Pattern", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show capture count
        cv2.putText(display_left, f"Captured: {count}/{MIN_IMAGES}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(display_right, f"Captured: {count}/{MIN_IMAGES}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Left Camera - Calibration", display_left)
        cv2.imshow("Right Camera - Calibration", display_right)
        
        key = cv2.waitKey(1) & 0xFF
        
        # SPACE key to capture
        if key == ord(' '):
            if ret_left_corners and ret_right_corners:
                # Save images
                cv2.imwrite(f"{CAPTURE_FOLDER}/left_{count:02d}.png", frame_left)
                cv2.imwrite(f"{CAPTURE_FOLDER}/right_{count:02d}.png", frame_right)
                count += 1
                print(f"[CAPTURED] Image pair {count}")
            else:
                print("[WARNING] Chessboard not detected in both cameras!")
        
        # 'q' to quit
        elif key == ord('q'):
            if count >= MIN_IMAGES:
                print(f"\n[INFO] Captured {count} image pairs")
                break
            else:
                print(f"\n[WARNING] Only {count} images captured. Need at least {MIN_IMAGES}.")
                print("Continue capturing or press 'q' again to quit anyway.")
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
    
    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    
    return count >= MIN_IMAGES

# ==============================================================================
# CALIBRATION COMPUTATION
# ==============================================================================

def calibrate_stereo_cameras():
    """
    Perform stereo calibration using captured images.
    
    STEPS:
    1. Load all captured image pairs
    2. Detect chessboard corners in each
    3. Calibrate each camera individually (intrinsics)
    4. Calibrate stereo setup (extrinsics)
    5. Compute rectification parameters
    6. Save all parameters
    
    MATHEMATICS:
    -----------
    Individual calibration solves:
        [u]   [fx  0  cx]   [r11 r12 r13 tx]   [X]
        [v] = [0  fy cy] * [r21 r22 r23 ty] * [Y]
        [1]   [0  0  1 ]   [r31 r32 r33 tz]   [Z]
                                               [1]
    
    For K (intrinsics) and distortion coefficients D.
    
    Stereo calibration solves for R, T where:
        P_right = R * P_left + T
    
    Rectification creates rotation matrices R1, R2 such that:
        - Both cameras share same image plane
        - Epipolar lines are horizontal
        - Rectified projection matrices: P1, P2
        - Disparity-to-depth matrix: Q
    """
    print("\n" + "="*70)
    print("STEREO CALIBRATION - COMPUTING PARAMETERS")
    print("="*70)
    
    # Load images
    left_images = sorted(glob.glob(f"{CAPTURE_FOLDER}/left_*.png"))
    right_images = sorted(glob.glob(f"{CAPTURE_FOLDER}/right_*.png"))
    
    if len(left_images) != len(right_images):
        print("[ERROR] Mismatch in number of left and right images!")
        return False
    
    print(f"\n[INFO] Found {len(left_images)} image pairs")
    
    # Prepare object points
    objp = prepare_object_points()
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints_left = []  # 2D points in left camera
    imgpoints_right = []  # 2D points in right camera
    
    print("[INFO] Detecting chessboard corners...")
    
    # Criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    img_shape = None
    
    for left_path, right_path in zip(left_images, right_images):
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        if img_shape is None:
            img_shape = gray_left.shape[::-1]
        
        # Find corners
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, CHESSBOARD_SIZE, None)
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, CHESSBOARD_SIZE, None)
        
        if ret_left and ret_right:
            # Refine corner positions to sub-pixel accuracy
            corners_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
            
            print(f"  ✓ {os.path.basename(left_path)}")
        else:
            print(f"  ✗ {os.path.basename(left_path)} - corners not found")
    
    print(f"\n[INFO] Using {len(objpoints)} valid image pairs for calibration")
    
    # ===========================================================================
    # STEP 1: Calibrate left camera
    # ===========================================================================
    print("\n[INFO] Calibrating LEFT camera...")
    ret_left, K1, D1, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, img_shape, None, None)
    
    print(f"  RMS reprojection error: {ret_left:.4f} pixels")
    print(f"  Focal length (fx, fy): ({K1[0,0]:.2f}, {K1[1,1]:.2f}) pixels")
    print(f"  Principal point (cx, cy): ({K1[0,2]:.2f}, {K1[1,2]:.2f})")
    
    # ===========================================================================
    # STEP 2: Calibrate right camera
    # ===========================================================================
    print("\n[INFO] Calibrating RIGHT camera...")
    ret_right, K2, D2, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, img_shape, None, None)
    
    print(f"  RMS reprojection error: {ret_right:.4f} pixels")
    print(f"  Focal length (fx, fy): ({K2[0,0]:.2f}, {K2[1,1]:.2f}) pixels")
    print(f"  Principal point (cx, cy): ({K2[0,2]:.2f}, {K2[1,2]:.2f})")
    
    # ===========================================================================
    # STEP 3: Stereo calibration
    # ===========================================================================
    print("\n[INFO] Computing STEREO parameters...")
    
    # Flags for stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC  # Keep K1, K2, D1, D2 fixed
    
    ret_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        K1, D1, K2, D2, img_shape,
        criteria=criteria,
        flags=flags
    )
    
    print(f"  RMS reprojection error: {ret_stereo:.4f} pixels")
    print(f"  Baseline (X): {T[0][0]:.2f} cm")
    print(f"  Vertical offset (Y): {T[1][0]:.2f} cm")
    print(f"  Depth offset (Z): {T[2][0]:.2f} cm")
    
    # ===========================================================================
    # STEP 4: Stereo rectification
    # ===========================================================================
    print("\n[INFO] Computing RECTIFICATION parameters...")
    
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        K1, D1, K2, D2, img_shape, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0  # 0 = crop to valid pixels, 1 = keep all pixels
    )
    
    print("  Rectification complete")
    print(f"  Left ROI: {roi_left}")
    print(f"  Right ROI: {roi_right}")
    
    # ===========================================================================
    # STEP 5: Save calibration data
    # ===========================================================================
    print("\n[INFO] Saving calibration data...")
    
    np.savez('stereo_calibration.npz',
             K1=K1, D1=D1,  # Left camera intrinsics
             K2=K2, D2=D2,  # Right camera intrinsics
             R=R, T=T,      # Rotation and translation
             E=E, F=F,      # Essential and fundamental matrices
             R1=R1, R2=R2,  # Rectification rotations
             P1=P1, P2=P2,  # Rectified projection matrices
             Q=Q,           # Disparity-to-depth matrix
             roi_left=roi_left,
             roi_right=roi_right,
             img_shape=img_shape)
    
    print("  Saved to 'stereo_calibration.npz'")
    print("\n" + "="*70)
    print("CALIBRATION COMPLETE!")
    print("="*70)
    print("\nYou can now use the calibration data for depth estimation.")
    print("Next step: Run the rectified stereo detection script.\n")
    
    return True

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":    
    # Check if calibration images already exist
    if os.path.exists(CAPTURE_FOLDER) and len(os.listdir(CAPTURE_FOLDER)) > 0:
        print(f"[INFO] Found existing calibration images in '{CAPTURE_FOLDER}'")
        response = input("Use existing images? (y/n): ").lower()
        if response != 'y':
            print("[INFO] Clearing old images...")
            for f in glob.glob(f"{CAPTURE_FOLDER}/*.png"):
                os.remove(f)
            capture_calibration_images()
    else:
        capture_calibration_images()
    
    # Perform calibration
    calibrate_stereo_cameras()