"""
Centerline extraction algorithms for road segmentation.

Provides robust centerline detection from binary road masks using
row scanning, jump filtering, and polynomial/RANSAC curve fitting.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SegmentationConfig


def extract_centerline_robust(
    mask: np.ndarray,
    min_road_width: int = 50,
    max_jump_threshold: int = 80,
    skip_rows: int = 5,
    use_ransac: bool = True,
    ransac_threshold: int = 20,
    prev_centerline: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Extract centerline with robust noise handling.
    
    Algorithm:
    1. Scan rows from bottom to top
    2. Find center of road for each row
    3. Apply jump threshold to filter outliers
    4. Use polynomial or RANSAC fitting for smooth curve
    
    Args:
        mask: Binary road mask (H x W)
        min_road_width: Minimum road width to consider valid
        max_jump_threshold: Maximum allowed jump between consecutive centers
        skip_rows: Process every Nth row
        use_ransac: Whether to use RANSAC for robust fitting
        ransac_threshold: Pixel threshold for RANSAC inliers
        prev_centerline: Previous frame's centerline for continuity
        
    Returns:
        Tuple of (centerline_points, debug_info)
    """
    h, w = mask.shape
    raw_centers = []
    debug_info = {
        'valid_rows': 0,
        'invalid_rows': 0,
        'filtered_jumps': 0,
        'interpolated': 0,
        'avg_road_width': 0.0,
    }
    
    prev_center = None
    road_widths = []
    
    # Phase 1: Extract raw centers with jump filtering
    for y in range(h - 1, -1, -skip_rows):
        row = mask[y, :]
        road_pixels = np.where(row == 1)[0]
        
        if len(road_pixels) < min_road_width:
            debug_info['invalid_rows'] += 1
            raw_centers.append((None, y))  # Mark as invalid
            continue
        
        left_edge = road_pixels[0]
        right_edge = road_pixels[-1]
        road_width = right_edge - left_edge
        center_x = (left_edge + right_edge) // 2
        
        # Jump filtering: reject if jump is too large
        if prev_center is not None:
            jump = abs(center_x - prev_center)
            if jump > max_jump_threshold:
                debug_info['filtered_jumps'] += 1
                raw_centers.append((None, y))  # Mark as filtered
                continue
        
        raw_centers.append((center_x, y))
        road_widths.append(road_width)
        debug_info['valid_rows'] += 1
        prev_center = center_x
    
    # Phase 2: Get valid points
    valid_points = [(x, y) for x, y in raw_centers if x is not None]
    
    if len(valid_points) < 5:
        # Not enough points, use previous centerline or empty
        return prev_centerline if prev_centerline else [], debug_info
    
    # Phase 3: Fit curve for smoothing
    if use_ransac and len(valid_points) > 10:
        centerline_points = fit_centerline_ransac(valid_points, ransac_threshold)
    else:
        centerline_points = fit_centerline_polynomial(valid_points)
    
    # Calculate statistics
    if road_widths:
        debug_info['avg_road_width'] = float(np.mean(road_widths))
    
    return centerline_points, debug_info


def fit_centerline_polynomial(
    points: List[Tuple[int, int]], 
    degree: int = 3
) -> List[Tuple[int, int]]:
    """
    Fit a polynomial curve to the centerline points.
    
    Args:
        points: List of (x, y) centerline points
        degree: Polynomial degree
        
    Returns:
        Smoothed centerline points
    """
    if len(points) < degree + 1:
        return points
    
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    
    try:
        # Fit polynomial: x = f(y)
        coeffs = np.polyfit(y_coords, x_coords, degree)
        poly = np.poly1d(coeffs)
        
        # Generate smooth centerline
        y_smooth = np.linspace(y_coords.min(), y_coords.max(), len(points))
        x_smooth = poly(y_smooth)
        
        return [(int(x), int(y)) for x, y in zip(x_smooth, y_smooth)]
    except Exception:
        return points


def fit_centerline_ransac(
    points: List[Tuple[int, int]],
    residual_threshold: int = 20,
    max_iterations: int = 100
) -> List[Tuple[int, int]]:
    """
    Use RANSAC-like approach to fit a robust centerline.
    Rejects outliers and fits a polynomial to inliers.
    
    Args:
        points: List of (x, y) centerline points
        residual_threshold: Pixel threshold for RANSAC inliers
        max_iterations: Maximum RANSAC iterations
        
    Returns:
        Smoothed centerline points (outliers removed)
    """
    if len(points) < 10:
        return fit_centerline_polynomial(points)
    
    x_coords = np.array([p[0] for p in points])
    y_coords = np.array([p[1] for p in points])
    
    best_inliers = None
    best_inlier_count = 0
    
    for _ in range(max_iterations):
        # Randomly sample 4 points to fit a cubic
        sample_idx = np.random.choice(len(points), min(4, len(points)), replace=False)
        sample_x = x_coords[sample_idx]
        sample_y = y_coords[sample_idx]
        
        try:
            coeffs = np.polyfit(sample_y, sample_x, 3)
            poly = np.poly1d(coeffs)
            
            # Calculate residuals
            predicted_x = poly(y_coords)
            residuals = np.abs(x_coords - predicted_x)
            
            # Find inliers
            inliers = residuals < residual_threshold
            inlier_count = np.sum(inliers)
            
            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
        except Exception:
            continue
    
    if best_inliers is None or best_inlier_count < 5:
        return fit_centerline_polynomial(points)
    
    # Refit using all inliers
    inlier_points = [
        (x_coords[i], y_coords[i]) 
        for i in range(len(points)) 
        if best_inliers[i]
    ]
    return fit_centerline_polynomial(inlier_points, degree=3)


def extract_centerline_from_config(
    mask: np.ndarray,
    config: "SegmentationConfig",
    prev_centerline: Optional[List[Tuple[int, int]]] = None
) -> Tuple[List[Tuple[int, int]], Dict[str, Any]]:
    """
    Extract centerline using parameters from config.
    
    Args:
        mask: Binary road mask
        config: Segmentation configuration
        prev_centerline: Previous frame's centerline
        
    Returns:
        Tuple of (centerline_points, debug_info)
    """
    return extract_centerline_robust(
        mask,
        min_road_width=config.min_road_width,
        max_jump_threshold=config.max_jump_threshold,
        skip_rows=config.centerline_skip_rows,
        use_ransac=config.use_ransac_fit,
        ransac_threshold=config.ransac_residual_threshold,
        prev_centerline=prev_centerline
    )


def draw_centerline(
    image: np.ndarray,
    centerline_points: List[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 0, 255),
    thickness: int = 3,
    draw_points: bool = True,
    point_interval: int = 5
) -> np.ndarray:
    """
    Draw the centerline on an image.
    
    Args:
        image: BGR image to draw on
        centerline_points: List of (x, y) centerline points
        color: Line color in BGR
        thickness: Line thickness
        draw_points: Whether to draw individual points
        point_interval: Draw a point every N points
        
    Returns:
        Image with centerline drawn
    """
    if len(centerline_points) < 2:
        return image
    
    result = image.copy()
    
    # Draw the centerline as connected points
    for i in range(len(centerline_points) - 1):
        pt1 = centerline_points[i]
        pt2 = centerline_points[i + 1]
        cv2.line(result, pt1, pt2, color, thickness)
    
    # Draw points at regular intervals
    if draw_points:
        for i, pt in enumerate(centerline_points):
            if i % point_interval == 0:
                cv2.circle(result, pt, 4, (0, 255, 255), -1)
    
    return result
