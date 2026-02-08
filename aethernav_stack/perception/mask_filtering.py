"""
Mask filtering utilities for road segmentation.

Provides morphological operations to clean segmentation masks
and reduce noise for more robust centerline extraction.
"""

import cv2
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SegmentationConfig


def clean_mask(
    mask: np.ndarray,
    kernel_size: int = 7,
    iterations: int = 2
) -> np.ndarray:
    """
    Apply morphological operations to clean the mask and reduce noise.
    
    Steps:
    1. Morphological closing - fills small holes in the road
    2. Morphological opening - removes small noise blobs
    3. Keep only the largest connected component (the main road)
    
    Args:
        mask: Binary mask (0 or 1 values, any dtype)
        kernel_size: Size of the morphology kernel
        iterations: Number of close/open iterations
        
    Returns:
        Cleaned binary mask (np.uint8, 0 or 1 values)
    """
    # Ensure mask is uint8 for OpenCV operations
    mask_uint8 = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
    
    # Create elliptical kernel for morphological operations
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, 
        (kernel_size, kernel_size)
    )
    
    # Close: fill small holes in the road region
    closed = cv2.morphologyEx(
        mask_uint8, 
        cv2.MORPH_CLOSE, 
        kernel, 
        iterations=iterations
    )
    
    # Open: remove small noise blobs
    opened = cv2.morphologyEx(
        closed, 
        cv2.MORPH_OPEN, 
        kernel, 
        iterations=iterations
    )
    
    # Keep only the largest connected component (the main road)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        opened, 
        connectivity=8
    )
    
    if num_labels <= 1:
        # No components found (besides background)
        return (opened > 0).astype(np.uint8)
    
    # Find the largest component (excluding background at index 0)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    # Create mask with only the largest component
    clean = (labels == largest_label).astype(np.uint8)
    
    return clean


def clean_mask_from_config(mask: np.ndarray, config: "SegmentationConfig") -> np.ndarray:
    """
    Clean mask using parameters from config.
    
    Args:
        mask: Binary segmentation mask
        config: Segmentation configuration
        
    Returns:
        Cleaned binary mask
    """
    return clean_mask(
        mask,
        kernel_size=config.morphology_kernel_size,
        iterations=config.morphology_iterations
    )
