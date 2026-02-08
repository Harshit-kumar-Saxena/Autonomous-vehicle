"""
Temporal smoothing for centerline detection.

Provides frame-to-frame smoothing to reduce jitter in centerline
extraction across video frames.
"""

from collections import deque
from typing import List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SegmentationConfig


class TemporalSmoother:
    """
    Temporal smoothing for centerline to reduce frame-to-frame jitter.
    
    Maintains a sliding window of historical centerlines and applies
    weighted averaging to produce a smoother output.
    """
    
    def __init__(self, window_size: int = 5, alpha: float = 0.7):
        """
        Initialize temporal smoother.
        
        Args:
            window_size: Number of frames to keep in history
            alpha: Weight for current frame (0-1), remainder for history
        """
        self.history: deque = deque(maxlen=window_size)
        self.alpha = alpha
        self.window_size = window_size
    
    def smooth(
        self, 
        current_centerline: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Apply temporal smoothing to the centerline.
        
        Args:
            current_centerline: Current frame's centerline points
            
        Returns:
            Smoothed centerline points
        """
        if not current_centerline:
            return current_centerline
        
        self.history.append(current_centerline)
        
        if len(self.history) < 2:
            return current_centerline
        
        # Align and average centerlines from history
        smoothed = []
        
        # Use the current frame's y-coordinates as reference
        for i, (x, y) in enumerate(current_centerline):
            # Weighted average with history
            total_weight = self.alpha
            weighted_x = x * self.alpha
            
            weight_decay = (1 - self.alpha) / len(self.history)
            
            for hist_cl in list(self.history)[:-1]:  # Exclude current
                # Find closest point in historical centerline
                closest_x = None
                min_dist = float('inf')
                
                for hx, hy in hist_cl:
                    dist = abs(hy - y)
                    if dist < min_dist:
                        min_dist = dist
                        closest_x = hx
                
                if closest_x is not None and min_dist < 20:  # Only if close enough
                    weighted_x += closest_x * weight_decay
                    total_weight += weight_decay
            
            smoothed_x = int(weighted_x / total_weight) if total_weight > 0 else x
            smoothed.append((smoothed_x, y))
        
        return smoothed
    
    def reset(self) -> None:
        """Clear history buffer."""
        self.history.clear()
    
    @classmethod
    def from_config(cls, config: "SegmentationConfig") -> "TemporalSmoother":
        """
        Create smoother from config.
        
        Args:
            config: Segmentation configuration
            
        Returns:
            Configured temporal smoother
        """
        return cls(
            window_size=config.temporal_window,
            alpha=config.temporal_alpha
        )
