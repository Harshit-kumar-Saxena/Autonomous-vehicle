"""
AetherNav Autonomous Vehicle Stack

A professional-grade, nav2-inspired modular architecture for autonomous
differential-drive robot navigation.
"""

__version__ = "1.0.0"
__author__ = "AetherNav Team"

from .core.types import Pose2D, Twist2D, WheelVelocities, SegmentationResult
from .core.interfaces import PlannerPlugin, ControllerPlugin

__all__ = [
    "Pose2D",
    "Twist2D", 
    "WheelVelocities",
    "SegmentationResult",
    "PlannerPlugin",
    "ControllerPlugin",
]
