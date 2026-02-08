"""Core types and interfaces for AetherNav stack."""

from .types import Pose2D, Twist2D, WheelVelocities, SegmentationResult, RobotState
from .interfaces import PlannerPlugin, ControllerPlugin
from .logging_config import get_logger, LogLevel
from .threading_utils import LatestHolder, StoppableThread
from .pipeline import CameraThread, PerceptionThread, ControlThread, FrameData, PerceptionData

__all__ = [
    "Pose2D",
    "Twist2D",
    "WheelVelocities",
    "SegmentationResult",
    "RobotState",
    "PlannerPlugin",
    "ControllerPlugin",
    "get_logger",
    "LogLevel",
    "LatestHolder",
    "StoppableThread",
    "CameraThread",
    "PerceptionThread",
    "ControlThread",
    "FrameData",
    "PerceptionData",
]
