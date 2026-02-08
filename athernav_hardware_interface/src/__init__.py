
from .hardware_interface import HardwareInterface
from .config import CommandID, AckCode, LogLevel
from .exceptions import (
    AetherNavException,
    SerialConnectionError,
    CommandTimeoutError,
    InvalidResponseError,
    BoardError,
    InvalidCommandError,
    BoardBusyError
)
from .board_utils import Logger, BoardStatus

__version__ = "1.0.0"
__author__ = "Nitin Maurya"

__all__ = [
    # Main interface
    "HardwareInterface",
    
    # Configuration
    "CommandID",
    "AckCode", 
    "LogLevel",
    
    # Exceptions
    "AetherNavException",
    "SerialConnectionError",
    "CommandTimeoutError",
    "InvalidResponseError",
    "BoardError",
    "InvalidCommandError",
    "BoardBusyError",
    
    # Utilities
    "Logger",
    "BoardStatus",
]
