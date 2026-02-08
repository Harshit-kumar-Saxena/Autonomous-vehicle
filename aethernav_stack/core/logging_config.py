"""
Centralized logging configuration for the AetherNav stack.

Provides consistent logging across all modules with configurable
levels and formats.
"""

import logging
import sys
from enum import IntEnum
from typing import Optional


class LogLevel(IntEnum):
    """Log level enumeration matching Python logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# Module-level logger cache
_loggers: dict[str, logging.Logger] = {}

# Default format
_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(
    level: LogLevel = LogLevel.INFO,
    log_file: Optional[str] = None
) -> None:
    """
    Configure the root logger for the AetherNav stack.
    
    Args:
        level: Minimum log level to display
        log_file: Optional file path to write logs to
    """
    # Create formatter
    formatter = logging.Formatter(_LOG_FORMAT, _DATE_FORMAT)
    
    # Get root logger for our package
    root_logger = logging.getLogger("aethernav")
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (e.g., "perception", "controls.pid")
        
    Returns:
        Configured logger instance
    """
    full_name = f"aethernav.{name}"
    
    if full_name not in _loggers:
        _loggers[full_name] = logging.getLogger(full_name)
    
    return _loggers[full_name]


class RateLimitedLogger:
    """
    Logger wrapper that rate-limits repeated messages.
    
    Useful for high-frequency loops where you don't want to spam
    the same warning every cycle.
    """
    
    def __init__(self, logger: logging.Logger, min_interval: float = 1.0):
        """
        Args:
            logger: Underlying logger
            min_interval: Minimum seconds between identical messages
        """
        self._logger = logger
        self._min_interval = min_interval
        self._last_log_times: dict[str, float] = {}
    
    def _should_log(self, msg: str) -> bool:
        """Check if enough time has passed to log this message again."""
        import time
        now = time.time()
        last_time = self._last_log_times.get(msg, 0.0)
        
        if now - last_time >= self._min_interval:
            self._last_log_times[msg] = now
            return True
        return False
    
    def debug(self, msg: str, *args, **kwargs) -> None:
        if self._should_log(msg):
            self._logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs) -> None:
        if self._should_log(msg):
            self._logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs) -> None:
        if self._should_log(msg):
            self._logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs) -> None:
        if self._should_log(msg):
            self._logger.error(msg, *args, **kwargs)
