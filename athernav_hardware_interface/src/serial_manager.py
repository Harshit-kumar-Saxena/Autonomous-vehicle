import serial
import time
from typing import Optional, Callable
from .config import (
    FRAME_FORMAT,
    FRAME_SIZE,
    DEFAULT_PORT,
    DEFAULT_BAUDRATE,
    DEFAULT_TIMEOUT,
)
from .exceptions import SerialConnectionError, CommandTimeoutError
from .board_utils import Logger


class SerialManager:

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT,
        logger: Optional[Logger] = None,
    ):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.logger = logger or Logger()
        self._serial: Optional[serial.Serial] = None
        self._is_connected = False

    def connect(self) -> bool:

        try:
            self._serial = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self.timeout
            )
            self._is_connected = True
            self.logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            return True

        except serial.SerialException as e:
            error_msg = f"Serial connection failed: {e}"
            self.logger.error(error_msg)
            raise SerialConnectionError(error_msg, self.port)

    def disconnect(self):
        if self._serial and self._serial.is_open:
            self._serial.close()
            self._is_connected = False
            self.logger.info("Serial connection closed")

    def is_connected(self) -> bool:
        return self._is_connected and self._serial and self._serial.is_open

    def send_frame(self, frame: bytes) -> None:

        if not self.is_connected():
            raise SerialConnectionError("Not connected to board")

        try:
            self._serial.write(frame)
            self.logger.debug(f"Sent frame: {frame.hex()}")
        except serial.SerialException as e:
            raise SerialConnectionError(f"Failed to send frame: {e}")

    def read_frame(self, timeout: Optional[float] = None) -> Optional[bytes]:

        if not self.is_connected():
            raise SerialConnectionError("Not connected to board")

        original_timeout = self._serial.timeout
        try:
            if timeout is not None:
                self._serial.timeout = timeout

            response = self._serial.read(FRAME_SIZE)

            if len(response) != FRAME_SIZE:
                self.logger.warning(
                    f"Incomplete frame received: {len(response)}/{FRAME_SIZE} bytes"
                )
                return None

            self.logger.debug(f"Received frame: {response.hex()}")
            return response

        except serial.SerialException as e:
            raise SerialConnectionError(f"Failed to read frame: {e}")
        finally:
            self._serial.timeout = original_timeout

    def send_and_receive(
        self, frame: bytes, timeout: Optional[float] = None
    ) -> Optional[bytes]:

        self.send_frame(frame)
        return self.read_frame(timeout)

    def flush_input(self):
        if self._serial and self._serial.is_open:
            self._serial.flushInput()

    def flush_output(self):
        if self._serial and self._serial.is_open:
            self._serial.flushOutput()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        self.disconnect()
