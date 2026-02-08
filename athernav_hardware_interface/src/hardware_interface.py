from typing import Optional, Dict, Any, Tuple
from .serial_manager import SerialManager
from .command_handler import CommandHandler
from .commands import AetherNavCommands
from .board_utils import Logger, BoardStatus, LogLevel
from .config import DEFAULT_PORT, DEFAULT_BAUDRATE, DEFAULT_TIMEOUT
from .exceptions import SerialConnectionError, BoardError


class HardwareInterface:

    def __init__(
        self,
        port: str = DEFAULT_PORT,
        baudrate: int = DEFAULT_BAUDRATE,
        timeout: float = DEFAULT_TIMEOUT,
        log_level: LogLevel = LogLevel.INFO,
        auto_reconnect: bool = True,
    ):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.auto_reconnect = auto_reconnect

        self.logger = Logger(log_level)
        self.board_status = BoardStatus()

        self._serial_manager: Optional[SerialManager] = None
        self._command_handler: Optional[CommandHandler] = None
        self._commands: Optional[AetherNavCommands] = None

        self._is_connected = False

    def connect(self) -> bool:

        try:
            self._serial_manager = SerialManager(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                logger=self.logger,
            )

            if not self._serial_manager.connect():
                return False

            self._command_handler = CommandHandler(self._serial_manager, self.logger)

            self._commands = AetherNavCommands(self._command_handler, self.board_status)

            if self._commands.test_communication():
                self._is_connected = True
                self.board_status.is_connected = True
                self.logger.info("Successfully connected to AetherNav board")
                return True
            else:
                self.logger.error("Failed to establish communication with board")
                self.disconnect()
                return False

        except Exception as e:
            self.logger.error(f"Connection failed: {e}")
            self.disconnect()
            return False

    def disconnect(self):
        if self._serial_manager:
            self._serial_manager.disconnect()

        self._is_connected = False
        self.board_status.is_connected = False
        self.logger.info("Disconnected from AetherNav board")

    def is_connected(self) -> bool:
        return (
            self._is_connected
            and self._serial_manager
            and self._serial_manager.is_connected()
        )

    def reconnect(self) -> bool:
        self.disconnect()
        return self.connect()

    def _ensure_connected(self):
        if not self.is_connected():
            if self.auto_reconnect:
                if not self.reconnect():
                    raise SerialConnectionError("Failed to reconnect to board")
            else:
                raise SerialConnectionError("Not connected to board")

    # Sensor reading methods
    def get_accelerometer(self) -> Optional[Tuple[float, float, float]]:
        self._ensure_connected()
        return self._commands.get_accelerometer()

    def get_gyroscope(self) -> Optional[Tuple[float, float, float]]:
        self._ensure_connected()
        return self._commands.get_gyroscope()

    def get_angles(self) -> Optional[Tuple[float, float, float]]:
        self._ensure_connected()
        return self._commands.get_angles()

    def get_velocity_x(self) -> Optional[float]:
        self._ensure_connected()
        return self._commands.get_velocity_x()

    def get_distance_x(self) -> Optional[float]:
        self._ensure_connected()
        return self._commands.get_distance_x()

    def get_board_status(self) -> Optional[Dict[str, Any]]:
        self._ensure_connected()
        return self._commands.get_board_status()

    # Motor control methods
    def set_pwm(self, left_pwm: float, right_pwm: float) -> bool:
        self._ensure_connected()
        return self._commands.set_pwm(left_pwm, right_pwm)

    def set_direction(self, left_dir: int, right_dir: int) -> bool:
        self._ensure_connected()
        return self._commands.set_direction(left_dir, right_dir)

    def move_motors(
        self, left_pwm: float, right_pwm: float, left_dir: int = 1, right_dir: int = 1
    ) -> bool:
        self._ensure_connected()
        return self._commands.move_motors(left_pwm, right_pwm, left_dir, right_dir)

    def stop_motors(self) -> bool:
        self._ensure_connected()
        return self._commands.stop_motors()

    def set_max_left_motor_accel(self, acceleration: float) -> bool:
        self._ensure_connected()
        return self._commands.set_max_left_motor_accel(acceleration)

    def set_max_right_motor_accel(self, acceleration: float) -> bool:
        self._ensure_connected()
        return self._commands.set_max_right_motor_accel(acceleration)

    def set_max_left_motor_speed(self, speed: float) -> bool:
        self._ensure_connected()
        return self._commands.set_max_left_motor_speed(speed)

    def set_max_right_motor_speed(self, speed: float) -> bool:
        self._ensure_connected()
        return self._commands.set_max_right_motor_speed(speed)

    # Diagnostic and utility methods
    def get_all_sensor_data(self) -> Dict[str, Any]:
        self._ensure_connected()
        return self._commands.get_all_sensor_data()

    def test_communication(self) -> bool:
        self._ensure_connected()
        return self._commands.test_communication()

    def get_statistics(self) -> Dict[str, Any]:
        return self.board_status.get_stats()

    def configure_retry_settings(self, retry_count: int = 3, retry_delay: float = 0.1):
        if self._command_handler:
            self._command_handler.configure_retry_settings(retry_count, retry_delay)

    def set_log_level(self, level: LogLevel):
        self.logger.level = level

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def __del__(self):
        self.disconnect()
