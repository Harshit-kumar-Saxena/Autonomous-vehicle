import struct
import time
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from .config import FRAME_FORMAT, CommandID, AckCode, LogLevel


class Logger:

    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level

    def _log(self, level: LogLevel, message: str):
        if level.value >= self.level.value:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level.name}] {message}")

    def debug(self, message: str):
        self._log(LogLevel.DEBUG, message)

    def info(self, message: str):
        self._log(LogLevel.INFO, message)

    def warning(self, message: str):
        self._log(LogLevel.WARNING, message)

    def error(self, message: str):
        self._log(LogLevel.ERROR, message)


@dataclass
class CommandResponse:
    command_id: Union[CommandID, int]
    ack: Union[AckCode, int]
    misc_data: int
    data: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    @property
    def is_successful(self) -> bool:
        return isinstance(self.ack, AckCode) and self.ack in [
            AckCode.CMD_OK,
            AckCode.CMD_DONE,
        ]

    @property
    def is_busy(self) -> bool:
        return isinstance(self.ack, AckCode) and self.ack == AckCode.CMD_BUSY

    @property
    def has_error(self) -> bool:
        return isinstance(self.ack, AckCode) and self.ack == AckCode.CMD_SENSOR_ERROR


class FrameBuilder:

    @staticmethod
    def build_frame(
        command_id: Union[CommandID, int],
        subcommand: int = 0,
        misc_data: int = 0,
        data: float = 0.0,
    ) -> bytes:

        cmd_value = (
            command_id.value if isinstance(command_id, CommandID) else command_id
        )
        return struct.pack(FRAME_FORMAT, cmd_value, subcommand, misc_data, data)


class ResponseParser:

    @staticmethod
    def parse_response(frame: bytes) -> Optional[CommandResponse]:

        if len(frame) != 8:
            return None

        try:
            command_id_raw, ack_raw, misc_data, value = struct.unpack(
                FRAME_FORMAT, frame
            )

            command_id = (
                CommandID(command_id_raw)
                if command_id_raw in CommandID._value2member_map_
                else command_id_raw
            )
            ack = AckCode(ack_raw) if ack_raw in AckCode._value2member_map_ else ack_raw

            return CommandResponse(
                command_id=command_id, ack=ack, misc_data=misc_data, data=value
            )

        except struct.error:
            return None


class BoardStatus:

    def __init__(self):
        self.is_connected = False
        self.last_response_time = None
        self.error_count = 0
        self.busy_count = 0
        self.command_count = 0

    def update_from_response(self, response: CommandResponse):
        self.last_response_time = response.timestamp
        self.command_count += 1

        if response.has_error:
            self.error_count += 1
        elif response.is_busy:
            self.busy_count += 1

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_connected": self.is_connected,
            "last_response_time": self.last_response_time,
            "total_commands": self.command_count,
            "error_count": self.error_count,
            "busy_count": self.busy_count,
            "success_rate": (self.command_count - self.error_count - self.busy_count)
            / max(self.command_count, 1)
            * 100,
        }


def validate_command_params(
    command_id: Union[CommandID, int],
    subcommand: int = 0,
    misc_data: int = 0,
    data: float = 0.0,
) -> bool:

    if not 0 <= subcommand <= 255:
        return False

    if not 0 <= misc_data <= 255:
        return False

    if isinstance(command_id, CommandID):
        return True
    elif isinstance(command_id, int) and command_id >= 0:
        return True

    return False
