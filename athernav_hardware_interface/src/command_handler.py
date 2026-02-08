from typing import Optional, Union, Dict, Any
import time
from .serial_manager import SerialManager
from .board_utils import (
    FrameBuilder,
    ResponseParser,
    CommandResponse,
    Logger,
    validate_command_params,
)
from .config import CommandID, AckCode
from .exceptions import (
    InvalidCommandError,
    CommandTimeoutError,
    InvalidResponseError,
    BoardError,
    BoardBusyError,
)


class CommandHandler:

    def __init__(self, serial_manager: SerialManager, logger: Optional[Logger] = None):

        self.serial_manager = serial_manager
        self.logger = logger or Logger()
        self.retry_count = 3
        self.retry_delay = 0.1  # seconds
        self.timeout = 1.0

    def send_command(
        self,
        command_id: Union[CommandID, int],
        subcommand: int = 0,
        misc_data: int = 0,
        data: float = 0.0,
        timeout: Optional[float] = None,
        wait_for_core_1: bool = False,
        retry_on_busy: bool = True,
    ) -> Optional[CommandResponse]:

        # Validate parameters
        if not validate_command_params(command_id, subcommand, misc_data, data):
            raise InvalidCommandError(
                "Invalid command parameters",
                command_id.value if isinstance(command_id, CommandID) else command_id,
            )

        # Build frame
        frame = FrameBuilder.build_frame(command_id, subcommand, misc_data, data)

        cmd_name = (
            command_id.name
            if isinstance(command_id, CommandID)
            else f"CMD_{command_id}"
        )
        self.logger.debug(f"Sending command: {cmd_name}")

        # Send command with retries
        for attempt in range(self.retry_count):
            try:
                response_frame = self.serial_manager.send_and_receive(
                    frame, timeout or self.timeout
                )

                if response_frame is None:
                    if attempt < self.retry_count - 1:
                        self.logger.warning(
                            f"Command timeout, retrying... (attempt {attempt + 1}/{self.retry_count})"
                        )
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        raise CommandTimeoutError(
                            f"Command {cmd_name} timed out after {self.retry_count} attempts"
                        )

                # Parse response
                response = ResponseParser.parse_response(response_frame)
                if response is None:
                    raise InvalidResponseError(
                        "Failed to parse response", response_frame
                    )

                # Handle special cases
                if (
                    response.is_busy
                    and retry_on_busy
                    and attempt < self.retry_count - 1
                ):
                    self.logger.warning(
                        f"Board busy, retrying... (attempt {attempt + 1}/{self.retry_count})"
                    )
                    time.sleep(self.retry_delay)
                    continue

                if response.is_busy:
                    raise BoardBusyError(f"Board is busy for command {cmd_name}")

                if response.has_error:
                    ack_name = (
                        response.ack.name
                        if isinstance(response.ack, AckCode)
                        else f"ACK_{response.ack}"
                    )
                    raise BoardError(
                        f"Board error: {ack_name}",
                        (
                            response.ack.value
                            if isinstance(response.ack, AckCode)
                            else response.ack
                        ),
                    )

                if wait_for_core_1 and response.ack == AckCode.CMD_OK:
                    second_response_frame = self.serial_manager.read_frame(
                        timeout or self.timeout
                    )
                    if second_response_frame:
                        second_response = ResponseParser.parse_response(
                            second_response_frame
                        )
                        if second_response:
                            # Update response with second frame data
                            response.ack = second_response.ack
                            response.data = second_response.data
                            response.misc_data = second_response.misc_data

                self.logger.debug(f"Command {cmd_name} successful")
                return response

            except (InvalidResponseError, BoardError, BoardBusyError) as e:
                # Don't retry on these errors
                raise
            except Exception as e:
                if attempt < self.retry_count - 1:
                    self.logger.warning(
                        f"Command failed, retrying... (attempt {attempt + 1}/{self.retry_count}): {e}"
                    )
                    time.sleep(self.retry_delay)
                    continue
                else:
                    raise

        return None

    def send_simple_command(
        self,
        command_id: Union[CommandID, int],
        data: float = 0.0,
        timeout: Optional[float] = None,
    ) -> Optional[float]:

        response = self.send_command(command_id, data=data, timeout=timeout)
        return response.data if response else None

    def get_sensor_value(
        self, sensor_command: Union[CommandID, int], timeout: Optional[float] = None
    ) -> Optional[float]:

        return self.send_simple_command(sensor_command, timeout=timeout)

    def set_parameter(
        self,
        param_command: Union[CommandID, int],
        value: float,
        subcommand: int = 0,
        timeout: Optional[float] = None,
    ) -> bool:

        response = self.send_command(
            param_command, subcommand=subcommand, data=value, timeout=timeout
        )
        return response.is_successful if response else False

    def configure_retry_settings(self, retry_count: int = 3, retry_delay: float = 0.1):

        self.retry_count = max(1, retry_count)
        self.retry_delay = max(0.01, retry_delay)

    def set_default_timeout(self, timeout: float):
        self.timeout = max(0.1, timeout)
