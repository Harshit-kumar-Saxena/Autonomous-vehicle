class AetherNavException(Exception):
    pass


class SerialConnectionError(AetherNavException):
    def __init__(self, message: str, port: str = None):
        super().__init__(message)
        self.port = port


class CommandTimeoutError(AetherNavException):
    def __init__(self, message: str, command_id: int = None):
        super().__init__(message)
        self.command_id = command_id


class InvalidResponseError(AetherNavException):
    def __init__(self, message: str, raw_data: bytes = None):
        super().__init__(message)
        self.raw_data = raw_data


class BoardError(AetherNavException):
    def __init__(self, message: str, ack_code: int = None):
        super().__init__(message)
        self.ack_code = ack_code


class InvalidCommandError(AetherNavException):
    def __init__(self, message: str, command_id: int = None):
        super().__init__(message)
        self.command_id = command_id


class BoardBusyError(AetherNavException):
    pass
