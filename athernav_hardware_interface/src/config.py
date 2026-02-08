from enum import Enum

FRAME_FORMAT = (
    ">HBBf"  # uint16 (command ID), uint8 (subcommand), uint8 (misc), float (4 bytes)
)
FRAME_SIZE = 8

DEFAULT_PORT = "/dev/ttyACM0"
DEFAULT_BAUDRATE = 500000
DEFAULT_TIMEOUT = 1.0


class CommandID(Enum):
    IDLE = 0
    GET_BOARD_STATUS = 500
    GET_AX = 501
    GET_AY = 502
    GET_AZ = 503
    GET_GX = 504
    GET_GY = 505
    GET_GZ = 506
    GET_ANGLE_X = 506
    GET_ANGLE_Y = 507
    GET_ANGLE_Z = 508
    GET_VEL_X = 509
    GET_DIST_X = 510
    SET_PWM = 511
    SET_DIR = 512
    SET_MAX_LEFT_MOTOR_ACCEL = 513
    SET_MAX_RIGHT_MOTOR_ACCEL = 514
    SET_MAX_LEFT_MOTOR_SPEED = 515
    SET_MAX_RIGHT_MOTOR_SPEED = 516


class AckCode(Enum):
    CMD_OK = 1
    CMD_DONE = 2
    CMD_BUSY = 3
    CMD_SENSOR_ERROR = 4


class LogLevel(Enum):
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
