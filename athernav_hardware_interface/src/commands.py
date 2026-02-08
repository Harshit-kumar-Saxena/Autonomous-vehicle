from typing import Optional, Dict, Any, Tuple
from .command_handler import CommandHandler
from .board_utils import BoardStatus
from .config import CommandID, AckCode
from .exceptions import BoardError


class AetherNavCommands:

    def __init__(self, command_handler: CommandHandler, board_status: BoardStatus):

        self.cmd = command_handler
        self.status = board_status

    def get_accelerometer(self) -> Optional[Tuple[float, float, float]]:
        ax = self.cmd.get_sensor_value(CommandID.GET_AX)
        ay = self.cmd.get_sensor_value(CommandID.GET_AY)
        az = self.cmd.get_sensor_value(CommandID.GET_AZ)

        if ax is not None and ay is not None and az is not None:
            return (ax, ay, az)
        return None

    def get_gyroscope(self) -> Optional[Tuple[float, float, float]]:
        gx = self.cmd.get_sensor_value(CommandID.GET_GX)
        gy = self.cmd.get_sensor_value(CommandID.GET_GY)
        gz = self.cmd.get_sensor_value(CommandID.GET_GZ)

        if gx is not None and gy is not None and gz is not None:
            return (gx, gy, gz)
        return None

    def get_angles(self) -> Optional[Tuple[float, float, float]]:
        angle_x = self.cmd.get_sensor_value(CommandID.GET_ANGLE_X)
        angle_y = self.cmd.get_sensor_value(CommandID.GET_ANGLE_Y)
        angle_z = self.cmd.get_sensor_value(CommandID.GET_ANGLE_Z)

        if angle_x is not None and angle_y is not None and angle_z is not None:
            return (angle_x, angle_y, angle_z)
        return None

    def get_velocity_x(self) -> Optional[float]:
        return self.cmd.get_sensor_value(CommandID.GET_VEL_X)

    def get_distance_x(self) -> Optional[float]:
        return self.cmd.get_sensor_value(CommandID.GET_DIST_X)

    def get_board_status(self) -> Optional[Dict[str, Any]]:
        response = self.cmd.send_command(CommandID.GET_BOARD_STATUS)
        if response and response.is_successful:
            return {
                "status": "OK",
                "misc_data": response.misc_data,
                "data": response.data,
                "timestamp": response.timestamp,
            }
        return None

    def set_pwm(self, left_pwm: float, right_pwm: float) -> bool:

        return self.cmd.set_parameter(
            CommandID.SET_PWM, left_pwm, subcommand=int(right_pwm * 255)
        )

    def set_direction(self, left_dir: int, right_dir: int) -> bool:

        subcommand = (left_dir & 0x01) | ((right_dir & 0x01) << 1)
        return self.cmd.set_parameter(CommandID.SET_DIR, 0.0, subcommand=subcommand)

    def set_max_left_motor_accel(self, acceleration: float) -> bool:
        return self.cmd.set_parameter(CommandID.SET_MAX_LEFT_MOTOR_ACCEL, acceleration)

    def set_max_right_motor_accel(self, acceleration: float) -> bool:
        return self.cmd.set_parameter(CommandID.SET_MAX_RIGHT_MOTOR_ACCEL, acceleration)

    def set_max_left_motor_speed(self, speed: float) -> bool:
        return self.cmd.set_parameter(CommandID.SET_MAX_LEFT_MOTOR_SPEED, speed)

    def set_max_right_motor_speed(self, speed: float) -> bool:
        return self.cmd.set_parameter(CommandID.SET_MAX_RIGHT_MOTOR_SPEED, speed)

    def move_motors(
        self, left_pwm: float, right_pwm: float, left_dir: int = 1, right_dir: int = 1
    ) -> bool:

        if not self.set_direction(left_dir, right_dir):
            return False

        return self.set_pwm(left_pwm, right_pwm)

    def stop_motors(self) -> bool:
        return self.move_motors(0.0, 0.0)

    def get_all_sensor_data(self) -> Dict[str, Any]:
        data = {}

        accel = self.get_accelerometer()
        if accel:
            data["accelerometer"] = {"x": accel[0], "y": accel[1], "z": accel[2]}

        gyro = self.get_gyroscope()
        if gyro:
            data["gyroscope"] = {"x": gyro[0], "y": gyro[1], "z": gyro[2]}

        angles = self.get_angles()
        if angles:
            data["angles"] = {"x": angles[0], "y": angles[1], "z": angles[2]}

        vel_x = self.get_velocity_x()
        if vel_x is not None:
            data["velocity_x"] = vel_x

        dist_x = self.get_distance_x()
        if dist_x is not None:
            data["distance_x"] = dist_x

        board_status = self.get_board_status()
        if board_status:
            data["board_status"] = board_status

        data["statistics"] = self.status.get_stats()

        return data

    def test_communication(self) -> bool:
        try:
            status = self.get_board_status()
            return status is not None
        except Exception:
            return False
