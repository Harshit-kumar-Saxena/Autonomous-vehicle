from src import HardwareInterface, LogLevel
import time

def main():
    
    board = HardwareInterface(
        port="/dev/ttyACM0",
        baudrate=500000,
        log_level=LogLevel.INFO,
        auto_reconnect=True
    )
    
    try:
        print("Connecting to AetherNav board...")
        if not board.connect():
            print("Failed to connect to board")
            return
        
        print("Connected successfully!")
        
        print("\nTesting communication...")
        if board.test_communication():
            print("Communication test passed!")
        else:
            print("Communication test failed!")
            return
        
        print("\nReading sensor data...")
        
        accel = board.get_accelerometer()
        if accel:
            print(f"Accelerometer - X: {accel[0]:.3f}, Y: {accel[1]:.3f}, Z: {accel[2]:.3f}")
        
        gyro = board.get_gyroscope()
        if gyro:
            print(f"Gyroscope - X: {gyro[0]:.3f}, Y: {gyro[1]:.3f}, Z: {gyro[2]:.3f}")
        
        angles = board.get_angles()
        if angles:
            print(f"Angles - X: {angles[0]:.3f}, Y: {angles[1]:.3f}, Z: {angles[2]:.3f}")
        
        vel_x = board.get_velocity_x()
        if vel_x is not None:
            print(f"Velocity X: {vel_x:.3f}")
        
        dist_x = board.get_distance_x()
        if dist_x is not None:
            print(f"Distance X: {dist_x:.3f}")
        
        status = board.get_board_status()
        if status:
            print(f"Board Status: {status}")
        
        print("\nTesting motor control...")
        
        print("Moving motors forward at 50% PWM...")
        if board.move_motors(0.5, 0.5, left_dir=1, right_dir=1):
            print("Motor command sent successfully!")
            time.sleep(1)  # Run for 1 second
        
        print("Stopping motors...")
        if board.stop_motors():
            print("Stop command sent successfully!")
        
        print("\nGetting all sensor data...")
        all_data = board.get_all_sensor_data()
        print("Complete sensor data:")
        for key, value in all_data.items():
            print(f"  {key}: {value}")
        
        stats = board.get_statistics()
        print(f"\nBoard Statistics: {stats}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
    
    finally:
        print("\nDisconnecting from board...")
        board.disconnect()
        print("Disconnected successfully!")


def example_with_context_manager():
    
    print("\n" + "="*50)
    print("Example using context manager:")
    print("="*50)
    
    try:
        with HardwareInterface(port="/dev/ttyACM0", log_level=LogLevel.DEBUG) as board:
            
            accel = board.get_accelerometer()
            if accel:
                print(f"Accelerometer: {accel}")
            
            board.move_motors(0.3, 0.3)  # 30% PWM
            time.sleep(0.5)
            board.stop_motors()
            
            
    except Exception as e:
        print(f"Context manager example error: {e}")


def example_custom_retry_settings():
    
    print("\n" + "="*50)
    print("Example with custom retry settings:")
    print("="*50)
    
    board = HardwareInterface(port="/dev/ttyACM0")
    
    try:
        if board.connect():
            board.configure_retry_settings(retry_count=5, retry_delay=0.2)
            
            accel = board.get_accelerometer()
            if accel:
                print(f"Accelerometer with custom retry: {accel}")
                
    except Exception as e:
        print(f"Custom retry example error: {e}")
    
    finally:
        board.disconnect()

if __name__ == "__main__":
    main()
    
    example_with_context_manager()
    example_custom_retry_settings()
