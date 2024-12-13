from ultralytics import YOLO
import cv2
import serial
import time

serial_port = '/dev/ttyACM0'  # Change this to the appropriate port
baud_rate = 9600
ser = serial.Serial(serial_port, baud_rate)
time.sleep(2)

model = YOLO('last_single_class_new.pt')
video_path = "vid1.mp4"


cap = cv2.VideoCapture(video_path)
cap.set(5, 1280)
cap.set(4, 1024)

cv2.namedWindow('YOLO V8 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO V8 Detection', 1280, 1024)

height_offset = 100
left_offset = -60
right_offset = 60

command = 'S'
val = ['S']

# PID constants
Kp = 0.1
Ki = 0.01
Kd = 0.01

prev_error = 0
integral = 0

while True:
    _, img = cap.read()

    results = model(img)

    if _:

        try:

            img_height = img.shape[0]
            img_width = img.shape[1]

            for r in results:

                if r.boxes:

                    boxes = r.boxes[0].cpu().numpy()
                    for box in boxes:
                        b = box.xyxy[0]

                        diagonal_center = [(int(b[0]) + int(b[2])) // 2, (int(b[1]) + int(b[3])) // 2]

                        error = diagonal_center[0] - (img_width // 2)
                        integral += error
                        derivative = error - prev_error

                        pid_output = Kp * error + Ki * integral + Kd * derivative

                        prev_error = error

                        if -20 <= pid_output <= 20:
                            command = 'F'
                        elif pid_output < -20:
                            command = 'L'
                        else:
                            command = 'R'

                else:
                    command = 'S'

            ser.write(command.encode())

            cv2.imshow('YOLO V8 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                command = 'S'
                ser.write(command.encode())
                break

        except Exception as e:
            print(e)

        if val[0] != command:

            ser.write(command.encode())
            val.pop()
            val.append(command)

cap.release()
command = 'S'
ser.write(command.encode())
cv2.destroyAllWindows()
