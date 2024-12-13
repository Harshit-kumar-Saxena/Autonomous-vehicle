from ultralytics import YOLO
import cv2
import time

# serial_port = '/dev/ttyACM0'  # Change this to the appropriate port
# baud_rate = 9600
# ser = serial.Serial(serial_port, baud_rate)
time.sleep(2)

#model = YOLO('last_single_class_new.pt')
model = YOLO('last.pt')
video_path = "vid1.mp4"


cap = cv2.VideoCapture(0)
cap.set(5, 1280)
cap.set(4, 1024)

cv2.namedWindow('YOLO V8 Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('YOLO V8 Detection', 1280, 1024)

height_offset = 100
left_offset = -30
right_offset = 30

command = 'S'
val = ['S']

while True:
    _, img = cap.read()

    # results = model.predict(img)
    results = model(img)

    # results1 = model(img)
    img = results[0].plot()
    # annotator = Annotator(img)

    if _:

        try:

            img_height = img.shape[0]
            img_width = img.shape[1]

            for r in results:

                if r.boxes:

                    boxes = r.boxes[0].cpu().numpy()
                    for box in boxes:
                        b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format

                        c = box.cls

                        # cv2 detection box
                        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 5)
                        diagonal_center = [(int(b[0]) + int(b[2])) // 2, (int(b[1]) + int(b[3])) // 2]
                        cv2.circle(img, (diagonal_center[0], img_height // 2 + height_offset), 10, (0, 255, 0), 10)
                        cv2.line(img, (diagonal_center[0], img_height // 2 + height_offset),
                                 (img_width // 2, img_height // 2 + height_offset), (255, 0, 0), 5)

                        # cv2 window
                        cv2.line(img, (img_width // 2, img_height), (img_width // 2, 0), (0, 0, 255), 3)
                        cv2.circle(img, (img_width // 2, img_height // 2 + height_offset), 10, (0, 0, 255), 10)
                        cv2.line(img, (img_width // 2 + left_offset, img_height), (img_width // 2 + left_offset, 0),
                                 (255, 0, 0), 5)

                        cv2.line(img, (img_width // 2 + right_offset, img_height), (img_width // 2 + right_offset, 0),
                                 (255, 0, 0), 5)

                        if img_width // 2 + left_offset <= diagonal_center[0] <= img_width // 2 + right_offset:
                            print("FORWARD")

                            cv2.putText(img, 'FORWARD', (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                            command = 'F'

                        if diagonal_center[0] < img_width // 2 + left_offset:
                            print("LEFT")
                            cv2.putText(img, 'LEFT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                        cv2.LINE_AA)

                            command = 'L'

                        if diagonal_center[0] > img_width // 2 + right_offset:
                            print("RIGHT")
                            cv2.putText(img, 'RIGHT', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                                        cv2.LINE_AA)

                            command = 'R'

                        # annotator.box_label(b, model.names[int(c)])
                else:
                    result = 'S'
                    cv2.putText(img, 'STOP', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # ser.write(command.encode())

            # img = annotator.result()
            cv2.imshow('YOLO V8 Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                command = 'S'
                # ser.write(command.encode())
                break

        except Exception as e:
            print(e)

        if val[0] != command:

            # ser.write(command.encode())
            val.pop()
            val.append(command)

cap.release()
# command = 'S'
# ser.write(command.encode())
cv2.destroyAllWindows()
