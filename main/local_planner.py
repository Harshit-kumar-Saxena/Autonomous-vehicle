from ultralytics import YOLO
import cv2
import time
import numpy as np
from threading import Thread, Lock
import os

class LocalPlanner:
    def __init__(self, model_path='last.pt', camera_index='/dev/video2'):
        # Camera initialization with specific device path
        self.cap = self.initialize_camera(camera_index)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"Critical: Could not open camera at {camera_index}")
        
        # Model initialization in separate thread
        self.model = None
        self.model_ready = False
        self.model_lock = Lock()
        Thread(target=self.load_model, args=(model_path,), daemon=True).start()
        
        self.left_offset = -30
        self.right_offset = 30
        self.last_frame = None
        self.latest_raw_frame = None
        self.frame_lock = Lock()
        self.latest_direction = "obstacle_stop"
        self.direction_lock = Lock()
        
        # Start camera capture thread
        self.running = True
        Thread(target=self.capture_thread, daemon=True).start()
        Thread(target=self.processing_thread, daemon=True).start()
        
        print("⏳ Warming up camera and model...")

    def initialize_camera(self, camera_index):
        # Try different backends for /dev/video2
        for backend in [cv2.CAP_V4L2, cv2.CAP_ANY]:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 15)
                print(f" Camera opened at {camera_index} with backend {backend}")
                return cap
            else:
                print(f"Failed with backend {backend}")
        return None

    def load_model(self, model_path):
        try:
            print("Loading YOLO model...")
            model = YOLO(model_path)
            
            # Warm up model with dummy data
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            model(dummy)
            
            with self.model_lock:
                self.model = model
                self.model_ready = True
            print("Model loaded and warmed up")
        except Exception as e:
            print(f"Model loading failed: {e}")

    def capture_thread(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.latest_raw_frame = frame
            time.sleep(0.01)

    def processing_thread(self):
        while self.running:
            if not self.model_ready:
                time.sleep(0.1)
                continue
                
            # Get latest frame
            with self.frame_lock:
                if self.latest_raw_frame is None:
                    time.sleep(0.01)
                    continue
                frame = self.latest_raw_frame.copy()
            
            # Process frame
            try:
                with self.model_lock:
                    results = self.model(frame)
                
                # Visualize results
                img = results[0].plot()
                self.last_frame = img
                
                # Process detections
                img_width = img.shape[1]
                boxes = results[0].boxes.xyxy.cpu().numpy()
                if len(boxes) > 0:
                    box = boxes[0]
                    center_x = (box[0] + box[2]) / 2
                    
                    if img_width//2 + self.left_offset <= center_x <= img_width//2 + self.right_offset:
                        direction = "obstacle_forward"
                    elif center_x < img_width//2 + self.left_offset:
                        direction = "obstacle_left"
                    else:
                        direction = "obstacle_right"
                else:
                    direction = "obstacle_stop"
                
                with self.direction_lock:
                    self.latest_direction = direction
                    
                # Print detection info
                if len(boxes) > 0:
                    print(f"Detected {len(boxes)} objects | Direction: {direction}")
                else:
                    print("🔍 No objects detected")
                    
            except Exception as e:
                print(f" Processing error: {e}")
                with self.direction_lock:
                    self.latest_direction = "obstacle_stop"
            
            time.sleep(0.1)  # Process at ~10 FPS

    def get_direction(self):
        with self.direction_lock:
            return self.latest_direction

    def get_frame(self):
        return self.last_frame

    def release(self):
        self.running = False
        time.sleep(0.2)  # Allow threads to exit
        if self.cap and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()