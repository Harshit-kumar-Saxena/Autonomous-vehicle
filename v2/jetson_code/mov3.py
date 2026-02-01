import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import serial
import struct
import signal
import sys
import os
import select
import termios
import tty

# ================= CONFIGURATION =================
# --- Hardware ---
PORT = "/dev/ttyTHS0"      
BAUD = 115200

# --- Models ---
ENGINE_PATH = "/home/wolf/stereo/fastscnn_road.engine"

# --- Camera ---
CAMERA_ID = 2  
FRAME_W = 640
FRAME_H = 480

# --- Model Input ---
INPUT_W = 512
INPUT_H = 256

# --- Motor Logic ---
CMD_SET_PWM      = 511  
CMD_SET_BOTH_PWM = 520  

# --- SPEED SETTINGS (Slower) ---
CRUISE_SPEED     = 100   
TURN_STRENGTH    = 40    

# ================= TERMINAL INPUT HANDLER =================
orig_settings = termios.tcgetattr(sys.stdin)

def getKey():
    """Reads a single key from terminal without pressing Enter"""
    try:
        tty.setcbreak(sys.stdin.fileno()) 
        if select.select([sys.stdin], [], [], 0)[0]: 
            key = sys.stdin.read(1)
            return key
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

# ================= UART HANDLER =================
class JetsonUART:
    def __init__(self):
        self.ser = None

    def connect(self):
        try:
            self.ser = serial.Serial(PORT, BAUD, timeout=0, write_timeout=0)
            time.sleep(2) 
            print(f"[INFO] UART Connected on {PORT}")
            return True
        except Exception as e:
            print(f"[ERROR] UART Open Failed: {e}")
            return False

    def send_sync_forward(self, speed):
        if not self.ser: return
        spd = max(0, min(255, int(speed)))
        packet = struct.pack('>HBBf', CMD_SET_BOTH_PWM, spd, spd, 0.0)
        try:
            self.ser.write(packet)
        except:
            pass

    def send_individual(self, motor_id, speed):
        if not self.ser: return
        spd = max(0, min(255, int(speed)))
        packet = struct.pack('>HBBf', CMD_SET_PWM, int(motor_id), 0, float(spd))
        try:
            self.ser.write(packet)
        except:
            pass

    def stop(self):
        if not self.ser: return
        # Send STOP command 520 with 0 speed
        packet1 = struct.pack('>HBBf', CMD_SET_PWM, 1, 0, 0.0)
        packet2 = struct.pack('>HBBf', CMD_SET_PWM, 2, 0, 0.0)
        try:
            # Send 3 times to ensure ESP32 receives it
            for _ in range(3): 
                self.ser.write(packet1)
                self.ser.write(packet2)
                time.sleep(0.01)
        except:
            pass

    def close(self):
        if self.ser:
            self.stop()
            self.ser.close()

bot = JetsonUART()

def signal_handler(sig, frame):
    print("\n[INFO] Ctrl+C Detected. Stopping Motors...")
    bot.stop()
    bot.close()
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings) 
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ================= TENSORRT WRAPPER =================
class TrtModel:
    def __init__(self, engine_path):
        print(f"[INFO] Loading Engine: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        if not os.path.exists(engine_path):
            print(f"[ERROR] Engine file not found: {engine_path}")
            sys.exit(1)
            
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(dev_mem))
            if self.engine.binding_is_input(i):
                self.inputs.append({'host': host_mem, 'dev': dev_mem})
            else:
                self.outputs.append({'host': host_mem, 'dev': dev_mem})

    def infer(self, img):
        resized = cv2.resize(img, (INPUT_W, INPUT_H))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = resized.transpose(2, 0, 1).ravel()
        
        np.copyto(self.inputs[0]['host'], blob)
        cuda.memcpy_htod_async(self.inputs[0]['dev'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['dev'], self.stream)
        self.stream.synchronize()
        return self.outputs[0]['host']

# ================= MAIN LOOP =================
def main():
    if not bot.connect():
        return

    model = TrtModel(ENGINE_PATH)
    cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    if not cap.isOpened():
        print(f"[ERROR] Camera {CAMERA_ID} not found")
        return

    print("========================================")
    print("   AUTONOMOUS BOT RUNNING (SLOW MODE)")
    print("   [s] -> STOP MOTORS & EXIT CODE")
    print("========================================")
    
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # 1. CHECK TERMINAL INPUT
            key = getKey()
            
            # *** NEW LOGIC FOR 's' ***
            if key == 's' or key == 'q':
                print("\n\n[USER STOP] Stopping Motors & Exiting...")
                bot.stop()      # Send 0 velocity
                time.sleep(0.1) # Wait briefly to ensure command is sent
                break           # Break loop -> triggers 'finally' cleanup

            # 2. CAMERA & INFERENCE
            ret, frame = cap.read()
            if not ret: continue

            output = model.infer(frame)
            mask = output.reshape(INPUT_H, INPUT_W)
            
            start_row = int(INPUT_H * 0.7)
            roi = mask[start_row:, :] 
            y_idxs, x_idxs = np.where(roi > 0.4)
            
            status = "WAIT"
            
            if len(x_idxs) > 100:
                center_x_px = np.mean(x_idxs)
                center_x = center_x_px / INPUT_W 
                
                # Logic: Slower Speed & Gentler Turns
                if 0.42 < center_x < 0.58:
                    bot.send_sync_forward(CRUISE_SPEED)
                    status = "FWD"
                else:
                    left_pwm = CRUISE_SPEED
                    right_pwm = CRUISE_SPEED
                    
                    if center_x < 0.42: 
                        left_pwm  -= TURN_STRENGTH
                        right_pwm += TURN_STRENGTH
                        status = "LEFT"
                    else: 
                        left_pwm  += TURN_STRENGTH
                        right_pwm -= TURN_STRENGTH
                        status = "RIGHT"
                    
                    bot.send_individual(1, left_pwm)
                    bot.send_individual(2, right_pwm)
            else:
                bot.stop()
                status = "LOST"

            # 3. STATS
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.time()
                fps = 10.0 / (now - start_time)
                sys.stdout.write(f"\r[FPS: {fps:.1f}] [STATUS: {status}] [CENTER: {center_x if 'center_x' in locals() else 0:.2f}]   ")
                sys.stdout.flush()
                start_time = now
            
            time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        # This runs when 'break' is called or Ctrl+C is pressed
        bot.close() # Sends stop again just to be safe
        cap.release()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, orig_settings)

if __name__ == "__main__":
    main()
