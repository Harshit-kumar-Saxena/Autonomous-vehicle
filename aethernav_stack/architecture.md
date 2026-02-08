# AetherNav Stack Architecture

Detailed technical documentation of the system architecture, data flow, and component interactions.

## System Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            EXECUTOR (run.py)                               │
│   Orchestrates startup, shutdown, and threads                              │
└─────────────────────────────────┬──────────────────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   CameraThread   │   │ PerceptionThread │   │  ControlThread   │
│   (30 Hz)        │──►│   (async)        │──►│   (30 Hz)        │
│                  │   │                  │   │                  │
│ FrameData        │   │ SegmentationRes  │   │ WheelVelocities  │
└──────────────────┘   └──────────────────┘   └──────────────────┘
          │                       │                       │
          ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│  CameraManager   │   │SegmentationEngine│   │  LaneFollower +  │
│                  │   │                  │   │  PIDController   │
└──────────────────┘   └──────────────────┘   └──────────────────┘
                                                        │
                                                        ▼
                                              ┌──────────────────┐
                                              │ HardwareInterface│
                                              │ (Serial/Mock)    │
                                              └──────────────────┘
```

---

## Entry Points

### run.py
```python
# Entry point that selects executor mode
if args.single_threaded:
    executor = Executor(...)      # executor.py
else:
    executor = ThreadedExecutor(...)  # executor_threaded.py
executor.start()
executor.run()
```

### executor_threaded.py (Recommended)

Multi-threaded pipeline with decoupled stages:

```python
class ThreadedExecutor:
    def start(self):
        # 1. Initialize segmentation
        self.segmentation.initialize()
        
        # 2. Open camera
        self.camera.open()
        
        # 3. Connect hardware
        self.hardware.connect()
        
        # 4. Create pipeline threads
        self._camera_thread = CameraThread(...)
        self._perception_thread = PerceptionThread(...)
        self._control_thread = ControlThread(...)
        
        # 5. Start all threads
        for thread in [camera, perception, control]:
            thread.start()
```

---

## Component Details

### 1. Configuration Layer

```
config/
├── config_loader.py      # Parses YAML → Dataclasses
├── stack_config.yaml     # Hardware, camera, segmentation, planner
└── robot_config.yaml     # Robot physical params, PID gains
```

**Data Flow:**
```
stack_config.yaml ─┐
                   ├──► ConfigLoader.load_*() ──► StackConfig, RobotConfig
robot_config.yaml ─┘                                      │
                                                          ▼
                                              Components receive typed config
```

**Key Dataclasses:**
| Class | Purpose |
|-------|---------|
| `StackConfig` | Top-level: hardware, camera, segmentation, planner, executor |
| `RobotConfig` | Robot params + PID gains |
| `SegmentationConfig` | Model paths, thresholds, centerline extraction |
| `LaneFollowerConfig` | Cruise speed, angular gain, deadband |

---

### 2. Perception Layer

```
perception/
├── camera_manager.py         # Capture from camera or video file
├── segmentation/
│   ├── engine.py             # Main SegmentationEngine
│   ├── base.py               # Abstract InferenceBackend
│   ├── onnx_backend.py       # ONNX Runtime (CPU/GPU)
│   ├── tensorrt_backend.py   # TensorRT (Jetson)
│   └── mock_backend.py       # Returns dummy mask
├── mask_filtering.py         # Morphological cleaning
├── centerline_extraction.py  # RANSAC polynomial fitting
└── temporal_smoothing.py     # EMA frame-to-frame smoothing
```

**Segmentation Pipeline:**
```
Frame (HxWx3) ────────────────────────────────────────────────────────►
     │
     ▼
┌────────────────┐    ┌────────────────┐    ┌────────────────┐
│  _preprocess() │───►│ Backend.infer()│───►│ _postprocess() │
│  Resize/Norm   │    │  ONNX/TRT/Mock │    │ Mask→Centerline│
└────────────────┘    └────────────────┘    └────────────────┘
                                                    │
                                                    ▼
                                            SegmentationResult
                                            ├── mask (256x512)
                                            ├── centerline_points [(x,y),...]
                                            ├── lane_detected (bool)
                                            └── confidence (float)
```

**Backend Selection:**
```python
# engine.py
def initialize(self):
    if config.inference_backend == "tensorrt":
        self._backend = TensorRTBackend(config.engine_path)
    elif config.inference_backend == "onnx":
        self._backend = ONNXBackend(config.onnx_path)
    else:
        self._backend = MockBackend()  # Fallback
```

---

### 3. Planning Layer

```
planner/
└── lane_follower.py    # LaneFollowerPlanner
```

**Lane Follower Logic:**
```python
def compute_velocity(seg_result, robot_state) -> Twist2D:
    if not seg_result.lane_detected:
        return recovery_behavior()
    
    # Error = deviation from center (0.5 = centered)
    error = seg_result.lane_center_normalized - 0.5
    
    # Deadband: no correction if nearly centered
    if deadband_min < lane_center < deadband_max:
        angular = 0.0
    else:
        angular = -angular_gain * error
    
    # Slow down on curves
    if slow_down_on_curve and abs(angular) > threshold:
        linear = min_linear_vel
    else:
        linear = cruise_linear_vel
    
    return Twist2D(linear, angular)
```

---

### 4. Control Layer

```
controls/
├── pid_controller.py   # PIDController
└── odometry.py         # Odometry estimation
```

**PID Controller:**
```python
def compute_wheel_velocities(cmd_vel: Twist2D, state, dt) -> WheelVelocities:
    # 1. Inverse kinematics: Twist → target wheel speeds
    target = kinematics.twist_to_wheels(cmd_vel)
    
    # 2. PID tracking (optional)
    if use_feedback:
        error = target - current_measured
        output = pid.update(error, dt)
    else:
        output = target  # Feedforward only
    
    # 3. Apply limits
    output = clamp(output, max_wheel_vel)
    
    return WheelVelocities(left, right)
```

**Odometry:**
```python
def update(wheel_velocities, imu_yaw=None) -> RobotState:
    # Forward kinematics
    v, omega = kinematics.forward(wheel_velocities)
    
    # Integrate pose
    pose.x += v * cos(pose.theta) * dt
    pose.y += v * sin(pose.theta) * dt
    pose.theta += omega * dt
    
    return RobotState(pose, velocity)
```

---

### 5. Robot Model

```
robot_model/
├── params.py       # RobotParams dataclass
└── diff_drive.py   # DiffDriveKinematics
```

**Differential Drive Kinematics:**
```python
class DiffDriveKinematics:
    # Inverse: Twist → Wheel Velocities
    def twist_to_wheels(twist: Twist2D) -> WheelVelocities:
        v_left = (twist.linear - twist.angular * track_width/2) / wheel_radius
        v_right = (twist.linear + twist.angular * track_width/2) / wheel_radius
        return WheelVelocities(v_left, v_right)
    
    # Forward: Wheel Velocities → Twist
    def forward(wheels: WheelVelocities) -> Twist2D:
        v = (wheels.left + wheels.right) * wheel_radius / 2
        omega = (wheels.right - wheels.left) * wheel_radius / track_width
        return Twist2D(v, omega)
```

---

### 6. Hardware Interface

```
hardware/
└── hw_interface.py    # AetherNavHardwareInterface
```

**Mock vs Real Hardware:**
```python
class AetherNavHardwareInterface:
    def connect(self):
        if config.mock_hardware:
            logger.info("MOCK HARDWARE mode")
            return  # No serial connection
        else:
            self._serial = Serial(config.port, config.baudrate)
    
    def send_wheel_velocities(self, wheel_vel):
        if self._mock_mode:
            return  # No-op, just log
        
        # Real: encode and send via serial
        packet = self._encode_velocity(wheel_vel)
        self._serial.write(packet)
```

---

### 7. Pipeline Threads (core/pipeline.py)

**Thread Communication:**
```
CameraThread ──► LatestHolder<FrameData> ──► PerceptionThread
                                                     │
                                                     ▼
ControlThread ◄── LatestHolder<PerceptionData> ◄────┘
```

**LatestHolder Pattern:**
```python
class LatestHolder(Generic[T]):
    """Thread-safe container holding the most recent value."""
    def put(self, value: T):
        with self._lock:
            self._value = value
    
    def get(self, timeout=None) -> T:
        # Returns latest value, discards stale frames
```

---

## Data Types (core/types.py)

| Type | Fields | Description |
|------|--------|-------------|
| `Pose2D` | `x, y, theta` | 2D position and heading |
| `Twist2D` | `linear, angular` | Velocity command |
| `WheelVelocities` | `left, right` | Wheel angular velocities (rad/s) |
| `FrameData` | `image, timestamp` | Camera frame |
| `SegmentationResult` | `mask, centerline_points, lane_detected, confidence` | Perception output |
| `RobotState` | `pose, velocity, wheel_velocities` | Full robot state |

---

## Call Graph: One Control Cycle

```
executor.run()
    └── _control_cycle()
            │
            ├── camera.read() ──────────────────► FrameData
            │
            ├── segmentation.infer(frame) ──────► SegmentationResult
            │       ├── _preprocess()
            │       ├── backend.infer()
            │       └── _postprocess()
            │               ├── mask_filtering.filter_mask()
            │               ├── centerline_extraction.extract()
            │               └── temporal_smoothing.smooth()
            │
            ├── planner.compute_velocity(seg_result, state) ──► Twist2D
            │
            ├── controller.compute_wheel_velocities(twist) ──► WheelVelocities
            │       └── kinematics.twist_to_wheels()
            │
            ├── odometry.update(wheel_vel) ──────► RobotState
            │
            └── hardware.send_wheel_velocities(wheel_vel)
```

---

## Tunable Parameters Summary

| Layer | File | Key Parameter | Effect |
|-------|------|---------------|--------|
| Perception | `stack_config.yaml` | `temporal_alpha` | Higher = more responsive, lower = smoother |
| Perception | `stack_config.yaml` | `temporal_window` | Larger = more averaging |
| Planner | `stack_config.yaml` | `angular_gain` | Higher = more aggressive steering |
| Planner | `stack_config.yaml` | `deadband` | Wider = fewer corrections |
| Controller | `robot_config.yaml` | `pid.kp/ki/kd` | Velocity tracking gains |
| Robot | `robot_config.yaml` | `max_linear_vel` | Speed limit |
