# AetherNav Stack

A modular, professional-grade autonomous vehicle navigation stack for differential drive robots. Designed for lane-following applications with real-time perception, planning, and control.

## Features

- **Multi-threaded Pipeline** - Decoupled camera, perception, and control loops
- **Modular Segmentation** - TensorRT (Jetson) and ONNX Runtime (cross-platform) backends
- **Lane Following Planner** - Real-time centerline extraction and steering
- **Differential Drive Controller** - PID velocity control with odometry
- **Trajectory Analysis** - Smoothness metrics and optimization tools
- **Mock Mode** - Test without hardware using video files

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with live camera
python run.py

# Run with video file (set use_video_file: true in config)
python run.py --log-level DEBUG
```

## Workspace Setup

For automated environment setup, use the setup script:

```bash
# Run from project root
./workspace_setup/setup.sh
```

**The script will:**

- Ask to pull latest from `dev/refactoring` branch
- Create `.aethernav_env` virtual environment
- Install all dependencies from requirements.txt
- Verify installation

After setup, activate with:

```bash
source .aethernav_env/bin/activate
```

## Directory Structure

```
aethernav_stack/
├── run.py                  # Entry point
├── executor.py             # Single-threaded executor
├── executor_threaded.py    # Multi-threaded executor (recommended)
├── config/
│   ├── stack_config.yaml   # System settings (camera, segmentation, planner)
│   ├── robot_config.yaml   # Robot parameters (wheel size, PID gains)
│   └── config_loader.py    # YAML parser and dataclasses
├── core/
│   ├── types.py            # Core data types (Pose2D, Twist2D, etc.)
│   ├── pipeline.py         # CameraThread, PerceptionThread, ControlThread
│   └── logging_config.py   # Logging setup
├── perception/
│   ├── camera_manager.py   # Camera/video capture
│   ├── segmentation/       # Modular inference backends
│   │   ├── engine.py       # Main SegmentationEngine
│   │   ├── onnx_backend.py # ONNX Runtime inference
│   │   ├── tensorrt_backend.py # TensorRT inference
│   │   └── mock_backend.py # Testing without model
│   ├── mask_filtering.py   # Morphological operations
│   ├── centerline_extraction.py # RANSAC centerline fitting
│   └── temporal_smoothing.py    # Frame-to-frame smoothing
├── planner/
│   └── lane_follower.py    # Lane-following velocity planner
├── controls/
│   ├── pid_controller.py   # PID velocity controller
│   └── odometry.py         # Wheel odometry estimation
├── robot_model/
│   ├── params.py           # Robot physical parameters
│   └── diff_drive.py       # Differential drive kinematics
├── hardware/
│   └── hw_interface.py     # Serial communication to motor board
├── models/                 # ONNX/TensorRT model files
└── tests/
    ├── run_trajectory_test.py  # Trajectory analysis tool
    ├── trajectory_logger.py    # Motion command logging
    └── trajectory_analyzer.py  # Smoothness metrics
```

## Configuration

### stack_config.yaml

| Section                   | Key Parameters                                           |
| ------------------------- | -------------------------------------------------------- |
| `hardware`              | `mock_hardware`, `port`, `baudrate`                |
| `camera`                | `use_video_file`, `video_path`, `fps`              |
| `segmentation`          | `inference_backend`, `onnx_path`, `temporal_alpha` |
| `planner.lane_follower` | `cruise_linear_vel`, `angular_gain`, `deadband`    |
| `executor`              | `loop_rate_hz`, `show_visualization`                 |

### robot_config.yaml

| Parameter              | Description             |
| ---------------------- | ----------------------- |
| `wheel_radius`       | Wheel radius in meters  |
| `track_width`        | Distance between wheels |
| `max_linear_vel`     | Maximum forward speed   |
| `pid.linear/angular` | PID gains (kp, ki, kd)  |

## Usage Modes

### 1. Live Camera + Real Hardware

```yaml
# stack_config.yaml
hardware:
  mock_hardware: false
camera:
  use_video_file: false
```

### 2. Video File + Mock Hardware (Testing)

```yaml
hardware:
  mock_hardware: true
camera:
  use_video_file: true
  video_path: "/path/to/video.webm"
```

### 3. Visualization Enabled

```yaml
executor:
  show_visualization: true
```

## Trajectory Analysis

Analyze motion smoothness for tuning:

```bash
# Run 60-second test
python -m aethernav_stack.tests.run_trajectory_test --duration 60

# Analyze existing log
python -m aethernav_stack.tests.run_trajectory_test --analyze-only logs/trajectory.csv
```

**Outputs:**

- `trajectory_path.png` - XY path visualization
- `velocity_profile.png` - Velocity/jerk plots
- `lane_tracking.png` - Lane center tracking
- `smoothness_report.txt` - Metrics summary

## Model Setup

Place your trained models in `models/`:

```bash
models/
├── fastscnn_road.onnx       # ONNX model for CPU/cross-platform
├── fastscnn_road.onnx.data  # External data (if applicable)
└── fastscnn_road.engine     # TensorRT engine for Jetson
```

## Requirements

- Python 3.10+
- OpenCV 4.x
- NumPy
- PyYAML
- ONNX Runtime (for ONNX backend)
- TensorRT + PyCUDA (for Jetson, optional)
- Matplotlib (for trajectory analysis)

## License
