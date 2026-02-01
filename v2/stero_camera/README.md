# Stereo Depth Estimation

This repository implements a stereo vision system for real-time depth estimation using two cameras. It detects colored circular objects, estimates their distance from the cameras, and displays tracking information and depth on live video feeds.

## Features

- **HSV Filtering:** Isolates objects of interest based on color using HSV masks ([HSV_filter.py](HSV_filter.py)).
- **Shape Recognition:** Detects circles in the filtered image ([shape_recognition.py](shape_recognition.py)).
- **Depth Calculation:** Computes the distance to the detected object using stereo triangulation ([triangulation.py](triangulation.py)).
- **Camera Calibration & Visualization:** Captures frames from two cameras, applies filters, detects objects, and displays annotated video streams ([camera_calibration.py](camera_calibration.py)).

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- imutils
- matplotlib

Install dependencies with:
```sh
pip install opencv-python numpy imutils matplotlib
```

## Usage

1. Connect two cameras to your system.
2. Run the main script:
    ```sh
    python camera_calibration.py
    ```
3. The program will display live video feeds from both cameras, highlight detected circles, and show the estimated distance. Press `q` to exit.

## File Structure

- [`camera_calibration.py`](camera_calibration.py): Main script for capturing video, filtering, detection, and depth estimation.
- [`HSV_filter.py`](HSV_filter.py): Provides HSV color filtering.
- [`shape_recognition.py`](shape_recognition.py): Detects circles in the filtered image.
- [`triangulation.py`](triangulation.py): Calculates depth using stereo vision principles.
- `.gitignore`, `LICENSE`: Standard repository files.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.