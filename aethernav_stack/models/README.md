# Model files for AetherNav perception

This directory contains the neural network models used by the segmentation engine.

## Files

- `fastscnn_road.onnx` - ONNX model (cross-platform, works on Ubuntu/Jetson)
- `fastscnn_road.engine` - TensorRT engine (Jetson-specific, must be regenerated per device)

## Notes

- The `.engine` files are **platform-specific** - they must be generated on the target Jetson device
- Only the `.onnx` file should be version controlled
- To convert ONNX to TensorRT, use the `trtexec` tool on the Jetson device

## Paths in Config

The `stack_config.yaml` uses relative paths:
```yaml
segmentation:
  engine_path: "models/fastscnn_road.engine"
  onnx_path: "models/fastscnn_road.onnx"
```

These paths are resolved relative to the `aethernav_stack` directory.
