# Model Weights Provenance and Licensing

Model weights (`.pt`, `.onnx`) are excluded from version control via `.gitignore` to keep the repository lightweight.

## Supported Pose Estimation Backends

1. **ONNX Runtime Backend (`OnnxPoseBackend`)**:
   - **Recommended Model**: `yolov8n-pose.onnx` or `yolo11n-pose.onnx` (exported at `imgsz=640`, opset `17`).
   - **Expected Path**: `project/models/yolov8n-pose.onnx`
   - **Runtime Dependency**: `onnxruntime==1.31.0` (pre-installed).
   - **Keypoint Convention**: COCO 17 keypoints (`[x, y, confidence]`).

2. **Ultralytics PyTorch Backend (`UltralyticsPoseBackend`)**:
   - **Recommended Model**: `yolov8n-pose.pt` or `yolo11n-pose.pt`
   - **Expected Path**: `project/models/yolov8n-pose.pt`
   - **Official Documentation**: [Ultralytics Pose Task](https://docs.ultralytics.com/tasks/pose)
   - **Export Command**:
     ```bash
     yolo export model=yolov8n-pose.pt format=onnx imgsz=640 simplify=True
     ```

3. **Deterministic Synthetic Backend (`SyntheticPoseBackend`)**:
   - Built-in deterministic pose generator used when testing offline or benchmarking pipeline throughput without external weight files.

## Licensing Notice

Pretrained Ultralytics YOLO weights are governed by the **GNU Affero General Public License v3.0 (AGPL-3.0)** unless an enterprise license is acquired from Ultralytics. Verify licensing compliance prior to commercial redistribution.
