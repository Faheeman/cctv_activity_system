"""Utility script to export Ultralytics YOLO-Pose weights to ONNX and record SHA-256 provenance."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Export YOLO-Pose model to ONNX and verify SHA-256 checksum")
    parser.add_argument("--weights", type=Path, default=Path("models/yolov8n-pose.pt"), help="Input .pt weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Error: ultralytics is not installed in this environment. Install with `pip install .[yolo]`.",
            file=sys.stderr,
        )
        return 1

    model = YOLO(str(args.weights))
    exported_path = Path(model.export(format="onnx", imgsz=args.imgsz, simplify=True))
    manifest = {
        "source_weights": str(args.weights),
        "exported_onnx": str(exported_path),
        "imgsz": args.imgsz,
        "sha256": compute_sha256(exported_path),
        "license": "AGPL-3.0 (or Ultralytics Enterprise License)",
    }
    manifest_path = exported_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
