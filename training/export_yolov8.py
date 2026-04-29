"""Export YOLOv8 model to ONNX for use in the C++ pipeline."""

import argparse
import shutil
from pathlib import Path


def export(
    model_path: str,
    output_path: str,
    input_size: int = 640,
    opset: int = 17,
    simplify: bool = True,
) -> None:
    from ultralytics import YOLO

    model = YOLO(model_path)
    print(f"Loaded: {model_path}")
    print(f"Exporting to ONNX (imgsz={input_size}, opset={opset}, simplify={simplify})...")

    exported = model.export(
        format="onnx",
        imgsz=input_size,
        opset=opset,
        simplify=simplify,
        dynamic=False,
        batch=1,
    )

    exported_path = Path(exported)
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exported_path, dest)
    print(f"Saved: {dest}  ({dest.stat().st_size / 1024 / 1024:.1f} MB)")

    # Validate output shape
    import onnx
    m = onnx.load(str(dest))
    onnx.checker.check_model(m)
    out_shape = [d.dim_value for d in m.graph.output[0].type.tensor_type.shape.dim]
    print(f"Output shape: {out_shape}")
    if len(out_shape) == 3 and out_shape[1] > 4:
        n_cls = out_shape[1] - 4
        n_anc = out_shape[2]
        print(f"YOLOv8 format confirmed: {n_cls} classes, {n_anc} anchors")
    print("Export complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")
    parser.add_argument("--model", default="models/yolov8n.pt",
                        help="Path to .pt model (default: models/yolov8n.pt)")
    parser.add_argument("--output", default="models/yolov8n_ground.onnx",
                        help="Destination ONNX path (default: models/yolov8n_ground.onnx)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    export(
        model_path=args.model,
        output_path=args.output,
        input_size=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
    )
