"""Fine-tune YOLOv8 on a custom ground-target dataset (VisDrone, DOTA, or custom YOLO-format data)."""

import argparse
from pathlib import Path


def finetune(
    base_model: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: str,
    name: str,
    export_onnx: bool,
) -> None:
    from ultralytics import YOLO

    model = YOLO(base_model)
    print(f"Base model: {base_model}")
    print(f"Dataset:    {data_yaml}")
    print(f"Training {epochs} epochs @ {imgsz}px, batch {batch}, device {device}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        # Ground-target friendly augmentation settings
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        degrees=5.0,       # UAV footage has limited roll
        perspective=0.001,
        flipud=0.0,        # ground targets have a fixed orientation
        fliplr=0.5,
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest checkpoint: {best_pt}")

    if export_onnx and best_pt.exists():
        print("Exporting best checkpoint to ONNX...")
        from training.export_yolov8 import export
        onnx_out = Path("models") / f"{name}.onnx"
        export(str(best_pt), str(onnx_out), input_size=imgsz)

    print("Fine-tuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for ground target detection")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Base model (.pt) — yolov8n/s/m/l/x (default: yolov8n.pt)")
    parser.add_argument("--data", required=True,
                        help="Path to dataset YAML (YOLO format, e.g. data/ground_targets.yaml)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="CUDA device id or 'cpu'")
    parser.add_argument("--project", default="runs/finetune")
    parser.add_argument("--name", default="ground_targets")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export best.pt to ONNX after training")
    args = parser.parse_args()

    finetune(
        base_model=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        export_onnx=args.export_onnx,
    )
