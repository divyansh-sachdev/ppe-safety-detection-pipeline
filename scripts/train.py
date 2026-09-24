"""Fine-tunes a YOLOv11 model on a PPE compliance dataset.

Usage:
    python scripts/train.py --data ppe.yaml --epochs 100
"""
import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to the dataset YAML (Ultralytics format)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", default="yolo11n.pt", help="Base weights to fine-tune from")
    args = parser.parse_args()

    model = YOLO(args.weights)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)
    model.export(format="onnx")


if __name__ == "__main__":
    main()
