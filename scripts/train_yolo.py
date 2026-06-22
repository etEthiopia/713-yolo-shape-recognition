#!/usr/bin/env python3
"""
Train YOLOv8 model on shape recognition dataset.
Downloads pretrained weights and fine-tunes on custom shapes.
"""
# IMPORTANT: Import SSL config FIRST before any network operations
from ssl_config import setup_ssl_certificates
setup_ssl_certificates()

from ultralytics import YOLO
from pathlib import Path
import argparse
import torch

def train_yolo(config_path: str,
               model_name: str = 'yolov8n.pt',
               epochs: int = 10,
               batch_size: int = 16,
               img_size: int = 640,
               device: str = None,
               project: str = 'runs/shapes',
               name: str = 'experiment_1',
               patience: int = 50,
               save_period: int = 10):
    """
    Train YOLOv8 model on shape dataset.

    Args:
        config_path: Path to dataset YAML config
        model_name: Pretrained model to use (yolov8n/s/m/l/x.pt)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Input image size
        device: Device to use (cuda:0, cpu, etc). None = auto-detect
        project: Project directory for runs
        name: Experiment name
        patience: Early stopping patience
        save_period: Save checkpoint every N epochs
    """
    print("=" * 70)
    print("YOLOv8 Shape Recognition Training")
    print("=" * 70)

    # Auto-detect device if not specified
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    print(f"\nConfiguration:")
    print(f"  Dataset config: {config_path}")
    print(f"  Pretrained model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}")
    print(f"  Output: {project}/{name}")

    # Load model
    print(f"\nLoading pretrained model: {model_name}")
    print("Note: First run will download pretrained weights (may take a few minutes)")
    model = YOLO(model_name)

    # Train
    print("\nStarting training...")
    print("-" * 70)

    results = model.train(
        data=config_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        patience=patience,
        save_period=save_period,
        save=True,
        plots=True,
        verbose=True,
        # Data augmentation settings
        hsv_h=0.015,      # Hue augmentation (minimal - shapes are grayscale)
        hsv_s=0.7,        # Saturation augmentation
        hsv_v=0.4,        # Value augmentation
        degrees=180.0,    # Rotation augmentation (shapes can be rotated)
        translate=0.1,    # Translation augmentation
        scale=0.5,        # Scale augmentation
        shear=0.0,        # No shear (shapes maintain topology)
        perspective=0.0,  # No perspective (2D silhouettes)
        flipud=0.5,       # Flip up-down
        fliplr=0.5,       # Flip left-right
        mosaic=1.0,       # Mosaic augmentation
        mixup=0.0,        # No mixup (classes are shape-based)
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    # Show results location
    weights_dir = Path(project) / name / 'weights'
    print(f"\nModel weights saved to:")
    print(f"  Best: {weights_dir / 'best.pt'}")
    print(f"  Last: {weights_dir / 'last.pt'}")

    results_dir = Path(project) / name
    print(f"\nTraining results saved to: {results_dir}")
    print(f"  - results.png (training curves)")
    print(f"  - confusion_matrix.png")
    print(f"  - results.csv (metrics per epoch)")

    # Print final metrics
    print("\n" + "-" * 70)
    print("Final Metrics:")
    print("-" * 70)

    # Get validation results from last epoch
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"  mAP@0.5: {metrics.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95: {metrics.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {metrics.get('metrics/recall(B)', 0):.4f}")

    return model, results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 on shape dataset')
    parser.add_argument('--config', type=str,
                       default='config/shapes.yaml',
                       help='Path to dataset YAML config')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'],
                       help='Pretrained model size')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda:0, cpu, etc). Auto-detect if not specified')
    parser.add_argument('--project', type=str, default='runs/shapes',
                       help='Project directory')
    parser.add_argument('--name', type=str, default='experiment_1',
                       help='Experiment name')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')

    args = parser.parse_args()

    # Get project directory
    PROJECT_DIR = Path(__file__).parent.parent
    config_path = PROJECT_DIR / args.config

    # Train
    train_yolo(
        config_path=str(config_path),
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience
    )
