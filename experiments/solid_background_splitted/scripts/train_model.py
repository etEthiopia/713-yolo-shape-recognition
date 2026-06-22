#!/usr/bin/env python3
"""
Train YOLO model for solid_background_splitted experiment.

Wrapper around main train_yolo.py with experiment-specific settings loaded from metadata.
"""
import sys
import json
from pathlib import Path

# Determine paths
script_dir = Path(__file__).resolve().parent
experiment_dir = script_dir.parent
project_root = experiment_dir.parent.parent

# Add project scripts to path
sys.path.insert(0, str(project_root / 'scripts'))

# Import main training function
from train_yolo import train_yolo


def load_experiment_metadata():
    """Load experiment configuration from metadata."""
    metadata_path = experiment_dir / 'metadata' / 'experiment_config.json'

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}\n"
            f"Run setup.py first!"
        )

    with open(metadata_path) as f:
        return json.load(f)


def main():
    """Train YOLO model with experiment-specific settings."""
    import argparse

    # Load metadata for defaults
    metadata = load_experiment_metadata()
    params = metadata['parameters']

    parser = argparse.ArgumentParser(
        description='Train YOLO for solid_background_splitted experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str,
                       default=str(experiment_dir / 'config' / 'shapes.yaml'),
                       help='YOLO config path')
    parser.add_argument('--model', type=str,
                       default=params['model'],
                       help='Pretrained model')
    parser.add_argument('--epochs', type=int,
                       default=params['epochs'],
                       help='Training epochs')
    parser.add_argument('--batch', type=int,
                       default=params['batch_size'],
                       help='Batch size')
    parser.add_argument('--img-size', type=int,
                       default=params['image_size'],
                       help='Input image size')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (auto-detect if not specified)')
    parser.add_argument('--project', type=str,
                       default=str(experiment_dir / 'training'),
                       help='Project directory')
    parser.add_argument('--name', type=str, default='run_1',
                       help='Experiment name')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs')

    args = parser.parse_args()

    print("=" * 70)
    print("Solid Background Experiment - Training")
    print("=" * 70)
    print(f"\nExperiment: {metadata['experiment_name']}")
    print(f"Categories: {', '.join(metadata['categories'])}")
    print(f"Excluded: {', '.join(metadata.get('excluded_categories', []))}")
    print(f"\nConfiguration:")
    print(f"  Config: {args.config}")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.img_size}")
    print(f"  Output: {args.project}/{args.name}")
    print()

    # Call main training function
    train_yolo(
        config_path=args.config,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save_period=args.save_period
    )

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nModel weights: {args.project}/{args.name}/weights/best.pt")
    print(f"\nNext steps:")
    print(f"  1. Fix training visualization colors: python fix_training_colors.py")
    print(f"  2. Evaluate model: python evaluate_model.py")
    print()


if __name__ == '__main__':
    main()
