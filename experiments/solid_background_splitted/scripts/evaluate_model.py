#!/usr/bin/env python3
"""
Evaluate YOLO model for solid_background_splitted experiment.

Wrapper around main evaluate_model.py with experiment-specific color scheme.
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

# Import main evaluation function
from evaluate_model import evaluate_model


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


def load_color_scheme_bgr(metadata):
    """
    Load color scheme and convert to BGR format for OpenCV.

    Returns:
        dict: Map from class_id -> BGR tuple
    """
    color_scheme = metadata['color_scheme']
    class_mapping = metadata['class_mapping']

    colors_bgr = {}
    for cat_name, color_info in color_scheme.items():
        class_id = class_mapping[cat_name]
        rgb = tuple(color_info['rgb'])
        # Convert RGB to BGR for OpenCV
        bgr = (rgb[2], rgb[1], rgb[0])
        colors_bgr[class_id] = bgr

    return colors_bgr


def find_latest_weights(training_dir):
    """Find the latest trained model weights."""
    training_path = Path(training_dir)

    # Look for run directories
    run_dirs = sorted([d for d in training_path.iterdir() if d.is_dir() and d.name.startswith('run')])

    if not run_dirs:
        raise FileNotFoundError(f"No training runs found in {training_dir}")

    # Get latest run
    latest_run = run_dirs[-1]
    best_weights = latest_run / 'weights' / 'best.pt'

    if not best_weights.exists():
        raise FileNotFoundError(f"No weights found at {best_weights}")

    return str(best_weights)


def main():
    """Evaluate YOLO model with experiment-specific settings."""
    import argparse

    # Load metadata
    metadata = load_experiment_metadata()
    color_scheme_bgr = load_color_scheme_bgr(metadata)

    parser = argparse.ArgumentParser(
        description='Evaluate YOLO for solid_background_splitted experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--weights', type=str,
                       help='Model weights path (auto-detect if not specified)')
    parser.add_argument('--config', type=str,
                       default=str(experiment_dir / 'config' / 'shapes.yaml'),
                       help='YOLO config path')
    parser.add_argument('--output-dir', type=str,
                       default=str(experiment_dir / 'evaluation'),
                       help='Output directory')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (auto-detect if not specified)')
    parser.add_argument('--save-predictions', action='store_true', default=True,
                       help='Save visualizations')
    parser.add_argument('--max-vis', type=int, default=50,
                       help='Maximum images to visualize')

    args = parser.parse_args()

    # Auto-detect weights if not specified
    if args.weights is None:
        print("Auto-detecting latest trained model...")
        args.weights = find_latest_weights(experiment_dir / 'training')
        print(f"Found: {args.weights}")

    print("=" * 70)
    print("Solid Background Experiment - Evaluation")
    print("=" * 70)
    print(f"\nExperiment: {metadata['experiment_name']}")
    print(f"Categories: {', '.join(metadata['categories'])}")
    print(f"Excluded: {', '.join(metadata.get('excluded_categories', []))}")
    print(f"\nColor Scheme:")
    for cat_name, color_info in metadata['color_scheme'].items():
        class_id = metadata['class_mapping'][cat_name]
        print(f"  Class {class_id} ({cat_name}): {color_info['hex']} - {color_info['display_name']}")
    print(f"\nConfiguration:")
    print(f"  Weights: {args.weights}")
    print(f"  Config: {args.config}")
    print(f"  Output: {args.output_dir}")
    print(f"  Confidence threshold: {args.conf_threshold}")
    print(f"  IoU threshold: {args.iou_threshold}")
    print()

    # Call main evaluation function
    # Note: The main evaluate_model doesn't support all these parameters yet
    # For now, we call it with the parameters it accepts
    evaluate_model(
        weights_path=args.weights,
        config_path=args.config,
        output_dir=args.output_dir,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold
    )

    # TODO: Color scheme integration
    # The main evaluate_model.py doesn't yet support custom color schemes.
    # For now, visualizations use default YOLO colors.
    # Future enhancement: modify main script to accept color_scheme_bgr parameter

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Metrics: {args.output_dir}/evaluation_summary.json")
    print(f"  - Visualizations: {args.output_dir}/predictions/")
    print()


if __name__ == '__main__':
    main()
