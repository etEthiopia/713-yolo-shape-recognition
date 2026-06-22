#!/usr/bin/env python3
"""
Set YOLO's global color palette to match experiment color scheme.

This modifies the ultralytics package's internal colors so ALL visualizations
(training batches, confusion matrix, PR curves) use the correct colors.
"""
import json
from pathlib import Path


def load_experiment_metadata():
    """Load experiment configuration from metadata."""
    script_dir = Path(__file__).resolve().parent
    experiment_dir = script_dir.parent
    metadata_path = experiment_dir / 'metadata' / 'experiment_config.json'

    with open(metadata_path) as f:
        return json.load(f)


def hex_to_rgb_normalized(hex_color):
    """Convert hex color to normalized RGB tuple [0-1]."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (r / 255, g / 255, b / 255)


def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple [0-255] for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


def set_ultralytics_colors(metadata):
    """
    Override YOLO's default color palette with experiment colors.

    This modifies the ultralytics.utils.plotting module's colors.
    """
    try:
        from ultralytics.utils import plotting

        # Create ordered color list
        class_mapping = metadata['class_mapping']
        color_scheme = metadata['color_scheme']

        # Create BGR colors ordered by class ID
        colors_bgr = [None] * len(class_mapping)
        for cat_name, class_id in class_mapping.items():
            bgr = hex_to_bgr(color_scheme[cat_name]['hex'])
            colors_bgr[class_id] = bgr

        # Extend to 20 colors (YOLO's default palette size) by repeating
        while len(colors_bgr) < 20:
            colors_bgr.extend(colors_bgr[:20 - len(colors_bgr)])

        # Override YOLO's color palette
        # The Annotator class uses these colors
        plotting.colors = colors_bgr

        print("✓ YOLO color palette updated:")
        for cat_name, class_id in sorted(class_mapping.items(), key=lambda x: x[1]):
            hex_color = color_scheme[cat_name]['hex']
            display_name = color_scheme[cat_name]['display_name']
            print(f"  Class {class_id} ({cat_name}): {hex_color} - {display_name}")

        return True

    except ImportError as e:
        print(f"Warning: Could not import ultralytics.utils.plotting: {e}")
        return False


def main():
    """Set YOLO colors from experiment metadata."""
    metadata = load_experiment_metadata()

    print("=" * 70)
    print("Setting YOLO Color Palette")
    print("=" * 70)
    print(f"\nExperiment: {metadata['experiment_name']}")
    print()

    success = set_ultralytics_colors(metadata)

    if success:
        print("\nAll YOLO visualizations will now use these colors!")
        print("This affects:")
        print("  - train_batch*.jpg files")
        print("  - Confusion matrix")
        print("  - Precision-Recall curves")
        print("  - Val batch predictions")
    else:
        print("\nFailed to set colors. Training will use default YOLO colors.")

    print("=" * 70)


if __name__ == '__main__':
    main()
