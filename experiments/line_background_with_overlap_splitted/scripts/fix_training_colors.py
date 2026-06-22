#!/usr/bin/env python3
"""
Fix colors in training visualizations to match experiment color scheme.

Regenerates train_batch*.jpg files with correct colors from metadata.
Run this AFTER training completes.
"""
import sys
import json
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import yaml

# Determine paths
script_dir = Path(__file__).resolve().parent
experiment_dir = script_dir.parent
project_root = experiment_dir.parent.parent


def load_experiment_metadata():
    """Load experiment configuration from metadata."""
    metadata_path = experiment_dir / 'metadata' / 'experiment_config.json'
    with open(metadata_path) as f:
        return json.load(f)


def hex_to_bgr(hex_color):
    """Convert hex color to BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV uses BGR


def create_color_palette_bgr(metadata):
    """Create BGR color palette ordered by class ID."""
    color_scheme = metadata['color_scheme']
    class_mapping = metadata['class_mapping']

    # Create list ordered by class ID
    colors = [None] * len(class_mapping)
    for cat_name, class_id in class_mapping.items():
        bgr = hex_to_bgr(color_scheme[cat_name]['hex'])
        colors[class_id] = bgr

    return colors


def draw_boxes_on_image(image, labels, colors, class_names):
    """
    Draw bounding boxes on image with correct colors.

    Args:
        image: numpy array (H, W, 3) in BGR
        labels: YOLO format labels (N, 5) - [class_id, x_center, y_center, width, height]
        colors: list of BGR tuples
        class_names: list of class names
    """
    h, w = image.shape[:2]

    for label in labels:
        class_id = int(label[0])
        x_center, y_center, box_w, box_h = label[1:]

        # Convert YOLO format to pixel coordinates
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # Get color for this class
        color = colors[class_id]
        class_name = class_names[class_id]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_text = f"{class_name}"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_h - baseline - 4),
            (x1 + text_w, y1),
            color,
            -1
        )

        # Draw label text
        cv2.putText(
            image,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return image


def create_batch_visualization(image_paths, label_paths, colors, class_names, grid_size=(2, 2)):
    """
    Create a grid of training images with bounding boxes.

    Args:
        image_paths: list of image paths
        label_paths: list of label paths
        colors: list of BGR colors
        class_names: list of class names
        grid_size: (rows, cols) for grid
    """
    rows, cols = grid_size
    n_images = min(len(image_paths), rows * cols)

    # Load and annotate images
    annotated = []
    for i in range(n_images):
        # Load image
        img = cv2.imread(str(image_paths[i]))
        if img is None:
            continue

        # Load labels
        labels = []
        if label_paths[i].exists():
            with open(label_paths[i]) as f:
                for line in f:
                    values = [float(x) for x in line.strip().split()]
                    labels.append(values)

        # Draw boxes
        if labels:
            img = draw_boxes_on_image(img, np.array(labels), colors, class_names)

        annotated.append(img)

    # Create grid
    if not annotated:
        return None

    # Resize all to same size
    target_h, target_w = annotated[0].shape[:2]
    resized = [cv2.resize(img, (target_w, target_h)) for img in annotated]

    # Pad if needed
    while len(resized) < rows * cols:
        resized.append(np.zeros_like(resized[0]))

    # Create grid
    grid_rows = []
    for r in range(rows):
        row_images = resized[r * cols:(r + 1) * cols]
        grid_rows.append(np.hstack(row_images))

    grid = np.vstack(grid_rows)
    return grid


def regenerate_training_visualizations(training_dir, config_path, colors, class_names, num_batches=3):
    """Regenerate train_batch*.jpg files with correct colors."""
    training_path = Path(training_dir)

    # Find dataset paths from config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_images_dir = Path(config['train']).parent / 'images' / 'train'
    train_labels_dir = Path(config['train']).parent / 'labels' / 'train'

    if not train_images_dir.exists():
        print(f"Training images not found: {train_images_dir}")
        return

    # Get image and label paths
    image_files = sorted(train_images_dir.glob('*.jpg')) or sorted(train_images_dir.glob('*.png'))

    print(f"Found {len(image_files)} training images")

    # Generate batch visualizations
    for batch_idx in range(num_batches):
        start_idx = batch_idx * 4
        batch_images = image_files[start_idx:start_idx + 4]
        batch_labels = [
            train_labels_dir / f"{img.stem}.txt"
            for img in batch_images
        ]

        if not batch_images:
            break

        print(f"Generating train_batch{batch_idx}.jpg...")
        grid = create_batch_visualization(
            batch_images,
            batch_labels,
            colors,
            class_names,
            grid_size=(2, 2)
        )

        if grid is not None:
            output_path = training_path / f"train_batch{batch_idx}.jpg"
            cv2.imwrite(str(output_path), grid)
            print(f"  ✓ Saved: {output_path}")


def main():
    import argparse

    # Load metadata
    metadata = load_experiment_metadata()

    parser = argparse.ArgumentParser(
        description='Fix training visualization colors to match experiment color scheme'
    )
    parser.add_argument('--training-dir', type=str,
                       default=str(experiment_dir / 'training' / 'run_1'),
                       help='Training directory')
    parser.add_argument('--config', type=str,
                       default=str(experiment_dir / 'config' / 'shapes.yaml'),
                       help='YOLO config path')
    parser.add_argument('--num-batches', type=int, default=3,
                       help='Number of batch visualizations to create')

    args = parser.parse_args()

    print("=" * 70)
    print("Fix Training Visualization Colors")
    print("=" * 70)
    print(f"\nExperiment: {metadata['experiment_name']}")
    print(f"Categories: {', '.join(metadata['categories'])}")
    print(f"\nColor Scheme:")
    for cat_name, color_info in metadata['color_scheme'].items():
        class_id = metadata['class_mapping'][cat_name]
        print(f"  Class {class_id} ({cat_name}): {color_info['hex']} - {color_info['display_name']}")
    print()

    # Create color palette
    colors_bgr = create_color_palette_bgr(metadata)
    class_names = [None] * len(colors_bgr)
    for cat_name, class_id in metadata['class_mapping'].items():
        class_names[class_id] = cat_name

    # Regenerate visualizations
    regenerate_training_visualizations(
        args.training_dir,
        args.config,
        colors_bgr,
        class_names,
        args.num_batches
    )

    print("\n" + "=" * 70)
    print("Complete!")
    print("=" * 70)
    print(f"\nUpdated files in: {args.training_dir}")
    print("  - train_batch0.jpg")
    print("  - train_batch1.jpg")
    print("  - train_batch2.jpg")
    print()


if __name__ == '__main__':
    main()
