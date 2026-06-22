#!/usr/bin/env python3
"""
Visualize Ground Truth vs Predictions Side-by-Side

Creates a side-by-side visualization of ground truth and predicted bounding boxes.

IMPORTANT: Only shows predictions with confidence >= 0.5 (matching evaluation threshold)
YOLO Confidence = IoU × p(object), so 0.5 threshold filters to high-confidence detections.

Usage:
    python visualize_predictions.py experiments/solid_background_with_overlap_splitted/detailed_evaluation test_0000
    python visualize_predictions.py experiments/solid_background_with_overlap_splitted/detailed_evaluation test_0000 --conf-threshold 0.7
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Illustrator compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Category colors (BGR for OpenCV, RGB for matplotlib)
CATEGORY_COLORS_RGB = {
    "cat1": (0/255, 170/255, 220/255),      # #00AADC
    "cat2": (0/255, 83/255, 155/255),       # #00539B
    "cat4": (127/255, 47/255, 141/255),     # #7F2F8D
    "cat5": (176/255, 31/255, 35/255)       # #B01F23
}

CATEGORY_NAMES = {
    "cat1": "Unconstrained",
    "cat2": "Local: Variance",
    "cat4": "Local: Match All",
    "cat5": "Natural"
}


def draw_bboxes_on_image(img, boxes, img_width, img_height, title, show_labels=True):
    """
    Draw bounding boxes on image.

    Args:
        img: PIL Image or numpy array
        boxes: List of dicts with 'bbox', 'category', and optional 'confidence', 'matched'
        img_width: Image width
        img_height: Image height
        title: Title for the subplot
        show_labels: Whether to show category labels

    Returns:
        matplotlib axes
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(img)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
    ax.axis('off')

    for box_info in boxes:
        bbox = box_info['bbox']
        category = box_info['category']
        matched = box_info.get('matched', True)
        confidence = box_info.get('confidence', None)

        # Convert normalized YOLO coordinates to pixel coordinates
        x_center = bbox['x_center'] * img_width
        y_center = bbox['y_center'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height

        # Convert to corner coordinates
        x = x_center - width / 2
        y = y_center - height / 2

        # Get color
        color = CATEGORY_COLORS_RGB.get(category, (0.5, 0.5, 0.5))

        # Style based on match status
        if matched:
            linewidth = 3
            linestyle = '-'
            alpha = 1.0
        else:
            linewidth = 3
            linestyle = '--'
            alpha = 0.7

        # Draw rectangle
        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle,
            alpha=alpha
        )
        ax.add_patch(rect)

        # Add label
        if show_labels:
            label = CATEGORY_NAMES.get(category, category)
            if confidence is not None:
                label = f"{label}\n{confidence:.2f}"

            # Add text background
            text = ax.text(
                x, y - 5,
                label,
                fontsize=10,
                color='white',
                weight='bold',
                va='bottom',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,
                    edgecolor='white',
                    linewidth=1,
                    alpha=0.9
                )
            )

    return ax


def visualize_image_predictions(image_path: Path, image_result: dict, output_path: Path,
                               conf_threshold: float = 0.5):
    """
    Create side-by-side visualization of GT vs Predictions.

    Args:
        image_path: Path to the test image
        image_result: Per-image result dict from evaluation
        output_path: Where to save the visualization
        conf_threshold: Confidence threshold for showing predictions (default: 0.5)
                       YOLO Confidence = IoU × p(object)
    """
    # Load image and convert to RGB if greyscale (preserve original colors)
    img = Image.open(image_path)
    if img.mode == 'L':  # Greyscale
        img = img.convert('RGB')  # Convert to RGB but keep greyscale appearance
    img_width, img_height = img.size

    # Prepare ground truth boxes
    gt_boxes = []
    for gt in image_result['ground_truth']:
        gt_boxes.append({
            'bbox': gt['bbox'],
            'category': gt['category'],
            'matched': True  # We'll mark FN as unmatched
        })

    # Mark false negatives as unmatched
    for fn in image_result['matching']['false_negatives']:
        gt_idx = fn['ground_truth_idx']
        if gt_idx < len(gt_boxes):
            gt_boxes[gt_idx]['matched'] = False

    # Prepare prediction boxes (filter by confidence threshold)
    pred_boxes = []
    pred_idx_map = {}  # Map original index to filtered index
    filtered_idx = 0

    for original_idx, pred in enumerate(image_result['predictions']):
        if pred['confidence'] >= conf_threshold:
            pred_boxes.append({
                'bbox': pred['bbox'],
                'category': pred['category'],
                'confidence': pred['confidence'],
                'matched': True  # We'll mark FP as unmatched
            })
            pred_idx_map[original_idx] = filtered_idx
            filtered_idx += 1

    # Mark false positives as unmatched (only for predictions >= conf_threshold)
    for fp in image_result['matching']['false_positives']:
        pred_idx = fp['prediction_idx']
        if pred_idx in pred_idx_map:
            filtered_idx = pred_idx_map[pred_idx]
            pred_boxes[filtered_idx]['matched'] = False

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left: Ground Truth
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Ground Truth', fontsize=18, fontweight='bold', pad=15)
    ax1.axis('off')

    for i, box_info in enumerate(gt_boxes):
        bbox = box_info['bbox']
        category = box_info['category']
        matched = box_info['matched']

        # Convert to pixel coordinates
        x_center = bbox['x_center'] * img_width
        y_center = bbox['y_center'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height
        x = x_center - width / 2
        y = y_center - height / 2

        color = CATEGORY_COLORS_RGB.get(category, (0.5, 0.5, 0.5))

        # Solid line for detected, dashed for missed
        linestyle = '-' if matched else '--'
        linewidth = 3

        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle
        )
        ax1.add_patch(rect)

    # Right: Predictions
    ax2.imshow(img, cmap='gray')
    ax2.set_title('Predictions', fontsize=18, fontweight='bold', pad=15)
    ax2.axis('off')

    for i, box_info in enumerate(pred_boxes):
        bbox = box_info['bbox']
        category = box_info['category']
        confidence = box_info['confidence']
        matched = box_info['matched']

        # Convert to pixel coordinates
        x_center = bbox['x_center'] * img_width
        y_center = bbox['y_center'] * img_height
        width = bbox['width'] * img_width
        height = bbox['height'] * img_height
        x = x_center - width / 2
        y = y_center - height / 2

        color = CATEGORY_COLORS_RGB.get(category, (0.5, 0.5, 0.5))

        # Solid line for TP, dashed for FP
        linestyle = '-' if matched else '--'
        linewidth = 3

        rect = patches.Rectangle(
            (x, y), width, height,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle
        )
        ax2.add_patch(rect)

        # Add confidence score above bounding box
        ax2.text(
            x, y - 5,
            f'Conf: {confidence:.2f}',
            fontsize=11,
            color='white',
            weight='bold',
            va='bottom',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=color,
                edgecolor='white',
                linewidth=1.5,
                alpha=0.9
            )
        )

    # Add legend at bottom
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # Category colors
    category_legend = [
        Patch(facecolor=CATEGORY_COLORS_RGB['cat1'], edgecolor='black', label='Unconstrained'),
        Patch(facecolor=CATEGORY_COLORS_RGB['cat2'], edgecolor='black', label='Local: Variance'),
        Patch(facecolor=CATEGORY_COLORS_RGB['cat4'], edgecolor='black', label='Local: Match All'),
        Patch(facecolor=CATEGORY_COLORS_RGB['cat5'], edgecolor='black', label='Natural')
    ]

    # Line styles
    style_legend = [
        Line2D([0], [0], color='black', linewidth=3, linestyle='-', label='Solid: True Positive'),
        Line2D([0], [0], color='black', linewidth=3, linestyle='--', label='Dashed: False Negative / False Positive')
    ]

    # Create two legend boxes
    legend1 = fig.legend(handles=category_legend, loc='lower left', ncol=4, fontsize=11,
                        frameon=True, title='Categories', title_fontsize=12,
                        bbox_to_anchor=(0.05, -0.02))
    legend2 = fig.legend(handles=style_legend, loc='lower right', ncol=2, fontsize=11,
                        frameon=True, title='Line Styles', title_fontsize=12,
                        bbox_to_anchor=(0.95, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize ground truth vs predictions side-by-side',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('eval_dir', type=str,
                       help='Path to detailed evaluation directory')
    parser.add_argument('image_name', type=str,
                       help='Image name (e.g., test_0000, test_0001)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                       help='Confidence threshold for showing predictions (default: 0.5)')

    args = parser.parse_args()

    # Paths
    eval_dir = Path(args.eval_dir)

    if not eval_dir.exists():
        print(f"Error: Evaluation directory does not exist: {eval_dir}")
        sys.exit(1)

    per_image_file = eval_dir / 'per_image_results.json'

    if not per_image_file.exists():
        print(f"Error: Per-image results not found: {per_image_file}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = eval_dir.parent / 'visualizations'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find image in dataset
    experiment_dir = eval_dir.parent

    # Determine if train or test
    if 'train_' in args.image_name:
        split = 'train'
    else:
        split = 'test'

    image_name_jpg = args.image_name if args.image_name.endswith('.jpg') else f"{args.image_name}.jpg"
    image_path = experiment_dir / 'dataset' / 'images' / split / image_name_jpg

    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("Visualize Ground Truth vs Predictions")
    print("=" * 70)
    print(f"\nImage: {image_name_jpg}")
    print(f"Confidence threshold: {args.conf_threshold}")
    print(f"Output: {output_dir}")
    print()

    # Load per-image results
    print("Loading evaluation results...")
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)

    # Find the image result
    image_result = None
    for result in per_image_data:
        if image_name_jpg in result['image_path']:
            image_result = result
            break

    if image_result is None:
        print(f"Error: No evaluation results found for {image_name_jpg}")
        sys.exit(1)

    print(f"  ✓ Found evaluation results")
    print()

    # Print summary
    total_predictions = len(image_result['predictions'])
    filtered_predictions = sum(1 for p in image_result['predictions'] if p['confidence'] >= args.conf_threshold)

    print("Summary:")
    print(f"  Ground truth shapes: {len(image_result['ground_truth'])}")
    print(f"  Total predictions: {total_predictions}")
    print(f"  Predictions >= {args.conf_threshold}: {filtered_predictions}")
    print(f"  Matches: {len(image_result['matching']['matches'])}")
    print(f"  False positives: {len(image_result['matching']['false_positives'])}")
    print(f"  False negatives: {len(image_result['matching']['false_negatives'])}")
    print()

    # Create visualization
    output_path = output_dir / f"{args.image_name}_visualization.jpg"
    print("Creating visualization...")
    visualize_image_predictions(image_path, image_result, output_path, args.conf_threshold)
    print()

    print(f"Done! Visualization saved to: {output_path}")
    print()


if __name__ == '__main__':
    main()
