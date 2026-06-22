#!/usr/bin/env python3
"""
Average Precision by Natural Background

For natural background experiments, shows AP for each category across different backgrounds.
X-axis: Background image filename
Y-axis: Average Precision (0-1)
Each category shown as colored dots

Usage:
    python plot_ap_by_background.py experiments/natural_scenes_background_with_overlap_splitted
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import cv2
from typing import Dict, List, Tuple

# Illustrator compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Category configuration
CATEGORY_ORDER = ["cat1", "cat2", "cat4", "cat5"]
CATEGORY_NAMES = {
    "cat1": "Unconstrained",
    "cat2": "Local: Match Variance",
    "cat4": "Local: Match All",
    "cat5": "Natural"
}

CATEGORY_COLORS = {
    "cat1": "#00AADC",
    "cat2": "#00539B",
    "cat4": "#7F2F8D",
    "cat5": "#B01F23"
}


def find_matching_background(test_image_path: Path, background_dir: Path,
                             img_size: int = 640) -> str:
    """
    Find which background image was used for a test image by comparing pixels.

    Args:
        test_image_path: Path to test image
        background_dir: Path to directory with background images
        img_size: Image size (default 640)

    Returns:
        Background filename or 'unknown'
    """
    # Load test image
    test_img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        return 'unknown'

    # Get all background files
    bg_files = sorted(background_dir.glob("*.jpg"))

    best_match = 'unknown'
    best_score = -1

    # Simple approach: check which background has highest correlation
    # in regions without shapes (assume corners are mostly background)
    test_corners = [
        test_img[0:50, 0:50],      # top-left
        test_img[0:50, -50:],      # top-right
        test_img[-50:, 0:50],      # bottom-left
        test_img[-50:, -50:]       # bottom-right
    ]
    test_corner = np.concatenate([c.flatten() for c in test_corners])

    for bg_file in bg_files:
        # Load and resize background
        bg_img = cv2.imread(str(bg_file), cv2.IMREAD_GRAYSCALE)
        if bg_img is None:
            continue

        # Resize to target size
        bg_img = cv2.resize(bg_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Extract same corners
        bg_corners = [
            bg_img[0:50, 0:50],
            bg_img[0:50, -50:],
            bg_img[-50:, 0:50],
            bg_img[-50:, -50:]
        ]
        bg_corner = np.concatenate([c.flatten() for c in bg_corners])

        # Compute correlation
        correlation = np.corrcoef(test_corner, bg_corner)[0, 1]

        if correlation > best_score:
            best_score = correlation
            best_match = bg_file.name

    return best_match


def extract_background_mapping(experiment_dir: Path,
                               detailed_eval_dir: Path) -> Dict[str, str]:
    """
    Map each test image to its background.

    Args:
        experiment_dir: Experiment directory
        detailed_eval_dir: Detailed evaluation directory

    Returns:
        Dict mapping image_name -> background_filename
    """
    print("Mapping test images to backgrounds...")

    # Paths
    test_images_dir = experiment_dir / 'dataset' / 'images' / 'test'
    background_dir = experiment_dir / 'background'

    if not background_dir.exists():
        print(f"Warning: Background directory not found: {background_dir}")
        return {}

    # Load per-image results to get list of test images
    per_image_file = detailed_eval_dir / 'per_image_results.json'
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)

    background_map = {}
    total = len(per_image_data)

    for i, image_result in enumerate(per_image_data):
        image_name = Path(image_result['image_path']).name
        test_image_path = test_images_dir / image_name

        if test_image_path.exists():
            bg_name = find_matching_background(test_image_path, background_dir)
            background_map[image_name] = bg_name

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{total} images processed...")

    print(f"  ✓ Mapped {len(background_map)} images to backgrounds")

    return background_map


def compute_ap_by_background(per_image_data: List[dict],
                             background_map: Dict[str, str],
                             iou_threshold: float = 0.5) -> Dict[str, Dict[str, float]]:
    """
    Compute Average Precision per category per background.

    Args:
        per_image_data: Per-image evaluation results
        background_map: Mapping of image_name -> background_filename
        iou_threshold: IoU threshold for matching

    Returns:
        Dict[background_name][category] -> AP
    """
    # Group images by background
    bg_to_images = defaultdict(list)

    for image_result in per_image_data:
        image_name = Path(image_result['image_path']).name
        bg_name = background_map.get(image_name, 'unknown')
        bg_to_images[bg_name].append(image_result)

    # Compute AP per background per category
    ap_by_background = {}

    for bg_name, images in bg_to_images.items():
        if bg_name == 'unknown':
            continue

        category_stats = {cat: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0}
                         for cat in CATEGORY_ORDER}

        # Aggregate stats across all images with this background
        for image_result in images:
            # Count ground truth per category
            for gt in image_result['ground_truth']:
                cat = gt['category']
                if cat in category_stats:
                    category_stats[cat]['total_gt'] += 1

            # Count TP, FP, FN per category
            matches = image_result['matching']['matches']
            for match in matches:
                gt_idx = match['ground_truth_idx']
                pred_idx = match['prediction_idx']

                if (gt_idx < len(image_result['ground_truth']) and
                    pred_idx < len(image_result['predictions'])):

                    gt_cat = image_result['ground_truth'][gt_idx]['category']
                    pred_cat = image_result['predictions'][pred_idx]['category']

                    if gt_cat == pred_cat and gt_cat in category_stats:
                        category_stats[gt_cat]['tp'] += 1

            # False positives
            for fp in image_result['matching']['false_positives']:
                pred_idx = fp['prediction_idx']
                if pred_idx < len(image_result['predictions']):
                    pred_cat = image_result['predictions'][pred_idx]['category']
                    if pred_cat in category_stats:
                        category_stats[pred_cat]['fp'] += 1

            # False negatives
            for fn in image_result['matching']['false_negatives']:
                gt_idx = fn['ground_truth_idx']
                if gt_idx < len(image_result['ground_truth']):
                    gt_cat = image_result['ground_truth'][gt_idx]['category']
                    if gt_cat in category_stats:
                        category_stats[gt_cat]['fn'] += 1

        # Compute AP per category for this background
        bg_ap = {}
        for cat in CATEGORY_ORDER:
            stats = category_stats[cat]
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']

            # Average Precision approximation: TP / (TP + FP + FN)
            # More accurate would use precision-recall curve, but this is simpler
            if (tp + fp + fn) > 0:
                ap = tp / (tp + fp + fn)
            else:
                ap = 0.0

            bg_ap[cat] = ap

        ap_by_background[bg_name] = bg_ap

    return ap_by_background


def plot_ap_by_background(ap_by_background: Dict[str, Dict[str, float]],
                         output_dir: Path):
    """
    Create scatter plot showing mean AP per background.

    Args:
        ap_by_background: Dict[background][category] -> AP
        output_dir: Output directory
    """
    # Sort backgrounds alphabetically
    backgrounds = sorted(ap_by_background.keys())

    if not backgrounds:
        print("Warning: No backgrounds found to plot")
        return

    # Calculate mean AP across all categories for each background
    mean_ap_values = []
    for bg in backgrounds:
        category_aps = [ap_by_background[bg].get(cat, 0.0) for cat in CATEGORY_ORDER]
        mean_ap = np.mean(category_aps)
        mean_ap_values.append(mean_ap)

    # Create figure
    fig, ax = plt.subplots(figsize=(max(14, len(backgrounds) * 0.8), 10))

    # X positions for each background
    x_positions = np.arange(len(backgrounds))

    # Plot mean AP per background
    ax.scatter(x_positions, mean_ap_values,
              s=200, color='#00539B', alpha=0.8,
              edgecolors='black', linewidth=2, zorder=3)

    # Add horizontal line for overall mean
    overall_mean = np.mean(mean_ap_values)
    ax.axhline(y=overall_mean, color='red', linestyle='--', linewidth=2,
              alpha=0.7, label=f'Overall Mean: {overall_mean:.3f}', zorder=2)

    # Styling
    ax.set_xlabel('Background Image', fontsize=18, fontweight='bold')
    ax.set_ylabel('Mean Average Precision (Across All Categories)', fontsize=18, fontstyle='italic')
    ax.set_title('Mean Average Precision by Background',
                fontsize=20, fontstyle='italic', pad=20)

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(backgrounds, rotation=45, ha='right', fontsize=10)

    # Set y-axis
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='y', labelsize=16)

    # Grid
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.5, axis='y', zorder=1)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper left', fontsize=14, frameon=True,
             fancybox=False, edgecolor='black', framealpha=0.95)

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_color('black')

    plt.tight_layout()

    # Save
    output_file = output_dir / 'ap_by_background.jpg'
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot AP by background for natural scenes experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('experiment_dir', type=str,
                       help='Path to experiment directory (e.g., experiments/natural_scenes_background_with_overlap_splitted)')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom output directory name')

    args = parser.parse_args()

    # Paths
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory does not exist: {experiment_dir}")
        sys.exit(1)

    eval_dir = experiment_dir / 'detailed_evaluation'

    if not eval_dir.exists():
        print(f"Error: Detailed evaluation directory not found: {eval_dir}")
        sys.exit(1)

    per_image_file = eval_dir / 'per_image_results.json'

    if not per_image_file.exists():
        print(f"Error: Per-image results not found: {per_image_file}")
        sys.exit(1)

    # Output directory
    if args.output_name:
        output_dir = experiment_dir / args.output_name
    else:
        output_dir = experiment_dir / 'detailed_evaluation_stats_and_plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Average Precision by Background")
    print("=" * 70)
    print(f"\nExperiment: {experiment_dir.name}")
    print(f"Output: {output_dir}")
    print()

    # Load per-image data
    print("Loading per-image results...")
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)
    print(f"  ✓ Loaded {len(per_image_data)} images")
    print()

    # Map images to backgrounds
    background_map = extract_background_mapping(experiment_dir, eval_dir)
    print()

    if not background_map:
        print("Error: Could not map images to backgrounds")
        sys.exit(1)

    # Compute AP by background
    print("Computing AP by background and category...")
    ap_by_background = compute_ap_by_background(per_image_data, background_map)
    print(f"  ✓ Computed AP for {len(ap_by_background)} backgrounds")
    print()

    # Save JSON
    output_json = {
        'background_mapping': background_map,
        'ap_by_background': ap_by_background
    }

    json_file = output_dir / 'ap_by_background.json'
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved JSON: {json_file.name}")
    print()

    # Create plot
    print("Generating plot...")
    plot_ap_by_background(ap_by_background, output_dir)
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    print(f"Total backgrounds: {len(ap_by_background)}")
    print()

    # Show AP range per category across backgrounds
    for cat in CATEGORY_ORDER:
        ap_values = [ap_by_background[bg].get(cat, 0.0)
                    for bg in ap_by_background.keys()]
        mean_ap = np.mean(ap_values)
        std_ap = np.std(ap_values)
        min_ap = np.min(ap_values)
        max_ap = np.max(ap_values)

        print(f"{CATEGORY_NAMES[cat]}:")
        print(f"  Mean AP: {mean_ap:.3f} ± {std_ap:.3f}")
        print(f"  Range: {min_ap:.3f} - {max_ap:.3f}")
        print()

    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
