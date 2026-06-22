#!/usr/bin/env python3
"""
Average Precision vs IoU Threshold Plotter

Calculates AP at multiple IoU thresholds and creates line plots with SEM shading.

Usage:
    python plot_ap_vs_iou_threshold.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Illustrator compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Category configuration (must match bar chart)
CATEGORY_CONFIG = {
    "cat1": {"display_name": "Unconstrained", "color": "#00AADC"},
    "cat2": {"display_name": "Local: Match Variance", "color": "#00539B"},
    "cat4": {"display_name": "Local: Match All", "color": "#7F2F8D"},
    "cat5": {"display_name": "Natural", "color": "#B01F23"},
}

CATEGORY_ORDER = ["cat1", "cat2", "cat4", "cat5"]

# IoU thresholds to test
IOU_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]


def calculate_iou(box1: dict, box2: dict) -> float:
    """Calculate IoU between two YOLO-format bounding boxes."""
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2

    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def match_predictions_at_iou_threshold(predictions: list, ground_truth: list, iou_threshold: float) -> dict:
    """
    Match predictions to ground truth at a specific IoU threshold.

    Returns per-category true positives counts.
    """
    category_results = {cat: {'tp': 0, 'fp': 0, 'fn': 0} for cat in CATEGORY_ORDER}

    # Track which GT boxes have been matched
    used_gt = set()

    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

    for pred_idx, pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in used_gt:
                continue

            # Only match same class
            if pred['class_id'] != gt['class_id']:
                continue

            iou = calculate_iou(pred['bbox'], gt['bbox'])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        pred_cat = pred['category']

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # True positive
            category_results[pred_cat]['tp'] += 1
            used_gt.add(best_gt_idx)
        else:
            # False positive
            category_results[pred_cat]['fp'] += 1

    # Count false negatives (unmatched ground truth)
    for gt_idx, gt in enumerate(ground_truth):
        if gt_idx not in used_gt:
            gt_cat = gt['category']
            category_results[gt_cat]['fn'] += 1

    return category_results


def calculate_ap_at_multiple_thresholds(per_image_data: list, iou_thresholds: list) -> dict:
    """
    Calculate Average Precision for each category at multiple IoU thresholds.

    Returns:
        Dict with threshold -> category -> {values, mean, sem, n}
    """
    results = {}

    for iou_thresh in iou_thresholds:
        # Collect precision values per category at this threshold
        category_precision = {cat: [] for cat in CATEGORY_ORDER}

        for image_result in per_image_data:
            # Get predictions and ground truth
            predictions = image_result.get('predictions', [])
            ground_truth = image_result.get('ground_truth', [])

            # Match at this IoU threshold
            matches = match_predictions_at_iou_threshold(predictions, ground_truth, iou_thresh)

            # Calculate precision per category for this image
            for cat in CATEGORY_ORDER:
                gt_count = sum(1 for gt in ground_truth if gt['category'] == cat)

                if gt_count > 0:
                    tp = matches[cat]['tp']
                    fp = matches[cat]['fp']

                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    category_precision[cat].append(precision)

        # Calculate mean and SEM for each category
        threshold_results = {}
        for cat in CATEGORY_ORDER:
            values = category_precision[cat]
            if values:
                mean = np.mean(values)
                sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
                threshold_results[cat] = {
                    'values': values,
                    'mean': float(mean),
                    'sem': float(sem),
                    'n': len(values)
                }
            else:
                threshold_results[cat] = {
                    'values': [],
                    'mean': 0.0,
                    'sem': 0.0,
                    'n': 0
                }

        results[iou_thresh] = threshold_results

    return results


def plot_ap_vs_iou(results: dict, output_dir: Path):
    """
    Create line plot of AP vs IoU threshold with SEM shading.

    Args:
        results: Dict with threshold -> category -> {mean, sem, n}
        output_dir: Directory to save plot
    """
    thresholds = sorted(results.keys())

    # Extract data for each category
    category_data = {}
    for cat in CATEGORY_ORDER:
        means = []
        sems = []

        for thresh in thresholds:
            cat_result = results[thresh][cat]
            means.append(cat_result['mean'] * 100)  # Convert to percentage
            sems.append(cat_result['sem'] * 100)

        category_data[cat] = {
            'means': np.array(means),
            'sems': np.array(sems),
            'color': CATEGORY_CONFIG[cat]['color'],
            'label': CATEGORY_CONFIG[cat]['display_name']
        }

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each category
    for cat in CATEGORY_ORDER:
        data = category_data[cat]
        means = data['means']
        sems = data['sems']
        color = data['color']
        label = data['label']

        # Plot line
        ax.plot(thresholds, means, color=color, linewidth=3, marker='o',
                markersize=8, label=label, zorder=5)

        # Plot SEM shading
        ax.fill_between(thresholds, means - sems, means + sems,
                        color=color, alpha=0.2, zorder=3)

    # Styling
    ax.set_xlabel('IoU Threshold', fontsize=24, fontstyle='italic')
    ax.set_ylabel('Average Precision (%)', fontsize=24, fontstyle='italic')
    ax.set_ylim(0, 100)
    ax.set_xlim(0.45, 0.95)

    # Set x-axis ticks
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.1f}' for t in thresholds], fontsize=24)

    # Set y-axis ticks
    ax.set_yticks(np.arange(0, 101, 10))
    ax.tick_params(axis='y', labelsize=24)

    # Grid
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.5)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='lower left', fontsize=18, frameon=True,
             fancybox=False, edgecolor='black', framealpha=0.9)

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_color('black')

    plt.tight_layout()

    # Save
    output_file = output_dir / 'ap_vs_iou_threshold.jpg'
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot Average Precision vs IoU Threshold',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('eval_dir', type=str,
                       help='Path to detailed evaluation directory')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom output directory name')

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
    if args.output_name:
        output_dir = eval_dir.parent / args.output_name
    else:
        output_dir = eval_dir.parent / 'detailed_evaluation_stats_and_plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Average Precision vs IoU Threshold")
    print("=" * 70)
    print(f"\nInput: {eval_dir}")
    print(f"Output: {output_dir}")
    print(f"IoU Thresholds: {IOU_THRESHOLDS}")
    print()

    # Load per-image data
    print("Loading per-image results...")
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)
    print(f"  ✓ Loaded {len(per_image_data)} images")
    print()

    # Calculate AP at multiple thresholds
    print("Calculating AP at multiple IoU thresholds...")
    results = calculate_ap_at_multiple_thresholds(per_image_data, IOU_THRESHOLDS)
    print("  ✓ Calculations complete")
    print()

    # Save results to JSON
    output_json = {
        'iou_thresholds': IOU_THRESHOLDS,
        'results': {}
    }

    for thresh in IOU_THRESHOLDS:
        output_json['results'][str(thresh)] = {}
        for cat in CATEGORY_ORDER:
            cat_result = results[thresh][cat]
            output_json['results'][str(thresh)][cat] = {
                'display_name': CATEGORY_CONFIG[cat]['display_name'],
                'color': CATEGORY_CONFIG[cat]['color'],
                'mean': cat_result['mean'],
                'sem': cat_result['sem'],
                'n': cat_result['n']
            }

    json_file = output_dir / 'ap_vs_iou_threshold.json'
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved JSON: {json_file.name}")
    print()

    # Create plot
    print("Generating plot...")
    plot_ap_vs_iou(results, output_dir)
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()

    for cat in CATEGORY_ORDER:
        print(f"{CATEGORY_CONFIG[cat]['display_name']}:")
        for thresh in IOU_THRESHOLDS:
            cat_result = results[thresh][cat]
            print(f"  IoU {thresh:.1f}: {cat_result['mean']*100:5.2f}% ± {cat_result['sem']*100:4.2f}%")
        print()

    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
