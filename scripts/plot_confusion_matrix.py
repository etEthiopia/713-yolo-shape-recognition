#!/usr/bin/env python3
"""
Confusion Matrix for Object Detection

Creates confusion matrix showing predicted vs actual categories.
Rows: Actual categories (ground truth)
Columns: Predicted categories + Background (false positives)

Usage:
    python plot_confusion_matrix.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

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


def build_confusion_matrix(per_image_data: list) -> tuple:
    """
    Build confusion matrix from evaluation results.

    Rows: Actual categories (GT) + Background (for false positives)
    Columns: Predicted categories + Background (for missed detections)

    Returns:
        (confusion_matrix, row_labels, col_labels)
    """
    # Initialize matrix: rows = GT + Background, cols = Predicted + Background
    categories = CATEGORY_ORDER
    n_cats = len(categories)

    # Matrix is (n_cats+1 x n_cats+1):
    # - Last row is "Background" (actual = nothing, predicted = something)
    # - Last column is "Background" (actual = something, predicted = nothing)
    matrix = np.zeros((n_cats + 1, n_cats + 1), dtype=int)

    # Map category to index
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    for image_result in per_image_data:
        ground_truth = image_result.get('ground_truth', [])
        predictions = image_result.get('predictions', [])
        matching = image_result.get('matching', {})

        # Process matches (correct and incorrect classifications)
        matches = matching.get('matches', [])
        for match in matches:
            gt_idx = match['ground_truth_idx']
            pred_idx = match['prediction_idx']

            if gt_idx < len(ground_truth) and pred_idx < len(predictions):
                gt_cat = ground_truth[gt_idx]['category']
                pred_cat = predictions[pred_idx]['category']

                if gt_cat in cat_to_idx and pred_cat in cat_to_idx:
                    row = cat_to_idx[gt_cat]
                    col = cat_to_idx[pred_cat]
                    matrix[row, col] += 1

        # Process false positives (predicted but no ground truth match)
        # Row = Background (actual), Column = Predicted category
        false_positives = matching.get('false_positives', [])
        for fp in false_positives:
            pred_idx = fp['prediction_idx']

            if pred_idx < len(predictions):
                pred_cat = predictions[pred_idx]['category']

                if pred_cat in cat_to_idx:
                    # Background row (last row), predicted category column
                    row = n_cats  # Background row
                    col = cat_to_idx[pred_cat]
                    matrix[row, col] += 1

        # Process false negatives (ground truth missed by detector)
        # Row = GT category, Column = Background (missed)
        false_negatives = matching.get('false_negatives', [])
        for fn in false_negatives:
            gt_idx = fn['ground_truth_idx']

            if gt_idx < len(ground_truth):
                gt_cat = ground_truth[gt_idx]['category']

                if gt_cat in cat_to_idx:
                    row = cat_to_idx[gt_cat]
                    col = n_cats  # Background column
                    matrix[row, col] += 1

    # Row labels (actual categories + background)
    row_labels = [CATEGORY_NAMES[cat] for cat in categories] + ["Background\n(False Positive)"]

    # Column labels (predicted categories + background)
    col_labels = [CATEGORY_NAMES[cat] for cat in categories] + ["Background\n(Missed)"]

    return matrix, row_labels, col_labels


def plot_confusion_matrix(matrix: np.ndarray, row_labels: list, col_labels: list,
                          output_dir: Path):
    """
    Create confusion matrix heatmap with normalized percentages.

    Args:
        matrix: Confusion matrix (rows=actual, cols=predicted)
        row_labels: Row labels (actual categories)
        col_labels: Column labels (predicted categories + background)
        output_dir: Directory to save plot
    """
    # Normalize each row by its sum (total GT instances for that category)
    row_sums = matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums

    # Create annotations showing only percentage
    annot = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            pct = normalized_matrix[i, j] * 100
            if pct > 0:
                annot[i, j] = f'{pct:.1f}%'
            else:
                annot[i, j] = ''

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Create heatmap with normalized values and green colormap
    sns.heatmap(normalized_matrix, annot=annot, fmt='', cmap='Greens',
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'label': 'Percentage (%)', 'format': '%.0f%%'},
                ax=ax, linewidths=1, linecolor='gray', vmin=0, vmax=1,
                annot_kws={'fontsize': 14, 'weight': 'bold'})

    # Labels
    ax.set_xlabel('Predicted Category', fontsize=20, fontstyle='italic', labelpad=15)
    ax.set_ylabel('Actual Category (Ground Truth)', fontsize=20, fontstyle='italic', labelpad=15)
    ax.set_title('Object Detection Confusion Matrix', fontsize=24, fontweight='bold', pad=20)

    # Tick styling
    ax.tick_params(axis='x', labelsize=16, rotation=15)
    ax.tick_params(axis='y', labelsize=16, rotation=0)

    # Color y-axis labels according to category colors
    for i, label in enumerate(ax.get_yticklabels()):
        if i < len(CATEGORY_ORDER):
            cat = CATEGORY_ORDER[i]
            label.set_color(CATEGORY_COLORS[cat])
            label.set_weight('bold')
        else:
            # Last row is background
            label.set_color('gray')
            label.set_weight('bold')

    # Color x-axis labels
    for i, label in enumerate(ax.get_xticklabels()):
        if i < len(CATEGORY_ORDER):
            cat = CATEGORY_ORDER[i]
            label.set_color(CATEGORY_COLORS[cat])
            label.set_weight('bold')
        else:
            # Last column is background
            label.set_color('gray')
            label.set_weight('bold')

    plt.tight_layout()

    # Save
    output_file = output_dir / 'confusion_matrix.jpg'
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")


def calculate_metrics_from_matrix(matrix: np.ndarray, categories: list) -> dict:
    """
    Calculate precision, recall, F1 per category from confusion matrix.

    Args:
        matrix: Confusion matrix (rows=actual, cols=predicted+background)
        categories: List of category names

    Returns:
        Dict with per-category metrics
    """
    n_cats = len(categories)
    metrics = {}

    for i, cat in enumerate(categories):
        # True positives (diagonal)
        tp = matrix[i, i]

        # False negatives (row sum excluding TP)
        fn = matrix[i, :].sum() - tp

        # False positives (column sum excluding TP)
        fp = matrix[:, i].sum() - tp

        # True negatives (all others)
        tn = matrix.sum() - tp - fn - fp

        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[cat] = {
            'display_name': CATEGORY_NAMES[cat],
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'support': int(tp + fn)  # Total ground truth instances
        }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix for object detection',
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
    print("Object Detection Confusion Matrix")
    print("=" * 70)
    print(f"\nInput: {eval_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load per-image data
    print("Loading per-image results...")
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)
    print(f"  ✓ Loaded {len(per_image_data)} images")
    print()

    # Build confusion matrix
    print("Building confusion matrix...")
    matrix, row_labels, col_labels = build_confusion_matrix(per_image_data)
    print("  ✓ Matrix built")
    print()

    # Calculate metrics
    print("Calculating per-category metrics...")
    metrics = calculate_metrics_from_matrix(matrix, CATEGORY_ORDER)
    print("  ✓ Metrics calculated")
    print()

    # Save to JSON
    output_json = {
        'confusion_matrix': matrix.tolist(),
        'row_labels': row_labels,
        'column_labels': col_labels,
        'per_category_metrics': metrics
    }

    json_file = output_dir / 'confusion_matrix.json'
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved JSON: {json_file.name}")
    print()

    # Create plot
    print("Generating confusion matrix plot...")
    plot_confusion_matrix(matrix, row_labels, col_labels, output_dir)
    print()

    # Print summary
    print("=" * 70)
    print("Confusion Matrix (Normalized by Row)")
    print("=" * 70)
    print()

    # Normalize matrix for display
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_matrix = matrix / row_sums

    # Print matrix
    print("                              PREDICTED →")
    header = "ACTUAL ↓".ljust(25) + " ".join([f"{col:>15s}" for col in col_labels])
    print(header)
    print("-" * len(header))

    for i, row_label in enumerate(row_labels):
        row_parts = [row_label.ljust(25)]
        for j in range(matrix.shape[1]):
            count = matrix[i, j]
            pct = normalized_matrix[i, j] * 100
            row_parts.append(f"{count} ({pct:5.1f}%)".rjust(15))
        print(" ".join(row_parts))

    print()
    print("=" * 70)
    print("Per-Category Metrics")
    print("=" * 70)
    print()

    for cat in CATEGORY_ORDER:
        m = metrics[cat]
        print(f"{m['display_name']}:")
        print(f"  Support (GT instances): {m['support']}")
        print(f"  True Positives: {m['true_positives']}")
        print(f"  False Positives: {m['false_positives']}")
        print(f"  False Negatives: {m['false_negatives']}")
        print(f"  Precision: {m['precision']*100:.1f}%")
        print(f"  Recall: {m['recall']*100:.1f}%")
        print(f"  F1 Score: {m['f1_score']*100:.1f}%")
        print()

    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
