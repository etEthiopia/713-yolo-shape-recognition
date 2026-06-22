#!/usr/bin/env python3
"""
Shape Count Scatter Plot

Creates scatter plot showing relationship between actual and predicted number of shapes per image.

Usage:
    python plot_shape_count_scatter.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
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


def extract_shape_counts(per_image_data: list) -> dict:
    """
    Extract actual and predicted shape counts from per-image data.

    Returns:
        Dict with 'actual_counts', 'predicted_counts', and 'image_names'
    """
    actual_counts = []
    predicted_counts = []
    image_names = []

    for image_result in per_image_data:
        # Count ground truth shapes
        ground_truth = image_result.get('ground_truth', [])
        actual_count = len(ground_truth)

        # Count predictions
        predictions = image_result.get('predictions', [])
        predicted_count = len(predictions)

        actual_counts.append(actual_count)
        predicted_counts.append(predicted_count)
        image_names.append(image_result.get('image_name', 'unknown'))

    return {
        'actual_counts': actual_counts,
        'predicted_counts': predicted_counts,
        'image_names': image_names
    }


def calculate_count_statistics(actual_counts: list, predicted_counts: list) -> dict:
    """
    Calculate statistics about shape count predictions.

    Returns:
        Dict with various statistics
    """
    actual_counts = np.array(actual_counts)
    predicted_counts = np.array(predicted_counts)

    # Count accuracy: exact match
    exact_matches = np.sum(actual_counts == predicted_counts)
    accuracy = exact_matches / len(actual_counts) if len(actual_counts) > 0 else 0.0

    # Difference statistics
    differences = predicted_counts - actual_counts
    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences))

    # Over/under prediction
    over_predictions = np.sum(differences > 0)
    under_predictions = np.sum(differences < 0)

    # Per actual-count statistics
    per_actual_count = {}
    unique_counts = sorted(set(actual_counts))

    for count in unique_counts:
        mask = actual_counts == count
        count_predictions = predicted_counts[mask]

        per_actual_count[int(count)] = {
            'n_images': int(np.sum(mask)),
            'mean_predicted': float(np.mean(count_predictions)),
            'std_predicted': float(np.std(count_predictions)),
            'exact_matches': int(np.sum(count_predictions == count)),
            'accuracy': float(np.sum(count_predictions == count) / np.sum(mask))
        }

    return {
        'total_images': len(actual_counts),
        'exact_match_count': int(exact_matches),
        'exact_match_accuracy': float(accuracy),
        'mean_difference': mean_diff,
        'std_difference': std_diff,
        'over_predictions': int(over_predictions),
        'under_predictions': int(under_predictions),
        'per_actual_count': per_actual_count
    }


def plot_shape_count_scatter(actual_counts: list, predicted_counts: list,
                             stats: dict, output_dir: Path):
    """
    Create scatter plot of actual vs predicted shape counts.

    Args:
        actual_counts: List of actual shape counts per image
        predicted_counts: List of predicted shape counts per image
        stats: Statistics dict
        output_dir: Directory to save plot
    """
    actual_counts = np.array(actual_counts)
    predicted_counts = np.array(predicted_counts)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Add jitter for visibility (small random offset)
    np.random.seed(42)
    jitter_x = np.random.normal(0, 0.08, size=len(actual_counts))
    jitter_y = np.random.normal(0, 0.08, size=len(predicted_counts))

    # Scatter plot with transparency
    ax.scatter(actual_counts + jitter_x, predicted_counts + jitter_y,
              alpha=0.4, s=80, color='#00539B', edgecolors='#00539B',
              linewidth=0.5, zorder=3)

    # Perfect prediction line (y=x)
    min_count = min(actual_counts)
    max_count = max(max(actual_counts), max(predicted_counts))
    ax.plot([min_count, max_count], [min_count, max_count],
           'r--', linewidth=3, label='Perfect Prediction (y=x)', zorder=5, alpha=0.8)

    # Calculate mean prediction per actual count for trend line
    unique_counts = sorted(set(actual_counts))
    mean_predictions = []
    for count in unique_counts:
        mask = actual_counts == count
        mean_predictions.append(np.mean(predicted_counts[mask]))

    # Add mean trend line
    ax.plot(unique_counts, mean_predictions, 'o-', color='#2CA02C',
           linewidth=5, markersize=14, label='Mean Prediction per Count',
           zorder=4, alpha=0.9)

    # Styling
    ax.set_xlabel('Actual Number of Shapes', fontsize=20, fontstyle='italic')
    ax.set_ylabel('Predicted Number of Shapes', fontsize=20, fontstyle='italic')

    # Set axis limits
    ax.set_xlim(1.5, max_count + 0.5)
    ax.set_ylim(min(predicted_counts) - 0.5, max(predicted_counts) + 0.5)

    # Integer ticks
    ax.set_xticks(range(2, int(max(actual_counts)) + 1))
    y_ticks = range(int(min(predicted_counts)), int(max(predicted_counts)) + 1)
    ax.set_yticks(y_ticks)

    ax.tick_params(axis='both', labelsize=24)

    # Grid
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.5, zorder=1)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper left', fontsize=18, frameon=True,
             fancybox=False, edgecolor='black', framealpha=0.9)

    # Add text box with statistics
    stats_text = f"Exact Match Accuracy: {stats['exact_match_accuracy']*100:.1f}%\n"
    stats_text += f"Mean Difference: {stats['mean_difference']:+.2f} ± {stats['std_difference']:.2f}\n"
    stats_text += f"Over-predictions: {stats['over_predictions']} | Under-predictions: {stats['under_predictions']}"

    ax.text(0.98, 0.03, stats_text, transform=ax.transAxes,
           fontsize=16, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black'),
           zorder=6)

    # Spines
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(2)
        ax.spines[spine].set_color('black')

    plt.tight_layout()

    # Save
    output_file = output_dir / 'shape_count_scatter.jpg'
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot shape count scatter (actual vs predicted)',
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
    print("Shape Count Scatter Plot")
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

    # Extract counts
    print("Extracting shape counts...")
    count_data = extract_shape_counts(per_image_data)
    print(f"  ✓ Extracted counts for {len(count_data['actual_counts'])} images")
    print()

    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_count_statistics(
        count_data['actual_counts'],
        count_data['predicted_counts']
    )
    print("  ✓ Statistics calculated")
    print()

    # Save to JSON
    output_json = {
        'statistics': stats,
        'per_image_counts': {
            'image_names': count_data['image_names'],
            'actual_counts': count_data['actual_counts'],
            'predicted_counts': count_data['predicted_counts']
        }
    }

    json_file = output_dir / 'shape_count_scatter.json'
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved JSON: {json_file.name}")
    print()

    # Create plot
    print("Generating scatter plot...")
    plot_shape_count_scatter(
        count_data['actual_counts'],
        count_data['predicted_counts'],
        stats,
        output_dir
    )
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Total images: {stats['total_images']}")
    print(f"Exact match accuracy: {stats['exact_match_accuracy']*100:.1f}%")
    print(f"Mean difference (predicted - actual): {stats['mean_difference']:+.2f} ± {stats['std_difference']:.2f}")
    print(f"Over-predictions: {stats['over_predictions']} ({stats['over_predictions']/stats['total_images']*100:.1f}%)")
    print(f"Under-predictions: {stats['under_predictions']} ({stats['under_predictions']/stats['total_images']*100:.1f}%)")
    print()

    print("Per-actual-count breakdown:")
    for count in sorted(stats['per_actual_count'].keys()):
        count_stats = stats['per_actual_count'][count]
        print(f"  Actual {count} shapes ({count_stats['n_images']} images):")
        print(f"    Mean predicted: {count_stats['mean_predicted']:.2f} ± {count_stats['std_predicted']:.2f}")
        print(f"    Exact matches: {count_stats['exact_matches']}/{count_stats['n_images']} ({count_stats['accuracy']*100:.1f}%)")

    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
