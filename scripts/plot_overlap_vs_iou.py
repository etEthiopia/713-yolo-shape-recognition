#!/usr/bin/env python3
"""
Overlap Ratio vs Mean IoU Scatter Plot

Creates scatter plot showing relationship between image overlap ratio and mean IoU.

Usage:
    python plot_overlap_vs_iou.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
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


def extract_overlap_and_iou(per_image_data: list) -> dict:
    """
    Extract overlap ratio (as percentage) and mean IoU from per-image data.

    Returns:
        Dict with 'overlap_percentages', 'mean_ious', and 'image_names'
    """
    overlap_percentages = []
    mean_ious = []
    image_names = []

    for image_result in per_image_data:
        # Get overlap ratio as percentage
        overlap_stats = image_result.get('overlap_stats', {})
        overlap_ratio = overlap_stats.get('overlap_ratio', 0.0)
        overlap_percentage = overlap_ratio * 100  # Convert to percentage

        # Get mean IoU
        metrics = image_result.get('metrics', {})
        mean_iou = metrics.get('mean_iou', 0.0)

        # Only include if mean_iou is available (i.e., there were matches)
        if mean_iou > 0:
            overlap_percentages.append(overlap_percentage)
            mean_ious.append(mean_iou)
            image_names.append(image_result.get('image_path', 'unknown'))

    return {
        'overlap_percentages': overlap_percentages,
        'mean_ious': mean_ious,
        'image_names': image_names
    }


def calculate_statistics(overlap_percentages: list, mean_ious: list) -> dict:
    """
    Calculate statistics about the relationship.

    Returns:
        Dict with correlation and other stats
    """
    overlap_percentages = np.array(overlap_percentages)
    mean_ious = np.array(mean_ious)

    # Correlation
    correlation = np.corrcoef(overlap_percentages, mean_ious)[0, 1]

    # Mean and std
    mean_overlap = float(np.mean(overlap_percentages))
    std_overlap = float(np.std(overlap_percentages))
    mean_iou = float(np.mean(mean_ious))
    std_iou = float(np.std(mean_ious))

    return {
        'correlation': float(correlation),
        'mean_overlap_percentage': mean_overlap,
        'std_overlap_percentage': std_overlap,
        'mean_iou': mean_iou,
        'std_iou': std_iou,
        'total_images': len(overlap_percentages),
        'overlap_range': [float(min(overlap_percentages)), float(max(overlap_percentages))],
        'iou_range': [float(min(mean_ious)), float(max(mean_ious))]
    }


def plot_overlap_vs_iou_scatter(overlap_percentages: list, mean_ious: list,
                                 stats: dict, output_dir: Path):
    """
    Create scatter plot of image overlap ratio vs mean IoU.

    Args:
        overlap_percentages: List of overlap percentages per image
        mean_ious: List of mean IoU values per image
        stats: Statistics dict
        output_dir: Directory to save plot
    """
    overlap_percentages = np.array(overlap_percentages)
    mean_ious = np.array(mean_ious)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Scatter plot with transparency
    ax.scatter(overlap_percentages, mean_ious,
              alpha=0.5, s=100, color='#00539B', edgecolors='#00539B',
              linewidth=0.5, zorder=3)

    # Add trend line (linear regression)
    z = np.polyfit(overlap_percentages, mean_ious, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(overlap_percentages), max(overlap_percentages), 100)
    ax.plot(x_trend, p(x_trend), 'r--', linewidth=3,
           label=f'Trend (r={stats["correlation"]:.3f})', zorder=4, alpha=0.8)

    # Styling
    ax.set_xlabel('Image Overlap Ratio (%)', fontsize=20, fontstyle='italic')
    ax.set_ylabel('Mean IoU', fontsize=20, fontstyle='italic')

    # Set axis limits
    x_margin = (max(overlap_percentages) - min(overlap_percentages)) * 0.05
    ax.set_xlim(min(overlap_percentages) - x_margin, max(overlap_percentages) + x_margin)
    ax.set_ylim(0, 1.0)

    # Set y-axis ticks
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', labelsize=24)

    # Grid
    ax.grid(True, alpha=0.6, linestyle='--', linewidth=1.5, zorder=1)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='upper right', fontsize=18, frameon=True,
             fancybox=False, edgecolor='black', framealpha=0.9)

    # Add text box with statistics
    stats_text = f"Correlation: {stats['correlation']:.3f}\n"
    stats_text += f"Mean Overlap: {stats['mean_overlap_percentage']:.1f}%\n"
    stats_text += f"Mean IoU: {stats['mean_iou']:.3f}"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=16, verticalalignment='top', horizontalalignment='left',
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
    output_file = output_dir / 'overlap_vs_iou.jpg'
    plt.savefig(output_file, dpi=500, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  ✓ Saved: {output_file.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot overlap ratio vs mean IoU (scatter plot)',
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
    print("Overlap Ratio vs Mean IoU Scatter Plot")
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

    # Extract overlap and IoU data
    print("Extracting overlap ratios and mean IoUs...")
    data = extract_overlap_and_iou(per_image_data)
    print(f"  ✓ Extracted {len(data['overlap_percentages'])} valid data points")
    print()

    # Calculate statistics
    print("Calculating statistics...")
    stats = calculate_statistics(data['overlap_percentages'], data['mean_ious'])
    print("  ✓ Statistics calculated")
    print()

    # Save to JSON
    output_json = {
        'statistics': stats,
        'per_image_data': {
            'image_names': data['image_names'],
            'overlap_percentages': data['overlap_percentages'],
            'mean_ious': data['mean_ious']
        }
    }

    json_file = output_dir / 'overlap_vs_iou.json'
    with open(json_file, 'w') as f:
        json.dump(output_json, f, indent=2)
    print(f"Saved JSON: {json_file.name}")
    print()

    # Create plot
    print("Generating scatter plot...")
    plot_overlap_vs_iou_scatter(
        data['overlap_percentages'],
        data['mean_ious'],
        stats,
        output_dir
    )
    print()

    # Print summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Total images with valid IoU: {stats['total_images']}")
    print(f"Overlap ratio range: {stats['overlap_range'][0]:.1f}% - {stats['overlap_range'][1]:.1f}%")
    print(f"Mean IoU range: {stats['iou_range'][0]:.3f} - {stats['iou_range'][1]:.3f}")
    print(f"Mean overlap ratio: {stats['mean_overlap_percentage']:.1f}% ± {stats['std_overlap_percentage']:.1f}%")
    print(f"Mean IoU: {stats['mean_iou']:.3f} ± {stats['std_iou']:.3f}")
    print()
    print(f"Pearson correlation: {stats['correlation']:.3f}")
    if stats['correlation'] < -0.3:
        print("  → Strong negative correlation: Higher overlap → Lower IoU")
    elif stats['correlation'] < -0.1:
        print("  → Weak negative correlation: Higher overlap → Slightly lower IoU")
    elif stats['correlation'] < 0.1:
        print("  → No correlation: Overlap doesn't affect IoU")
    elif stats['correlation'] < 0.3:
        print("  → Weak positive correlation: Higher overlap → Slightly higher IoU")
    else:
        print("  → Strong positive correlation: Higher overlap → Higher IoU")

    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
