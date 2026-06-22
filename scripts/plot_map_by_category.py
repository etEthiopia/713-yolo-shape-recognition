#!/usr/bin/env python3
"""
Average Precision (AP) Plotter

Creates publication-quality bar charts showing Average Precision per category
with SEM error bars and pairwise t-test statistics.

Usage:
    python plot_map_by_category.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt

# Illustrator compatibility: Keep fonts as editable text
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Category configuration with colors
CATEGORY_CONFIG = {
    "cat1": {"display_name": "Unconstrained", "color": "#00AADC"},
    "cat2": {"display_name": "Local: Match Variance", "color": "#00539B"},
    "cat4": {"display_name": "Local: Match All", "color": "#7F2F8D"},
    "cat5": {"display_name": "Natural", "color": "#B01F23"},
}

CATEGORY_ORDER = ["cat1", "cat2", "cat4", "cat5"]


def load_evaluation_data(eval_dir: Path) -> dict:
    """Load aggregate metrics and per-image results."""
    aggregate_file = eval_dir / 'aggregate_metrics.json'
    per_image_file = eval_dir / 'per_image_results.json'

    if not aggregate_file.exists():
        raise FileNotFoundError(f"Aggregate metrics not found: {aggregate_file}")

    if not per_image_file.exists():
        raise FileNotFoundError(f"Per-image results not found: {per_image_file}")

    with open(aggregate_file, 'r') as f:
        aggregate_data = json.load(f)

    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)

    return {
        'aggregate': aggregate_data,
        'per_image': per_image_data
    }


def calculate_per_category_image_metrics(per_image_data: list) -> dict:
    """
    Calculate per-category Average Precision for each image, then compute SEM.

    Average Precision (AP) is calculated as precision per image where the category appears.

    Returns:
        Dict with category -> {values, mean, sem, n}
    """
    # Organize precision values by category
    category_precision = {cat: [] for cat in CATEGORY_ORDER}

    for image_result in per_image_data:
        per_cat = image_result.get('per_category', {})
        gt_counts = per_cat.get('ground_truth_counts', {})
        pred_counts = per_cat.get('prediction_counts', {})
        tp_counts = per_cat.get('true_positive_counts', {})

        for cat in CATEGORY_ORDER:
            gt = gt_counts.get(cat, 0)
            pred = pred_counts.get(cat, 0)
            tp = tp_counts.get(cat, 0)

            # Only calculate if there are ground truth instances
            if gt > 0:
                precision = tp / pred if pred > 0 else 0.0
                category_precision[cat].append(precision)

    # Calculate mean and SEM (Average Precision across images)
    results = {}
    for cat in CATEGORY_ORDER:
        values = category_precision[cat]
        if values:
            mean = np.mean(values)
            sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
            results[cat] = {
                'values': values,
                'mean': float(mean),
                'sem': float(sem),
                'n': len(values)
            }
        else:
            results[cat] = {
                'values': [],
                'mean': 0.0,
                'sem': 0.0,
                'n': 0
            }

    return results


def calculate_pairwise_ttests(category_data: dict) -> dict:
    """
    Calculate pairwise t-tests between all category pairs.

    Args:
        category_data: Dict with category -> {values, mean, sem, n}

    Returns:
        Dict with pairwise comparisons and p-values
    """
    comparisons = {}

    for i, cat1 in enumerate(CATEGORY_ORDER):
        for cat2 in CATEGORY_ORDER[i+1:]:
            values1 = category_data[cat1]['values']
            values2 = category_data[cat2]['values']

            if len(values1) > 1 and len(values2) > 1:
                # Independent samples t-test
                t_stat, p_value = stats.ttest_ind(values1, values2)

                comparison_key = f"{cat1}_vs_{cat2}"
                comparisons[comparison_key] = {
                    'p_value': float(p_value),
                    't_statistic': float(t_stat),
                    'cat1_mean': float(np.mean(values1)),
                    'cat2_mean': float(np.mean(values2)),
                    'cat1_n': len(values1),
                    'cat2_n': len(values2)
                }

    return comparisons


def create_aggregated_stats(data: dict) -> dict:
    """Create aggregated statistics JSON for plotting."""
    aggregate = data['aggregate']
    per_image_stats = calculate_per_category_image_metrics(data['per_image'])

    # Calculate pairwise t-tests
    pairwise_ttests = calculate_pairwise_ttests(per_image_stats)

    stats = {
        'experiment_info': {
            'total_images': aggregate['overall']['total_images'],
            'total_ground_truth': aggregate['overall']['total_ground_truth'],
            'total_predictions': aggregate['overall']['total_predictions'],
            'iou_threshold': 0.5,
            'confidence_threshold': 0.25
        },
        'overall_metrics': {
            'average_precision': aggregate['overall']['precision']
        },
        'per_category': {},
        'pairwise_ttests': pairwise_ttests
    }

    for cat in CATEGORY_ORDER:
        cat_aggregate = aggregate['per_category'][cat]
        cat_per_image = per_image_stats[cat]

        stats['per_category'][cat] = {
            'display_name': CATEGORY_CONFIG[cat]['display_name'],
            'color': CATEGORY_CONFIG[cat]['color'],
            'average_precision': {
                'mean': cat_per_image['mean'],
                'sem': cat_per_image['sem'],
                'n': cat_per_image['n'],
                'values': cat_per_image['values']  # Keep for potential future analysis
            },
            'aggregate_counts': {
                'ground_truth': cat_aggregate['ground_truth'],
                'predictions': cat_aggregate['predictions'],
                'true_positives': cat_aggregate['true_positives']
            }
        }

    return stats


def plot_average_precision_bar_chart(stats: dict, output_dir: Path):
    """
    Create publication-quality bar chart for Average Precision.

    Args:
        stats: Aggregated statistics
        output_dir: Directory to save plots
    """
    categories = CATEGORY_ORDER
    n_cats = len(categories)

    # Extract data
    means = []
    sems = []
    colors = []
    labels = []

    for cat in categories:
        cat_data = stats['per_category'][cat]['average_precision']
        means.append(cat_data['mean'] * 100)  # Convert to percentage
        sems.append(cat_data['sem'] * 100)
        colors.append(CATEGORY_CONFIG[cat]['color'])
        labels.append(CATEGORY_CONFIG[cat]['display_name'])

    # Bar positioning
    ind = np.arange(n_cats)
    width = 0.6

    # Figure setup
    fig, ax = plt.subplots(figsize=(14, 9))
    border_color = "#000000"
    axis_linewidth = 2.0

    # Plot bars with error bars
    for i, (pos, mean_val, sem_val, color, label) in enumerate(zip(ind, means, sems, colors, labels)):
        # Main bar
        ax.bar(pos, mean_val, width, color=color, edgecolor='none', zorder=5)

        # Error bars (black)
        ax.errorbar(
            pos, mean_val, yerr=sem_val, fmt='none',
            capsize=16, ecolor='grey', linewidth=4.5, capthick=4.5, alpha=1, zorder=6
        )

        # Add white percentage label inside the bar (just below the SEM error bar)
        ax.text(pos, mean_val - sem_val - 2, f'{int(round(mean_val))}%',
               ha='center', va='top', fontsize=24, fontweight='bold', color='white', zorder=7)

    # X-axis with colored diagonal labels (15 degrees)
    ax.set_xticks(ind)
    ax.spines['bottom'].set_visible(True)
    ax.spines['bottom'].set_color(border_color)
    ax.spines['bottom'].set_linewidth(axis_linewidth)
    ax.tick_params(axis='x', width=axis_linewidth, length=8, pad=10)

    # Set labels with matching colors
    ax.set_xticklabels(labels, rotation=10, fontstyle='italic', ha='center', fontsize=20)
    for tick_label, color in zip(ax.get_xticklabels(), colors):
        tick_label.set_color(color)

    # Remove unused spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Y-axis formatting
    ax.yaxis.set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_color(border_color)
    ax.spines['left'].set_linewidth(axis_linewidth)

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel('Average Precision (%)', fontsize=36, fontstyle='italic', labelpad=20)
    ax.tick_params(axis='y', colors="#000000", labelsize=30, width=axis_linewidth, length=8)
    plt.setp(ax.get_yticklabels(), fontstyle='italic')

    plt.tight_layout()

    # Save outputs (JPG only)
    out_jpg = output_dir / 'average_precision_by_category.jpg'

    plt.savefig(out_jpg, dpi=500, transparent=False, bbox_inches='tight', facecolor='white')

    plt.close()

    print(f"  ✓ Saved: {out_jpg.name}")


def plot_all_metrics_combined_DEPRECATED(stats: dict, output_dir: Path, show_yaxis: bool = True):
    """
    Create a combined plot with all three metrics (Precision, Recall, F1).

    Args:
        stats: Aggregated statistics
        output_dir: Directory to save plots
        show_yaxis: Whether to show Y-axis
    """
    categories = CATEGORY_ORDER
    n_cats = len(categories)
    metrics = ['precision', 'recall', 'f1_score']
    metric_labels = ['Precision', 'Recall', 'F1 Score']

    # Extract data for all metrics
    data_by_metric = {}
    for metric in metrics:
        means = []
        sems = []
        for cat in categories:
            cat_data = stats['per_category'][cat]['per_image_stats'][metric]
            means.append(cat_data['mean'] * 100)
            sems.append(cat_data['sem'] * 100)
        data_by_metric[metric] = {'means': means, 'sems': sems}

    # Bar positioning
    ind = np.arange(n_cats)
    width = 0.25
    offset = width

    positions = {
        'precision': ind - offset,
        'recall': ind,
        'f1_score': ind + offset
    }

    # Figure setup
    fig, ax = plt.subplots(figsize=(18, 9))
    border_color = "#000000"
    axis_linewidth = 2.0

    # Plot bars for each metric
    for metric_idx, metric in enumerate(metrics):
        means = data_by_metric[metric]['means']
        sems = data_by_metric[metric]['sems']
        pos = positions[metric]

        for i, cat in enumerate(categories):
            color = CATEGORY_CONFIG[cat]['color']

            # Adjust alpha for different metrics to distinguish them
            alpha = 1.0 if metric == 'precision' else 0.7 if metric == 'recall' else 0.5

            ax.bar(pos[i], means[i], width, color=color, edgecolor='none',
                   alpha=alpha, zorder=5, label=metric_labels[metric_idx] if i == 0 else "")

            ax.errorbar(
                pos[i], means[i], yerr=sems[i], fmt='none',
                capsize=15, ecolor=color, linewidth=5, capthick=5, alpha=alpha, zorder=6
            )

    # Remove unused spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Remove X-axis
    ax.xaxis.set_visible(False)

    # Y-axis formatting
    if show_yaxis:
        ax.yaxis.set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color(border_color)
        ax.spines['left'].set_linewidth(axis_linewidth)

        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel('Performance (%)', fontsize=36, fontstyle='italic', labelpad=20)
        ax.tick_params(axis='y', colors="#000000", labelsize=30, width=axis_linewidth, length=8)
        plt.setp(ax.get_yticklabels(), fontstyle='italic')
    else:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_ylim(0, 100)

    # Legend
    ax.legend(loc='upper right', fontsize=24, frameon=False)

    plt.tight_layout()

    # Save outputs
    suffix = "with_yaxis" if show_yaxis else "no_yaxis"
    out_pdf = output_dir / f"all_metrics_combined_{suffix}.pdf"
    out_png = output_dir / f"all_metrics_combined_{suffix}.jpg"

    plt.savefig(out_pdf, transparent=True, bbox_inches='tight')
    plt.savefig(out_png, dpi=500, transparent=True, bbox_inches='tight')

    plt.close()

    print(f"  ✓ Saved: {out_pdf.name}")
    print(f"  ✓ Saved: {out_png.name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate publication-quality mAP plots from detailed evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_map_by_category.py experiments/solid_background_with_overlap_splitted/detailed_evaluation

  python plot_map_by_category.py experiments/natural_scenes_background_with_overlap_splitted/detailed_evaluation \\
      --no-yaxis
        """
    )

    parser.add_argument('eval_dir', type=str,
                       help='Path to detailed evaluation directory')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Custom output directory name (default: detailed_evaluation_stats_and_plots)')

    args = parser.parse_args()

    # Paths
    eval_dir = Path(args.eval_dir)

    if not eval_dir.exists():
        print(f"Error: Evaluation directory does not exist: {eval_dir}")
        sys.exit(1)

    # Output directory
    if args.output_name:
        output_dir = eval_dir.parent / args.output_name
    else:
        output_dir = eval_dir.parent / 'detailed_evaluation_stats_and_plots'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("mAP Aggregator and Plotter")
    print("=" * 70)
    print(f"\nInput: {eval_dir}")
    print(f"Output: {output_dir}")
    print()

    # Load data
    print("Loading evaluation data...")
    data = load_evaluation_data(eval_dir)
    print(f"  ✓ Loaded {len(data['per_image'])} images")
    print()

    # Create aggregated stats
    print("Computing aggregated statistics...")
    stats = create_aggregated_stats(data)

    # Save stats JSON
    stats_file = output_dir / 'aggregated_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ Saved: {stats_file.name}")
    print()

    # Create plot
    print("Generating Average Precision plot...")
    plot_average_precision_bar_chart(stats, output_dir)

    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"\nAverage Precision per Category (mean ± SEM):")
    print()

    for cat in CATEGORY_ORDER:
        cat_data = stats['per_category'][cat]
        ap = cat_data['average_precision']
        print(f"{cat_data['display_name']:25s}: {ap['mean']*100:5.2f}% ± {ap['sem']*100:4.2f}%  (n={ap['n']})")

    print()
    print("Pairwise t-tests:")
    for comparison, result in stats['pairwise_ttests'].items():
        cat1, cat2 = comparison.split('_vs_')
        name1 = CATEGORY_CONFIG[cat1]['display_name']
        name2 = CATEGORY_CONFIG[cat2]['display_name']
        print(f"  {name1:25s} vs {name2:25s}: p = {result['p_value']:.4f}")

    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
