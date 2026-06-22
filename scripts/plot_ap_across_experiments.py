#!/usr/bin/env python3
"""
Plot Average Precision Across Experiments

Creates a scatter plot showing how each category's AP changes across different
experimental conditions (no overlap, overlap, line background, natural scenes).

Usage:
    python plot_ap_across_experiments.py
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Illustrator compatibility
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Category colors
CATEGORY_COLORS = {
    "cat1": "#00AADC",  # Cyan blue
    "cat2": "#00539B",  # Dark blue
    "cat4": "#7F2F8D",  # Purple
    "cat5": "#B01F23"   # Red
}

CATEGORY_NAMES = {
    "cat1": "Unconstrained",
    "cat2": "Local: Match Variance",
    "cat4": "Local: Match All",
    "cat5": "Natural"
}

# Experiment order
EXPERIMENTS = [
    {
        'name': 'Solid\n(No Overlap)',
        'path': 'experiments/solid_background_splitted/detailed_evaluation_stats_and_plots/aggregated_stats.json'
    },
    {
        'name': 'Solid\n(With Overlap)',
        'path': 'experiments/solid_background_with_overlap_splitted/detailed_evaluation_stats_and_plots/aggregated_stats.json'
    },
    {
        'name': 'Line\nBackground',
        'path': 'experiments/line_background_with_overlap_splitted/detailed_evaluation_stats_and_plots/aggregated_stats.json'
    },
    {
        'name': 'Natural\nScenes',
        'path': 'experiments/natural_scenes_background_with_overlap_splitted/detailed_evaluation_stats_and_plots/aggregated_stats.json'
    }
]

CATEGORIES = ['cat1', 'cat2', 'cat4', 'cat5']


def load_experiment_data():
    """Load AP data from all experiments."""
    data = {}

    for exp in EXPERIMENTS:
        exp_name = exp['name']
        exp_path = Path(exp['path'])

        if not exp_path.exists():
            print(f"Warning: {exp_path} not found, skipping...")
            continue

        with open(exp_path, 'r') as f:
            exp_data = json.load(f)

        data[exp_name] = {}
        for cat in CATEGORIES:
            if cat in exp_data['per_category']:
                mean_ap = exp_data['per_category'][cat]['average_precision']['mean']
                data[exp_name][cat] = mean_ap * 100  # Convert to percentage
            else:
                print(f"Warning: {cat} not found in {exp_name}")

    return data


def create_scatter_plot(data, output_path):
    """Create scatter plot with connecting lines."""
    fig, ax = plt.subplots(figsize=(14, 10))

    # X-axis positions
    x_positions = np.arange(len(EXPERIMENTS))
    x_labels = [exp['name'] for exp in EXPERIMENTS]

    # Plot each category
    for cat in CATEGORIES:
        color = CATEGORY_COLORS[cat]
        display_name = CATEGORY_NAMES[cat]

        # Collect y-values for this category
        y_values = []
        x_values = []

        for i, exp in enumerate(EXPERIMENTS):
            exp_name = exp['name']
            if exp_name in data and cat in data[exp_name]:
                y_values.append(data[exp_name][cat])
                x_values.append(i)

        # Plot scatter points
        ax.scatter(x_values, y_values,
                  color=color,
                  s=200,  # Point size
                  alpha=0.8,
                  edgecolors='white',
                  linewidth=2,
                  zorder=3,
                  label=display_name)

        # Connect points with dotted line
        if len(x_values) > 1:
            ax.plot(x_values, y_values,
                   color=color,
                   linestyle='--',
                   linewidth=2.5,
                   alpha=0.6,
                   zorder=2)

    # Styling
    ax.set_xlabel('Experimental Condition', fontsize=22, fontweight='bold')
    ax.set_ylabel('Average Precision (%)', fontsize=22, fontweight='bold')
    ax.set_title('Category Performance Across Experimental Conditions',
                fontsize=24, fontweight='bold', pad=20)

    # Set y-axis limits and ticks
    ax.set_ylim(50, 100)
    ax.set_yticks(np.arange(50, 101, 5))
    ax.tick_params(axis='both', labelsize=18)

    # Set x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, fontsize=18)
    ax.set_xlim(-0.5, len(EXPERIMENTS) - 0.5)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1, zorder=1)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(loc='lower left', fontsize=16, frameon=True,
             fancybox=False, edgecolor='black', framealpha=0.95)

    # Add reference line at 100%
    ax.axhline(y=100, color='gray', linestyle='-', linewidth=1, alpha=0.3, zorder=1)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved: {output_path}")

    plt.close()


def print_summary(data):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Average Precision Summary (%)".center(80))
    print("=" * 80)
    print()

    # Header
    header = f"{'Category':<20}"
    for exp in EXPERIMENTS:
        header += f"{exp['name'].replace(chr(10), ' '):<20}"
    print(header)
    print("-" * 80)

    # Data rows
    for cat in CATEGORIES:
        row = f"{CATEGORY_NAMES[cat]:<20}"
        for exp in EXPERIMENTS:
            exp_name = exp['name']
            if exp_name in data and cat in data[exp_name]:
                value = data[exp_name][cat]
                row += f"{value:>18.2f}  "
            else:
                row += f"{'N/A':>18}  "
        print(row)

    print()


def main():
    print("=" * 80)
    print("Plot Average Precision Across Experiments")
    print("=" * 80)
    print()

    # Load data
    print("Loading data from all experiments...")
    data = load_experiment_data()
    print(f"✓ Loaded data from {len(data)} experiments")
    print()

    # Print summary
    print_summary(data)

    # Create plot
    output_path = Path('experiments/ap_across_experiments.jpg')
    print(f"Creating scatter plot...")
    create_scatter_plot(data, output_path)
    print()

    print("=" * 80)
    print(f"Done! Plot saved to: {output_path}")
    print("=" * 80)


if __name__ == '__main__':
    main()
