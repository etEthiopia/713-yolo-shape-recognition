#!/usr/bin/env python3
"""
Generate All Plots for an Experiment

Runs all plotting scripts for a given experiment folder.

Usage:
    python generate_all_plots.py experiments/solid_background_with_overlap_splitted
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_script(script_name, args, description):
    """Run a plotting script and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}")

    cmd = [sys.executable, script_name] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ {description} failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Generate all plots for an experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all_plots.py experiments/solid_background_with_overlap_splitted
  python generate_all_plots.py experiments/line_background_with_overlap_splitted
        """
    )

    parser.add_argument('experiment_dir', type=str,
                       help='Path to experiment directory')
    parser.add_argument('--skip', type=str, nargs='+',
                       choices=['shape_count', 'overlap_iou', 'map', 'confusion', 'ap_iou', 'ap_background'],
                       help='Skip specific plots')

    args = parser.parse_args()

    # Validate paths
    experiment_dir = Path(args.experiment_dir)

    if not experiment_dir.exists():
        print(f"Error: Experiment directory does not exist: {experiment_dir}")
        sys.exit(1)

    eval_dir = experiment_dir / 'detailed_evaluation'

    if not eval_dir.exists():
        print(f"Error: Detailed evaluation directory not found: {eval_dir}")
        print("Please run detailed_evaluation.py first!")
        sys.exit(1)

    # Get script directory
    script_dir = Path(__file__).parent

    print("="*70)
    print("Generate All Plots")
    print("="*70)
    print(f"\nExperiment: {experiment_dir}")
    print(f"Evaluation: {eval_dir}")
    print()

    # Prepare skip set
    skip = set(args.skip) if args.skip else set()

    # Track results
    results = {}

    # 1. Average Precision by Category (Bar Chart)
    if 'map' not in skip:
        results['map'] = run_script(
            script_dir / 'plot_map_by_category.py',
            [str(eval_dir)],
            "Average Precision by Category (Bar Chart)"
        )

    # 2. Confusion Matrix
    if 'confusion' not in skip:
        results['confusion'] = run_script(
            script_dir / 'plot_confusion_matrix.py',
            [str(eval_dir)],
            "Confusion Matrix"
        )

    # 3. AP vs IoU Threshold (Line Graph)
    if 'ap_iou' not in skip:
        results['ap_iou'] = run_script(
            script_dir / 'plot_ap_vs_iou_threshold.py',
            [str(eval_dir)],
            "AP vs IoU Threshold (Line Graph)"
        )

    # 4. Overlap vs Mean IoU (Scatter Plot)
    if 'overlap_iou' not in skip:
        results['overlap_iou'] = run_script(
            script_dir / 'plot_overlap_vs_iou.py',
            [str(eval_dir)],
            "Overlap Ratio vs Mean IoU (Scatter Plot)"
        )

    # 5. Shape Count Scatter Plot
    if 'shape_count' not in skip:
        results['shape_count'] = run_script(
            script_dir / 'plot_shape_count_scatter.py',
            [str(eval_dir)],
            "Shape Count Scatter Plot"
        )

    # 6. AP by Background (for natural background experiments only)
    background_dir = experiment_dir / 'background'
    if background_dir.exists() and 'ap_background' not in skip:
        results['ap_background'] = run_script(
            script_dir / 'plot_ap_by_background.py',
            [str(experiment_dir)],
            "Average Precision by Background (Natural Scenes)"
        )

    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print()

    total = len(results)
    successful = sum(1 for v in results.values() if v)

    print(f"Total plots: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print()

    if successful == total:
        print("✓ All plots generated successfully!")
    else:
        print("✗ Some plots failed. See output above for details.")

    print()
    print(f"Output directory: {experiment_dir / 'detailed_evaluation_stats_and_plots'}")
    print()

    # Exit with error code if any failed
    sys.exit(0 if successful == total else 1)


if __name__ == '__main__':
    main()
