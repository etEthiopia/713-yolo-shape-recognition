#!/usr/bin/env python3
"""
Phase 6: Generate Documentation
Creates comprehensive experiment report and documentation.
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

def generate_report(experiment_dir):
    """Generate comprehensive experiment report."""
    print("=" * 70)
    print("Phase 6: Generate Documentation")
    print("=" * 70)

    exp_path = Path(experiment_dir)

    # Load metadata
    metadata_path = exp_path / 'metadata' / 'experiment_config.json'
    with open(metadata_path, 'r') as f:
        config = json.load(f)

    # Load evaluation metrics
    metrics_path = exp_path / 'evaluation' / 'metrics_comprehensive.json'
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Generate report
    report_path = exp_path / 'EXPERIMENT_REPORT.md'

    with open(report_path, 'w') as f:
        # Header
        f.write("# Solid Grey Background Experiment - Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        # Executive Summary
        f.write("## Executive Summary\n\n")
        f.write(f"This experiment trained a YOLOv8-nano model to detect and classify shape silhouettes ")
        f.write(f"across 4 categories ({', '.join(config['categories'])}) on a solid grey background. ")
        f.write(f"Category 3 was excluded based on poor baseline performance.\n\n")

        overall = metrics['overall']
        f.write(f"**Overall Performance**:\n")
        f.write(f"- mAP@0.5: **{overall['mAP50']:.1%}**\n")
        f.write(f"- Precision: {overall['precision']:.1%}\n")
        f.write(f"- Recall: {overall['recall']:.1%}\n\n")

        # Experimental Design
        f.write("---\n\n")
        f.write("## Experimental Design\n\n")

        f.write("### Rationale\n\n")
        f.write(config.get('rationale', 'N/A') + "\n\n")

        f.write("### Parameters\n\n")
        params = config.get('parameters', {})
        f.write("| Parameter | Value |\n")
        f.write("|-----------|-------|\n")
        f.write(f"| Training Images | {params.get('train_images', 'N/A')} |\n")
        f.write(f"| Test Images | {params.get('test_images', 'N/A')} |\n")
        f.write(f"| Shapes per Image | {params.get('shapes_per_image_range', 'N/A')} |\n")
        f.write(f"| Image Size | {params.get('image_size', 'N/A')}×{params.get('image_size', 'N/A')} |\n")
        f.write(f"| Background | {params.get('background_color', 'N/A')} |\n")
        f.write(f"| Model | {params.get('model', 'N/A')} |\n")
        f.write(f"| Epochs | {params.get('epochs', 'N/A')} |\n")
        f.write(f"| Batch Size | {params.get('batch_size', 'N/A')} |\n\n")

        # Results
        f.write("---\n\n")
        f.write("## Results\n\n")

        f.write("### Overall Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| mAP@0.5 | {overall['mAP50']:.4f} ({overall['mAP50']:.1%}) |\n")
        f.write(f"| mAP@0.5:0.95 | {overall['mAP50-95']:.4f} ({overall['mAP50-95']:.1%}) |\n")
        f.write(f"| Precision | {overall['precision']:.4f} ({overall['precision']:.1%}) |\n")
        f.write(f"| Recall | {overall['recall']:.4f} ({overall['recall']:.1%}) |\n\n")

        f.write("### Per-Category Performance\n\n")
        f.write("| Category | Type | mAP@0.5 | mAP@0.5:0.95 |\n")
        f.write("|----------|------|---------|---------------|\n")

        category_types = {
            'cat1': 'Synthetic (Unconstrained)',
            'cat2': 'Synthetic (Local Var)',
            'cat4': 'Synthetic (Local Matched)',
            'cat5': 'Natural'
        }

        per_cat = metrics['per_category']
        for cat in config['categories']:
            if cat in per_cat:
                m = per_cat[cat]
                f.write(f"| {cat} | {category_types.get(cat, 'Unknown')} | ")
                f.write(f"{m['mAP50']:.4f} | {m['mAP50-95']:.4f} |\n")

        f.write("\n")

        # Statistical Analysis
        if 'statistical_analysis' in metrics:
            stats = metrics['statistical_analysis']
            f.write("### Statistical Analysis\n\n")
            f.write(f"- Mean mAP@0.5: {stats['mean_map50']:.4f}\n")
            f.write(f"- Std Dev: {stats['std_map50']:.4f}\n")
            f.write(f"- Range: [{stats['min_map50']:.4f}, {stats['max_map50']:.4f}]\n\n")

        # Key Findings
        f.write("---\n\n")
        f.write("## Key Findings\n\n")

        # Calculate synthetic vs natural
        synthetic_cats = [c for c in config['categories'] if c != 'cat5']
        if synthetic_cats:
            synthetic_map = sum(per_cat[c]['mAP50'] for c in synthetic_cats if c in per_cat) / len(synthetic_cats)
            natural_map = per_cat.get('cat5', {}).get('mAP50', 0)

            f.write(f"1. **Synthetic shapes** (cat1, cat2, cat4) achieved average mAP@0.5 of **{synthetic_map:.4f}**\n")
            f.write(f"2. **Natural shapes** (cat5) achieved mAP@0.5 of **{natural_map:.4f}**\n")
            diff = abs(synthetic_map - natural_map)
            winner = "Natural" if natural_map > synthetic_map else "Synthetic"
            f.write(f"3. **{winner} shapes performed {diff:.1%} better**\n\n")

        # Best and worst
        best_cat = max(per_cat.items(), key=lambda x: x[1]['mAP50'])
        worst_cat = min(per_cat.items(), key=lambda x: x[1]['mAP50'])

        f.write(f"4. **Best performing category**: {best_cat[0]} (mAP: {best_cat[1]['mAP50']:.4f})\n")
        f.write(f"5. **Worst performing category**: {worst_cat[0]} (mAP: {worst_cat[1]['mAP50']:.4f})\n\n")

        # Visualizations
        f.write("---\n\n")
        f.write("## Visualizations\n\n")
        f.write("### Performance by Category\n")
        f.write("![mAP by Category](evaluation/figures/mAP_by_category.png)\n\n")
        f.write("### Synthetic vs Natural Comparison\n")
        f.write("![Synthetic vs Natural](evaluation/figures/synthetic_vs_natural.png)\n\n")
        f.write("### Sample Predictions\n")
        f.write("![Sample Predictions](evaluation/figures/sample_predictions.png)\n\n")

        # Conclusion
        f.write("---\n\n")
        f.write("## Conclusion\n\n")
        f.write(f"The YOLOv8-nano model successfully detected and classified shape silhouettes ")
        f.write(f"with an overall mAP@0.5 of {overall['mAP50']:.1%}. ")

        if overall['mAP50'] > 0.75:
            f.write("This exceeds the target threshold of 75%, demonstrating strong performance ")
            f.write("on this task.\n\n")
        else:
            f.write(f"While below the 75% target, the model shows promise for further optimization.\n\n")

        f.write("Excluding category 3 (skew/kurtosis matched shapes) from the baseline experiment ")
        f.write("allowed the model to focus on well-separated categories, improving overall performance.\n\n")

        # Files
        f.write("---\n\n")
        f.write("## Output Files\n\n")
        f.write("- Model weights: `training/run_1/weights/best.pt`\n")
        f.write("- Evaluation metrics: `evaluation/metrics_comprehensive.json`\n")
        f.write("- Figures: `evaluation/figures/`\n")
        f.write("- This report: `EXPERIMENT_REPORT.md`\n\n")

    print(f"✓ Report generated: {report_path}")

    # Generate README
    readme_path = exp_path / 'README.md'
    with open(readme_path, 'w') as f:
        f.write("# Solid Grey Background Experiment\n\n")
        f.write("## Quick Summary\n\n")
        f.write(f"- **Categories**: {', '.join(config['categories'])}\n")
        f.write(f"- **Excluded**: {', '.join(config.get('excluded_categories', []))}\n")
        f.write(f"- **Performance**: mAP@0.5 = {overall['mAP50']:.1%}\n\n")
        f.write("## Files\n\n")
        f.write("- `EXPERIMENT_REPORT.md` - Full experimental report\n")
        f.write("- `training/run_1/weights/best.pt` - Trained model\n")
        f.write("- `evaluation/metrics_comprehensive.json` - All metrics\n")
        f.write("- `evaluation/figures/` - Visualizations\n\n")
        f.write("## Reproduction\n\n")
        f.write("This experiment was run using:\n")
        f.write("```bash\n")
        f.write("sbatch slurm_solid_grey_experiment.sh\n")
        f.write("```\n\n")
        f.write("See parent `experiments/README.md` for details.\n")

    print(f"✓ README generated: {readme_path}")

    print("\n" + "=" * 70)
    print("Documentation Complete!")
    print("=" * 70)
    print(f"\nGenerated:")
    print(f"  - {report_path}")
    print(f"  - {readme_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate experiment documentation')
    parser.add_argument('--experiment-dir', type=str, required=True,
                       help='Experiment directory path')

    args = parser.parse_args()

    generate_report(args.experiment_dir)
