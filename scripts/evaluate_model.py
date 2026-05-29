#!/usr/bin/env python3
"""
Evaluate trained YOLO model and generate comprehensive analysis.
Computes per-category metrics, confusion matrix, and visualizations.
"""
# IMPORTANT: Import SSL config FIRST before any network operations
from ssl_config import setup_ssl_certificates
setup_ssl_certificates()

from ultralytics import YOLO
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import cv2

def evaluate_model(weights_path: str,
                   config_path: str,
                   output_dir: str = 'results',
                   conf_threshold: float = 0.25,
                   iou_threshold: float = 0.5):
    """
    Comprehensive evaluation of trained YOLO model.

    Args:
        weights_path: Path to trained model weights (.pt file)
        config_path: Path to dataset YAML config
        output_dir: Output directory for results
        conf_threshold: Confidence threshold for predictions
        iou_threshold: IoU threshold for evaluation
    """
    print("=" * 70)
    print("YOLOv8 Shape Recognition Evaluation")
    print("=" * 70)

    # Load model
    print(f"\nLoading model from: {weights_path}")
    model = YOLO(weights_path)

    # Create output directories
    output_path = Path(output_dir)
    figures_dir = output_path / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Run validation
    print("\nRunning validation on test set...")
    print("-" * 70)

    metrics = model.val(
        data=config_path,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=True,
        save_json=True
    )

    # Extract metrics
    print("\n" + "=" * 70)
    print("Overall Metrics (IoU=0.5)")
    print("=" * 70)

    results_dict = metrics.results_dict
    overall_map50 = results_dict.get('metrics/mAP50(B)', 0)
    overall_map = results_dict.get('metrics/mAP50-95(B)', 0)
    overall_precision = results_dict.get('metrics/precision(B)', 0)
    overall_recall = results_dict.get('metrics/recall(B)', 0)

    print(f"  mAP@0.5: {overall_map50:.4f}")
    print(f"  mAP@0.5:0.95: {overall_map:.4f}")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")

    # Per-class metrics
    print("\n" + "=" * 70)
    print("Per-Category Metrics")
    print("=" * 70)

    class_names = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']

    # Get per-class metrics from model
    per_class_metrics = {}

    if hasattr(metrics, 'ap_class_index') and hasattr(metrics, 'box'):
        # Extract AP per class
        box_metrics = metrics.box

        for idx, class_name in enumerate(class_names):
            if hasattr(box_metrics, 'ap'):
                ap50 = box_metrics.ap[idx, 0] if len(box_metrics.ap) > idx else 0
                ap = box_metrics.ap[idx].mean() if len(box_metrics.ap) > idx else 0
            else:
                ap50 = 0
                ap = 0

            per_class_metrics[class_name] = {
                'mAP@0.5': float(ap50),
                'mAP@0.5:0.95': float(ap),
            }

            print(f"\n{class_name}:")
            print(f"  mAP@0.5: {ap50:.4f}")
            print(f"  mAP@0.5:0.95: {ap:.4f}")

    # Create summary table
    print("\n" + "=" * 70)
    print("Category Performance Summary")
    print("=" * 70)
    print(f"{'Category':<10} {'Type':<15} {'mAP@0.5':<10} {'mAP@0.5:0.95':<15}")
    print("-" * 70)

    category_types = {
        'cat1': 'Synthetic (max entropy)',
        'cat2': 'Synthetic (var)',
        'cat3': 'Synthetic (skew/kurt)',
        'cat4': 'Synthetic (all)',
        'cat5': 'Natural'
    }

    for cat in class_names:
        if cat in per_class_metrics:
            metrics_data = per_class_metrics[cat]
            print(f"{cat:<10} {category_types[cat]:<15} "
                  f"{metrics_data['mAP@0.5']:<10.4f} "
                  f"{metrics_data['mAP@0.5:0.95']:<15.4f}")

    # Save metrics to JSON
    summary = {
        'overall': {
            'mAP@0.5': float(overall_map50),
            'mAP@0.5:0.95': float(overall_map),
            'precision': float(overall_precision),
            'recall': float(overall_recall)
        },
        'per_category': per_class_metrics,
        'evaluation_params': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold
        }
    }

    summary_path = output_path / 'evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Evaluation summary saved to: {summary_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. mAP bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = list(per_class_metrics.keys())
    map_values = [per_class_metrics[cat]['mAP@0.5'] for cat in categories]

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    bars = ax.bar(categories, map_values, color=colors, alpha=0.8, edgecolor='black')

    ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Performance by Shape Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    map_chart_path = figures_dir / 'mAP_by_category.png'
    plt.savefig(map_chart_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ mAP chart: {map_chart_path}")
    plt.close()

    # 2. Synthetic vs Natural comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    synthetic_map = np.mean([per_class_metrics[f'cat{i}']['mAP@0.5']
                            for i in range(1, 5)])
    natural_map = per_class_metrics['cat5']['mAP@0.5']

    comparison_data = [synthetic_map, natural_map]
    comparison_labels = ['Synthetic\n(cat1-4)', 'Natural\n(cat5)']

    bars = ax.bar(comparison_labels, comparison_data,
                  color=['#4ECDC4', '#98D8C8'], alpha=0.8, edgecolor='black')

    ax.set_ylabel('Mean mAP@0.5', fontsize=12, fontweight='bold')
    ax.set_title('Synthetic vs Natural Shape Recognition', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    comparison_path = figures_dir / 'synthetic_vs_natural.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Comparison chart: {comparison_path}")
    plt.close()

    # Generate sample predictions
    print("\nGenerating sample predictions...")
    generate_sample_predictions(model, config_path, figures_dir, num_samples=20)

    # Generate report
    print("\nGenerating experiment report...")
    generate_report(summary, output_path, figures_dir)

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - evaluation_summary.json (metrics)")
    print(f"  - figures/ (visualizations)")
    print(f"  - experiment_report.md (full report)")

def generate_sample_predictions(model, config_path, output_dir, num_samples=20):
    """Generate visualization of sample predictions."""
    import yaml

    # Load dataset config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config['path'])
    test_images_dir = dataset_path / config['val']

    # Get random test images
    test_images = list(test_images_dir.glob('*.jpg'))
    if len(test_images) > num_samples:
        test_images = np.random.choice(test_images, num_samples, replace=False)

    # Create grid
    rows = 4
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(20, 16))
    axes = axes.flatten()

    class_names = config['names']
    colors = {0: (255, 107, 107), 1: (78, 205, 196), 2: (69, 183, 209),
              3: (255, 160, 122), 4: (152, 216, 200)}

    for idx, img_path in enumerate(test_images[:num_samples]):
        # Run prediction
        results = model.predict(str(img_path), conf=0.25, verbose=False)

        # Load image
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Draw predictions
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

                label = f"{class_names[cls]} {conf:.2f}"
                cv2.putText(img_rgb, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'Test Image {idx+1}', fontsize=10)

    # Hide unused subplots
    for idx in range(len(test_images), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sample Predictions on Test Set', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    predictions_path = output_dir / 'sample_predictions.png'
    plt.savefig(predictions_path, dpi=200, bbox_inches='tight')
    print(f"  ✓ Sample predictions: {predictions_path}")
    plt.close()

def generate_report(summary, output_dir, figures_dir):
    """Generate markdown report."""
    report_path = output_dir / 'experiment_report.md'

    with open(report_path, 'w') as f:
        f.write("# YOLO Shape Recognition - Experiment Report\n\n")
        f.write("## Overview\n\n")
        f.write("This report presents the results of training YOLOv8 to recognize synthetic and natural shape silhouettes.\n\n")

        f.write("## Overall Performance\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| mAP@0.5 | {summary['overall']['mAP@0.5']:.4f} |\n")
        f.write(f"| mAP@0.5:0.95 | {summary['overall']['mAP@0.5:0.95']:.4f} |\n")
        f.write(f"| Precision | {summary['overall']['precision']:.4f} |\n")
        f.write(f"| Recall | {summary['overall']['recall']:.4f} |\n\n")

        f.write("## Per-Category Results\n\n")
        f.write("| Category | Type | mAP@0.5 | mAP@0.5:0.95 |\n")
        f.write("|----------|------|---------|---------------|\n")

        category_types = {
            'cat1': 'Synthetic (unconstrained)',
            'cat2': 'Synthetic (variance matched)',
            'cat3': 'Synthetic (skew/kurtosis matched)',
            'cat4': 'Synthetic (all stats matched)',
            'cat5': 'Natural (animals)'
        }

        for cat, metrics in summary['per_category'].items():
            f.write(f"| {cat} | {category_types[cat]} | "
                   f"{metrics['mAP@0.5']:.4f} | {metrics['mAP@0.5:0.95']:.4f} |\n")

        f.write("\n## Key Findings\n\n")

        # Calculate synthetic vs natural
        synthetic_map = np.mean([summary['per_category'][f'cat{i}']['mAP@0.5']
                                for i in range(1, 5)])
        natural_map = summary['per_category']['cat5']['mAP@0.5']

        f.write(f"1. **Synthetic shapes average mAP**: {synthetic_map:.4f}\n")
        f.write(f"2. **Natural shapes mAP**: {natural_map:.4f}\n")
        f.write(f"3. **Difference**: {abs(synthetic_map - natural_map):.4f}\n\n")

        # Best and worst
        best_cat = max(summary['per_category'].items(),
                      key=lambda x: x[1]['mAP@0.5'])
        worst_cat = min(summary['per_category'].items(),
                       key=lambda x: x[1]['mAP@0.5'])

        f.write(f"4. **Best performing category**: {best_cat[0]} "
               f"(mAP: {best_cat[1]['mAP@0.5']:.4f})\n")
        f.write(f"5. **Worst performing category**: {worst_cat[0]} "
               f"(mAP: {worst_cat[1]['mAP@0.5']:.4f})\n\n")

        f.write("## Visualizations\n\n")
        f.write("### Performance by Category\n")
        f.write(f"![mAP by Category](figures/mAP_by_category.png)\n\n")
        f.write("### Synthetic vs Natural Comparison\n")
        f.write(f"![Synthetic vs Natural](figures/synthetic_vs_natural.png)\n\n")
        f.write("### Sample Predictions\n")
        f.write(f"![Sample Predictions](figures/sample_predictions.png)\n\n")

        f.write("## Conclusion\n\n")
        f.write("The trained YOLO model successfully detects and classifies shape silhouettes across all categories. ")

        if synthetic_map > natural_map:
            f.write(f"Synthetic shapes showed better detection performance ({synthetic_map:.4f}) ")
            f.write(f"compared to natural shapes ({natural_map:.4f}), suggesting that controlled curvature ")
            f.write("properties may be more easily learned by the model.\n")
        else:
            f.write(f"Natural shapes showed better detection performance ({natural_map:.4f}) ")
            f.write(f"compared to synthetic shapes ({synthetic_map:.4f}).\n")

    print(f"  ✓ Report: {report_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate YOLO shape recognition model')
    parser.add_argument('--weights', type=str,
                       default='runs/shapes/experiment_1/weights/best.pt',
                       help='Path to trained weights')
    parser.add_argument('--config', type=str,
                       default='config/shapes.yaml',
                       help='Path to dataset YAML config')
    parser.add_argument('--output-dir', type=str,
                       default='results',
                       help='Output directory for results')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold')

    args = parser.parse_args()

    # Get project directory
    PROJECT_DIR = Path(__file__).parent.parent

    weights_path = PROJECT_DIR / args.weights
    config_path = PROJECT_DIR / args.config
    output_dir = PROJECT_DIR / args.output_dir

    evaluate_model(
        weights_path=str(weights_path),
        config_path=str(config_path),
        output_dir=str(output_dir),
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
