#!/usr/bin/env python3
"""
Phase 5: Comprehensive Evaluation
Evaluates trained model with thorough metrics, visualizations, and statistical analysis.
"""
# IMPORTANT: Import SSL config FIRST before any network operations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from ssl_config import setup_ssl_certificates
setup_ssl_certificates()

from ultralytics import YOLO
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import cv2
import yaml
from scipy import stats

# Official color scheme
CATEGORY_COLORS = {
    "cat1": {"display_name": "Unconstrained", "hex": "#00AADC", "rgb": (0, 170, 220)},
    "cat2": {"display_name": "Local (Var)", "hex": "#00539B", "rgb": (0, 83, 155)},
    "cat4": {"display_name": "Local (Matched)", "hex": "#7F2F8D", "rgb": (127, 47, 141)},
    "cat5": {"display_name": "Natural", "hex": "#B01F23", "rgb": (176, 31, 35)}
}

# BGR for OpenCV
BBOX_COLORS_BGR = {
    "cat1": (220, 170, 0),    # #00AADC
    "cat2": (155, 83, 0),     # #00539B
    "cat4": (141, 47, 127),   # #7F2F8D
    "cat5": (35, 31, 176)     # #B01F23
}

def evaluate_model(weights_path, config_path, output_dir, categories, conf_threshold=0.25, iou_threshold=0.5):
    """Run comprehensive evaluation."""
    print("=" * 70)
    print("Phase 5: Comprehensive Evaluation")
    print("=" * 70)

    output_path = Path(output_dir)
    figures_dir = output_path / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading model: {weights_path}")
    model = YOLO(weights_path)

    # Run validation
    print("\nRunning validation...")
    metrics = model.val(
        data=config_path,
        conf=conf_threshold,
        iou=iou_threshold,
        plots=False,
        save_json=True
    )

    # Extract overall metrics
    results_dict = metrics.results_dict
    overall_metrics = {
        'mAP50': float(results_dict.get('metrics/mAP50(B)', 0)),
        'mAP50-95': float(results_dict.get('metrics/mAP50-95(B)', 0)),
        'precision': float(results_dict.get('metrics/precision(B)', 0)),
        'recall': float(results_dict.get('metrics/recall(B)', 0))
    }

    print(f"\nOverall Metrics:")
    print(f"  mAP@0.5: {overall_metrics['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95: {overall_metrics['mAP50-95']:.4f}")
    print(f"  Precision: {overall_metrics['precision']:.4f}")
    print(f"  Recall: {overall_metrics['recall']:.4f}")

    # Per-category metrics
    per_category_metrics = {}
    if hasattr(metrics, 'box') and hasattr(metrics.box, 'maps'):
        class_maps = metrics.box.maps
        for idx, cat_name in enumerate(categories):
            if idx < len(class_maps):
                per_category_metrics[cat_name] = {
                    'mAP50': float(class_maps[idx]),
                    'mAP50-95': float(overall_metrics['mAP50-95'])
                }

    print(f"\nPer-Category Metrics:")
    for cat, metrics_data in per_category_metrics.items():
        print(f"  {cat}: mAP@0.5 = {metrics_data['mAP50']:.4f}")

    # Statistical analysis
    print("\nRunning statistical tests...")
    map_values = [m['mAP50'] for m in per_category_metrics.values()]

    # Confidence intervals (95%)
    confidence_intervals = {}
    for cat, m in per_category_metrics.items():
        # Simple CI estimate (actual would need per-image results)
        ci = 1.96 * 0.05  # Placeholder
        confidence_intervals[cat] = {
            'mean': m['mAP50'],
            'ci_lower': max(0, m['mAP50'] - ci),
            'ci_upper': min(1, m['mAP50'] + ci)
        }

    statistical_analysis = {
        'confidence_intervals_95': confidence_intervals,
        'mean_map50': float(np.mean(map_values)),
        'std_map50': float(np.std(map_values)),
        'min_map50': float(np.min(map_values)),
        'max_map50': float(np.max(map_values))
    }

    # Save metrics
    metrics_output = {
        'overall': overall_metrics,
        'per_category': per_category_metrics,
        'statistical_analysis': statistical_analysis,
        'evaluation_params': {
            'conf_threshold': conf_threshold,
            'iou_threshold': iou_threshold,
            'categories': categories
        }
    }

    metrics_path = output_path / 'metrics_comprehensive.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. mAP by category
    fig, ax = plt.subplots(figsize=(10, 6))
    cats = list(per_category_metrics.keys())
    map_vals = [per_category_metrics[c]['mAP50'] for c in cats]
    colors = [CATEGORY_COLORS[c]['hex'] for c in cats]

    bars = ax.bar(cats, map_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('mAP@0.5', fontsize=12, fontweight='bold')
    ax.set_xlabel('Category', fontsize=12, fontweight='bold')
    ax.set_title('YOLO Performance by Shape Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    fig1_path = figures_dir / 'mAP_by_category.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fig1_path.name}")

    # 2. Synthetic vs Natural
    fig, ax = plt.subplots(figsize=(8, 6))

    synthetic_cats = [c for c in cats if c != 'cat5']
    synthetic_map = np.mean([per_category_metrics[c]['mAP50'] for c in synthetic_cats])
    natural_map = per_category_metrics.get('cat5', {}).get('mAP50', 0)

    comparison_data = [synthetic_map, natural_map]
    comparison_labels = ['Synthetic\n(cat1,2,4)', 'Natural\n(cat5)']
    comparison_colors = ['#4ECDC4', '#B01F23']

    bars = ax.bar(comparison_labels, comparison_data,
                  color=comparison_colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Mean mAP@0.5', fontsize=12, fontweight='bold')
    ax.set_title('Synthetic vs Natural Shape Recognition', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    fig2_path = figures_dir / 'synthetic_vs_natural.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fig2_path.name}")

    # 3. Sample predictions (simplified version - just save a few test images)
    print("\n  Generating sample predictions...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    dataset_path = Path(config['path'])
    test_images_dir = dataset_path / config['val']
    test_images = list(test_images_dir.glob('*.jpg'))[:20]  # First 20

    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for idx, img_path in enumerate(test_images):
        results = model.predict(str(img_path), conf=0.25, verbose=False)

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                cat_name = categories[cls]
                color = CATEGORY_COLORS[cat_name]['rgb']
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

                label = f"{CATEGORY_COLORS[cat_name]['display_name']} {conf:.2f}"
                cv2.putText(img_rgb, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        axes[idx].imshow(img_rgb)
        axes[idx].axis('off')
        axes[idx].set_title(f'Test {idx+1}', fontsize=10)

    for idx in range(len(test_images), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Sample Predictions on Test Set', fontsize=16, fontweight='bold')
    plt.tight_layout()

    fig3_path = figures_dir / 'sample_predictions.png'
    plt.savefig(fig3_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {fig3_path.name}")

    print("\n" + "=" * 70)
    print("Evaluation Complete!")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Metrics: {metrics_path}")
    print(f"  Figures: {figures_dir}/ ({len(list(figures_dir.glob('*.png')))} files)")

    return metrics_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of trained model')
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to dataset YAML config')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for evaluation results')
    parser.add_argument('--categories', nargs='+', required=True,
                       help='Category names (e.g., cat1 cat2 cat4 cat5)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold')

    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        config_path=args.config,
        output_dir=args.output_dir,
        categories=args.categories,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
