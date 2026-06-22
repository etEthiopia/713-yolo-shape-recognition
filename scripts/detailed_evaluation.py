#!/usr/bin/env python3
"""
Detailed YOLO Model Evaluation Script

Evaluates a trained YOLO model with comprehensive per-image metrics.
Supports cross-experiment evaluation (e.g., train on natural scenes, test on line backgrounds).

Usage:
    python detailed_evaluation.py experiments/line_background_with_overlap_splitted
    python detailed_evaluation.py experiments/line_background_with_overlap_splitted --test-data experiments/natural_scenes_background_with_overlap_splitted
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from ultralytics import YOLO
from shapely.geometry import box
from shapely.ops import unary_union

# Category mapping (consistent across all experiments)
CLASS_MAPPING = {
    0: 'cat1',
    1: 'cat2',
    2: 'cat4',
    3: 'cat5'
}

CATEGORY_NAMES = {
    'cat1': 'Unconstrained',
    'cat2': 'Local (Var)',
    'cat4': 'Local (Matched)',
    'cat5': 'Natural'
}


def load_ground_truth(label_path: Path, overlap_dir: Path = None) -> Tuple[List[Dict], Dict]:
    """
    Load ground truth labels from YOLO format file and calculate symmetric overlap using bounding boxes.

    Args:
        label_path: Path to YOLO label file
        overlap_dir: Not used (kept for API compatibility)

    Returns:
        Tuple of (labels, image_overlap_stats)
    """
    if not label_path.exists():
        return [], {'has_overlap': False, 'overlap_ratio': 0.0}

    # Load labels
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                labels.append({
                    'class_id': class_id,
                    'category': CLASS_MAPPING[class_id],
                    'bbox': {
                        'x_center': x_center,
                        'y_center': y_center,
                        'width': width,
                        'height': height
                    }
                })

    # Calculate symmetric overlap ratios using bounding boxes (per-shape)
    overlap_ratios = calculate_shape_overlap_ratios(labels)

    # Add overlap ratios to labels
    for label, overlap_ratio in zip(labels, overlap_ratios):
        label['overlap_ratio'] = overlap_ratio
        label['visibility_ratio'] = 1.0 - min(overlap_ratio, 1.0)

    # Calculate image-level overlap statistics using Shapely (CORRECT METHOD)
    # Convert bboxes to Shapely polygons
    polygons = []
    sum_of_areas = 0.0

    for label in labels:
        bbox = label['bbox']
        x1 = bbox['x_center'] - bbox['width'] / 2
        y1 = bbox['y_center'] - bbox['height'] / 2
        x2 = bbox['x_center'] + bbox['width'] / 2
        y2 = bbox['y_center'] + bbox['height'] / 2

        poly = box(x1, y1, x2, y2)
        polygons.append(poly)
        sum_of_areas += poly.area

    if polygons:
        # Calculate true geometric union
        union_poly = unary_union(polygons)
        union_area = union_poly.area

        # Calculate shared area using pairwise intersections (CORRECT METHOD)
        # This finds the actual physical area where 2+ boxes overlap
        pairwise_intersections = []
        for i in range(len(polygons)):
            for j in range(i + 1, len(polygons)):
                inter = polygons[i].intersection(polygons[j])
                if not inter.is_empty:
                    pairwise_intersections.append(inter)

        if pairwise_intersections:
            # Union of all pairwise intersections = actual overlapping area
            intersection_poly = unary_union(pairwise_intersections)
            shared_area = intersection_poly.area
        else:
            shared_area = 0.0

        # Overlap ratio = shared_area / union_area
        # "What fraction of the footprint has overlapping shapes?"
        overlap_ratio = shared_area / union_area if union_area > 0 else 0.0
    else:
        union_area = 0.0
        sum_of_areas = 0.0
        shared_area = 0.0
        overlap_ratio = 0.0

    image_overlap_stats = {
        'has_overlap': shared_area > 0,
        'overlap_ratio': overlap_ratio,
        'num_overlapping_shapes': sum(1 for ratio in overlap_ratios if ratio > 0),
        'metadata_available': False,
        'method': 'shapely_pairwise_intersection',
        'union_area': union_area,
        'sum_of_areas': sum_of_areas,
        'shared_area': shared_area
    }

    return labels, image_overlap_stats


def calculate_intersection_area(box1: Dict, box2: Dict) -> float:
    """
    Calculate intersection area between two boxes in YOLO format (normalized coordinates).

    Args:
        box1, box2: Dicts with 'x_center', 'y_center', 'width', 'height' (all normalized 0-1)

    Returns:
        Intersection area (normalized, 0-1)
    """
    # Convert to corner coordinates
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2

    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2

    # Calculate intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    intersection = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    return intersection


def calculate_iou(box1: Dict, box2: Dict) -> float:
    """
    Calculate IoU between two boxes in YOLO format (normalized coordinates).

    Args:
        box1, box2: Dicts with 'x_center', 'y_center', 'width', 'height' (all normalized 0-1)

    Returns:
        IoU value between 0 and 1
    """
    intersection = calculate_intersection_area(box1, box2)

    # Calculate union
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_shape_overlap_ratios(shapes: List[Dict]) -> List[float]:
    """
    Calculate overlap ratio for each shape based on intersection with ALL other shapes.

    Args:
        shapes: List of shape dicts with 'bbox' key

    Returns:
        List of overlap ratios (one per shape)
    """
    n = len(shapes)
    overlap_ratios = []

    for i in range(n):
        shape = shapes[i]
        shape_area = shape['bbox']['width'] * shape['bbox']['height']

        if shape_area == 0:
            overlap_ratios.append(0.0)
            continue

        # Calculate total intersection area with all other shapes
        total_intersection = 0.0

        for j in range(n):
            if i == j:
                continue

            other_shape = shapes[j]
            intersection = calculate_intersection_area(shape['bbox'], other_shape['bbox'])
            total_intersection += intersection

        # Overlap ratio = total intersection area / shape area
        overlap_ratio = total_intersection / shape_area
        overlap_ratios.append(overlap_ratio)

    return overlap_ratios


def match_predictions_to_ground_truth(predictions: List[Dict],
                                     ground_truth: List[Dict],
                                     iou_threshold: float = 0.5) -> Dict:
    """
    Match predictions to ground truth boxes using Hungarian algorithm (greedy approximation).

    Returns:
        Dict with 'matches', 'false_positives', 'false_negatives'
    """
    matches = []
    false_positives = []
    false_negatives = list(range(len(ground_truth)))
    used_gt = set()

    # Sort predictions by confidence (highest first)
    sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1]['confidence'], reverse=True)

    for pred_idx, pred in sorted_preds:
        best_iou = 0
        best_gt_idx = -1

        # Find best matching ground truth
        for gt_idx in false_negatives:
            if gt_idx in used_gt:
                continue

            gt = ground_truth[gt_idx]

            # Only match same class
            if pred['class_id'] != gt['class_id']:
                continue

            iou = calculate_iou(pred['bbox'], gt['bbox'])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            # Match found
            matches.append({
                'prediction_idx': pred_idx,
                'ground_truth_idx': best_gt_idx,
                'iou': best_iou,
                'class_id': pred['class_id'],
                'confidence': pred['confidence'],
                'correct_class': True
            })
            used_gt.add(best_gt_idx)
            false_negatives.remove(best_gt_idx)
        else:
            # False positive
            false_positives.append({
                'prediction_idx': pred_idx,
                'class_id': pred['class_id'],
                'confidence': pred['confidence'],
                'bbox': pred['bbox']
            })

    return {
        'matches': matches,
        'false_positives': false_positives,
        'false_negatives': [{'ground_truth_idx': idx, 'class_id': ground_truth[idx]['class_id']}
                           for idx in false_negatives]
    }


def evaluate_image(model: YOLO,
                  image_path: Path,
                  label_path: Path,
                  overlap_dir: Path,
                  conf_threshold: float = 0.5,
                  iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate model on a single image with detailed metrics including overlap statistics.

    Args:
        conf_threshold: Confidence threshold (YOLO confidence = IoU × p(object))
                       Default 0.5 filters predictions to show only confident detections
        iou_threshold: IoU threshold for matching predictions to ground truth
                      Default 0.5 means 50% bounding box overlap required for True Positive

    Returns:
        Dict with comprehensive per-image results
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    img_height, img_width = image.shape[:2]

    # Load ground truth with overlap metadata
    ground_truth, image_overlap_stats = load_ground_truth(label_path, overlap_dir)

    # Run inference
    results = model(image, conf=conf_threshold, verbose=False)[0]

    # Extract predictions
    predictions = []
    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xywhn.cpu().numpy()  # Normalized xywh
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confidences = results.boxes.conf.cpu().numpy()

        for box, class_id, conf in zip(boxes, classes, confidences):
            predictions.append({
                'class_id': int(class_id),
                'category': CLASS_MAPPING[int(class_id)],
                'confidence': float(conf),
                'bbox': {
                    'x_center': float(box[0]),
                    'y_center': float(box[1]),
                    'width': float(box[2]),
                    'height': float(box[3])
                }
            })

    # Match predictions to ground truth
    matching = match_predictions_to_ground_truth(predictions, ground_truth, iou_threshold)

    # Add overlap ratios to predictions based on matched ground truth
    for pred_idx, pred in enumerate(predictions):
        pred['overlap_ratio'] = 0.0  # Default for unmatched predictions

        # Find if this prediction matched a ground truth
        for match in matching['matches']:
            if match['prediction_idx'] == pred_idx:
                gt_idx = match['ground_truth_idx']
                if gt_idx < len(ground_truth):
                    pred['overlap_ratio'] = ground_truth[gt_idx].get('overlap_ratio', 0.0)
                break

    # Calculate per-image metrics
    num_gt = len(ground_truth)
    num_pred = len(predictions)
    num_tp = len(matching['matches'])
    num_fp = len(matching['false_positives'])
    num_fn = len(matching['false_negatives'])

    precision = num_tp / num_pred if num_pred > 0 else 0.0
    recall = num_tp / num_gt if num_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Per-category counts
    gt_per_category = {}
    pred_per_category = {}
    tp_per_category = {}

    for cat_id in CLASS_MAPPING.keys():
        cat_name = CLASS_MAPPING[cat_id]
        gt_per_category[cat_name] = sum(1 for gt in ground_truth if gt['class_id'] == cat_id)
        pred_per_category[cat_name] = sum(1 for pred in predictions if pred['class_id'] == cat_id)
        tp_per_category[cat_name] = sum(1 for match in matching['matches'] if match['class_id'] == cat_id)

    return {
        'image_path': str(image_path.name),
        'image_size': {'width': img_width, 'height': img_height},
        'overlap_stats': image_overlap_stats,
        'ground_truth': ground_truth,
        'predictions': predictions,
        'matching': matching,
        'metrics': {
            'num_ground_truth': num_gt,
            'num_predictions': num_pred,
            'true_positives': num_tp,
            'false_positives': num_fp,
            'false_negatives': num_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_iou': np.mean([m['iou'] for m in matching['matches']]) if matching['matches'] else 0.0
        },
        'per_category': {
            'ground_truth_counts': gt_per_category,
            'prediction_counts': pred_per_category,
            'true_positive_counts': tp_per_category
        }
    }


def aggregate_metrics(image_results: List[Dict], iou_thresholds: List[float] = None) -> Dict:
    """
    Aggregate metrics across all images.

    Args:
        image_results: List of per-image evaluation results
        iou_thresholds: List of IoU thresholds for mAP calculation

    Returns:
        Dict with aggregated metrics
    """
    if iou_thresholds is None:
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Overall counts
    total_gt = sum(r['metrics']['num_ground_truth'] for r in image_results)
    total_pred = sum(r['metrics']['num_predictions'] for r in image_results)
    total_tp = sum(r['metrics']['true_positives'] for r in image_results)
    total_fp = sum(r['metrics']['false_positives'] for r in image_results)
    total_fn = sum(r['metrics']['false_negatives'] for r in image_results)

    # Overall metrics
    overall_precision = total_tp / total_pred if total_pred > 0 else 0.0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0

    # Per-category aggregation
    categories = list(CLASS_MAPPING.values())
    per_category = {}

    for cat in categories:
        cat_gt = sum(r['per_category']['ground_truth_counts'][cat] for r in image_results)
        cat_pred = sum(r['per_category']['prediction_counts'][cat] for r in image_results)
        cat_tp = sum(r['per_category']['true_positive_counts'][cat] for r in image_results)
        cat_fp = cat_pred - cat_tp
        cat_fn = cat_gt - cat_tp

        cat_precision = cat_tp / cat_pred if cat_pred > 0 else 0.0
        cat_recall = cat_tp / cat_gt if cat_gt > 0 else 0.0
        cat_f1 = 2 * cat_precision * cat_recall / (cat_precision + cat_recall) if (cat_precision + cat_recall) > 0 else 0.0

        per_category[cat] = {
            'ground_truth': cat_gt,
            'predictions': cat_pred,
            'true_positives': cat_tp,
            'false_positives': cat_fp,
            'false_negatives': cat_fn,
            'precision': cat_precision,
            'recall': cat_recall,
            'f1_score': cat_f1
        }

    # Confusion matrix
    confusion_matrix = np.zeros((len(categories), len(categories)), dtype=int)

    for result in image_results:
        for match in result['matching']['matches']:
            gt_class = match['class_id']
            pred_class = match['class_id']  # Same since we only match same class
            confusion_matrix[gt_class, pred_class] += 1

        # False positives
        for fp in result['matching']['false_positives']:
            pred_class = fp['class_id']
            # Count as misclassification (we don't know what it should have been)

    # Per-image statistics
    precisions = [r['metrics']['precision'] for r in image_results]
    recalls = [r['metrics']['recall'] for r in image_results]
    f1_scores = [r['metrics']['f1_score'] for r in image_results]

    # Overlap statistics
    num_images_with_overlap = sum(1 for r in image_results if r.get('overlap_stats', {}).get('has_overlap', False))
    num_images_without_overlap = len(image_results) - num_images_with_overlap

    # Performance on overlapping vs non-overlapping images
    overlap_metrics = {'with_overlap': {}, 'without_overlap': {}}

    for overlap_type, has_overlap in [('with_overlap', True), ('without_overlap', False)]:
        subset = [r for r in image_results if r.get('overlap_stats', {}).get('has_overlap', False) == has_overlap]

        if subset:
            subset_tp = sum(r['metrics']['true_positives'] for r in subset)
            subset_gt = sum(r['metrics']['num_ground_truth'] for r in subset)
            subset_pred = sum(r['metrics']['num_predictions'] for r in subset)

            subset_prec = subset_tp / subset_pred if subset_pred > 0 else 0.0
            subset_rec = subset_tp / subset_gt if subset_gt > 0 else 0.0
            subset_f1 = 2 * subset_prec * subset_rec / (subset_prec + subset_rec) if (subset_prec + subset_rec) > 0 else 0.0

            overlap_metrics[overlap_type] = {
                'num_images': len(subset),
                'precision': subset_prec,
                'recall': subset_rec,
                'f1_score': subset_f1
            }
        else:
            overlap_metrics[overlap_type] = {'num_images': 0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

    return {
        'overall': {
            'total_images': len(image_results),
            'total_ground_truth': total_gt,
            'total_predictions': total_pred,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'images_with_overlap': num_images_with_overlap,
            'images_without_overlap': num_images_without_overlap
        },
        'per_category': per_category,
        'overlap_analysis': overlap_metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'image_statistics': {
            'precision': {
                'mean': float(np.mean(precisions)),
                'std': float(np.std(precisions)),
                'min': float(np.min(precisions)),
                'max': float(np.max(precisions)),
                'median': float(np.median(precisions))
            },
            'recall': {
                'mean': float(np.mean(recalls)),
                'std': float(np.std(recalls)),
                'min': float(np.min(recalls)),
                'max': float(np.max(recalls)),
                'median': float(np.median(recalls))
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'std': float(np.std(f1_scores)),
                'min': float(np.min(f1_scores)),
                'max': float(np.max(f1_scores)),
                'median': float(np.median(f1_scores))
            }
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Detailed YOLO evaluation with cross-experiment support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate model on its own test data
  python detailed_evaluation.py experiments/line_background_with_overlap_splitted

  # Cross-experiment evaluation (train on natural, test on lines)
  python detailed_evaluation.py experiments/natural_scenes_background_with_overlap_splitted \\
      --test-data experiments/line_background_with_overlap_splitted
        """
    )

    parser.add_argument('model_experiment', type=str,
                       help='Path to trained experiment (e.g., experiments/line_background_with_overlap_splitted)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to experiment containing test data (default: same as model_experiment)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for matching (default: 0.5)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for results (default: model_experiment/detailed_evaluation)')

    args = parser.parse_args()

    # Paths
    model_exp_path = Path(args.model_experiment)
    test_exp_path = Path(args.test_data) if args.test_data else model_exp_path

    if not model_exp_path.exists():
        print(f"Error: Model experiment path does not exist: {model_exp_path}")
        sys.exit(1)

    if not test_exp_path.exists():
        print(f"Error: Test data experiment path does not exist: {test_exp_path}")
        sys.exit(1)

    # Find model weights
    weights_path = model_exp_path / 'training' / 'run_1' / 'weights' / 'best.pt'
    if not weights_path.exists():
        print(f"Error: Model weights not found at: {weights_path}")
        print("Train the model first using train_model.py")
        sys.exit(1)

    # Test data paths
    test_img_dir = test_exp_path / 'dataset' / 'images' / 'test'
    test_lbl_dir = test_exp_path / 'dataset' / 'labels' / 'test'
    test_overlap_dir = test_exp_path / 'dataset' / 'overlap' / 'test'

    if not test_img_dir.exists():
        print(f"Error: Test images not found at: {test_img_dir}")
        sys.exit(1)

    if not test_overlap_dir.exists():
        print(f"Warning: Overlap metadata not found at: {test_overlap_dir}")
        print("Overlap statistics will not be available")
        print()

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_exp_path / 'detailed_evaluation'
        if args.test_data:
            # Add suffix for cross-experiment evaluation
            test_name = test_exp_path.name
            output_dir = model_exp_path / f'detailed_evaluation_on_{test_name}'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Detailed YOLO Evaluation")
    print("=" * 70)
    print(f"\nModel: {model_exp_path.name}")
    print(f"Weights: {weights_path}")
    print(f"Test data: {test_exp_path.name}")
    print(f"Test images: {test_img_dir}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Output: {output_dir}")
    print()

    # Load model
    print("Loading model...")
    model = YOLO(str(weights_path))
    print("✓ Model loaded")
    print()

    # Get test images
    test_images = sorted(test_img_dir.glob('*.jpg'))
    print(f"Found {len(test_images)} test images")
    print()

    # Evaluate each image
    print("Evaluating images...")
    image_results = []

    for i, img_path in enumerate(test_images):
        lbl_path = test_lbl_dir / img_path.name.replace('.jpg', '.txt')

        result = evaluate_image(model, img_path, lbl_path, test_overlap_dir, args.conf, args.iou)

        if result is not None:
            image_results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test_images)} images evaluated")

    print(f"✓ Evaluated {len(image_results)} images")
    print()

    # Aggregate metrics
    print("Computing aggregate metrics...")
    aggregate = aggregate_metrics(image_results)
    print("✓ Metrics computed")
    print()

    # Save results
    print("Saving results...")

    # Per-image results
    per_image_file = output_dir / 'per_image_results.json'
    with open(per_image_file, 'w') as f:
        json.dump(image_results, f, indent=2)
    print(f"  ✓ Per-image results: {per_image_file}")

    # Aggregate metrics
    aggregate_file = output_dir / 'aggregate_metrics.json'
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)
    print(f"  ✓ Aggregate metrics: {aggregate_file}")

    # Evaluation summary
    summary = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'model_experiment': str(model_exp_path.name),
            'test_experiment': str(test_exp_path.name),
            'model_weights': str(weights_path),
            'test_images_dir': str(test_img_dir),
            'num_test_images': len(image_results),
            'conf_threshold': args.conf,
            'iou_threshold': args.iou,
            'is_cross_experiment': args.test_data is not None
        },
        'metrics': aggregate
    }

    summary_file = output_dir / 'evaluation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Summary: {summary_file}")
    print()

    # Print summary
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"\nOverall Metrics:")
    print(f"  Precision: {aggregate['overall']['precision']:.4f}")
    print(f"  Recall:    {aggregate['overall']['recall']:.4f}")
    print(f"  F1 Score:  {aggregate['overall']['f1_score']:.4f}")
    print()
    print(f"Per-Category Precision:")
    for cat in ['cat1', 'cat2', 'cat4', 'cat5']:
        prec = aggregate['per_category'][cat]['precision']
        print(f"  {CATEGORY_NAMES[cat]:20s}: {prec:.4f}")
    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
