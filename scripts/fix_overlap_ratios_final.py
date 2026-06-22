#!/usr/bin/env python3
"""
Fix Overlap Ratios Using Shapely (CORRECT METHOD)

Uses geometric union to correctly calculate overlapping regions without double-counting.

Usage:
    python fix_overlap_ratios_final.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import shutil
from shapely.geometry import box
from shapely.ops import unary_union


def calculate_bbox_intersection(box1: Dict, box2: Dict) -> float:
    """Calculate intersection area between two bounding boxes (for per-shape ratios)."""
    x1_min = box1['x_center'] - box1['width'] / 2
    y1_min = box1['y_center'] - box1['height'] / 2
    x1_max = box1['x_center'] + box1['width'] / 2
    y1_max = box1['y_center'] + box1['height'] / 2

    x2_min = box2['x_center'] - box2['width'] / 2
    y2_min = box2['y_center'] - box2['height'] / 2
    x2_max = box2['x_center'] + box2['width'] / 2
    y2_max = box2['y_center'] + box2['height'] / 2

    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max <= x_inter_min or y_inter_max <= y_inter_min:
        return 0.0

    return (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)


def calculate_symmetric_overlap_ratios(labels: List[Dict]) -> List[float]:
    """
    Calculate overlap ratio for each shape using symmetric bounding box intersection.
    """
    n = len(labels)
    overlap_ratios = []

    for i in range(n):
        shape = labels[i]
        shape_area = shape['bbox']['width'] * shape['bbox']['height']

        if shape_area == 0:
            overlap_ratios.append(0.0)
            continue

        # Calculate total intersection area with all other shapes
        total_intersection = 0.0

        for j in range(n):
            if i == j:
                continue

            other_shape = labels[j]
            intersection = calculate_bbox_intersection(shape['bbox'], other_shape['bbox'])
            total_intersection += intersection

        overlap_ratio = total_intersection / shape_area
        overlap_ratios.append(overlap_ratio)

    return overlap_ratios


def calculate_union_using_shapely(labels: List[Dict]) -> tuple:
    """
    Calculate true geometric union using Shapely.

    Returns:
        (union_area, sum_of_areas, shared_area)
    """
    if not labels:
        return 0.0, 0.0, 0.0

    polygons = []
    sum_of_areas = 0.0

    for label in labels:
        bbox = label['bbox']
        # Convert YOLO center format to corner coordinates
        x1 = bbox['x_center'] - bbox['width'] / 2
        y1 = bbox['y_center'] - bbox['height'] / 2
        x2 = bbox['x_center'] + bbox['width'] / 2
        y2 = bbox['y_center'] + bbox['height'] / 2

        # Create shapely box
        b = box(x1, y1, x2, y2)
        polygons.append(b)
        sum_of_areas += b.area

    # Calculate true geometric union (no double-counting!)
    union_polygon = unary_union(polygons)
    union_area = union_polygon.area

    # Shared space = what was counted multiple times
    shared_area = sum_of_areas - union_area

    return union_area, sum_of_areas, shared_area


def fix_image_overlap_stats(image_result: Dict) -> Dict:
    """Fix overlap statistics using Shapely for correct geometric union."""
    ground_truth = image_result.get('ground_truth', [])

    if not ground_truth:
        return image_result

    # Calculate per-shape overlap ratios (still using pairwise intersections)
    overlap_ratios = calculate_symmetric_overlap_ratios(ground_truth)

    # Update ground truth with overlap ratios
    for gt, overlap_ratio in zip(ground_truth, overlap_ratios):
        gt['overlap_ratio'] = overlap_ratio
        gt['visibility_ratio'] = 1.0 - min(overlap_ratio, 1.0)

    # Calculate image-level overlap using Shapely
    union_area, sum_of_areas, shared_area = calculate_union_using_shapely(ground_truth)

    # Overlap ratio = shared_area / sum_of_areas (always ≤ 1.0)
    overlap_ratio = shared_area / sum_of_areas if sum_of_areas > 0 else 0.0

    image_result['overlap_stats'] = {
        'has_overlap': shared_area > 0,
        'overlap_ratio': overlap_ratio,
        'num_overlapping_shapes': sum(1 for ratio in overlap_ratios if ratio > 0),
        'metadata_available': False,
        'method': 'shapely_union',
        'union_area': union_area,
        'sum_of_areas': sum_of_areas,
        'shared_area': shared_area
    }

    # Update predictions
    predictions = image_result.get('predictions', [])
    matches = image_result.get('matching', {}).get('matches', [])

    for pred in predictions:
        pred['overlap_ratio'] = 0.0

    for match in matches:
        pred_idx = match['prediction_idx']
        gt_idx = match['ground_truth_idx']
        if pred_idx < len(predictions) and gt_idx < len(ground_truth):
            predictions[pred_idx]['overlap_ratio'] = ground_truth[gt_idx]['overlap_ratio']

    return image_result


def main():
    parser = argparse.ArgumentParser(
        description='Fix overlap ratios using Shapely geometric union'
    )
    parser.add_argument('eval_dir', type=str,
                       help='Path to detailed evaluation directory')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup before fixing')

    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    per_image_file = eval_dir / 'per_image_results.json'

    if not per_image_file.exists():
        print(f"Error: {per_image_file} not found")
        return

    print("="*70)
    print("Fix Overlap Ratios (Shapely Geometric Union - CORRECT)")
    print("="*70)
    print()

    # Backup
    if args.backup:
        backup_file = eval_dir / 'per_image_results.json.backup_final'
        shutil.copy(per_image_file, backup_file)
        print(f"Created backup: {backup_file.name}")
        print()

    # Load data
    print(f"Loading: {per_image_file.name}")
    with open(per_image_file, 'r') as f:
        per_image_data = json.load(f)
    print(f"  ✓ Loaded {len(per_image_data)} images")
    print()

    # Fix each image
    print("Fixing overlap ratios with Shapely geometric union...")
    for i, image_result in enumerate(per_image_data):
        per_image_data[i] = fix_image_overlap_stats(image_result)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(per_image_data)} images...")

    print(f"  ✓ Fixed {len(per_image_data)} images")
    print()

    # Save
    print(f"Saving corrected data...")
    with open(per_image_file, 'w') as f:
        json.dump(per_image_data, f, indent=2)
    print("  ✓ Saved")
    print()

    # Statistics
    print("="*70)
    print("Verification")
    print("="*70)
    print()

    # Check test_0053 specifically
    test_0053 = next((img for img in per_image_data if 'test_0053' in img['image_path']), None)
    if test_0053:
        stats = test_0053['overlap_stats']
        print("test_0053.jpg (6 shapes, 5 overlapping):")
        print(f"  Sum of areas: {stats.get('sum_of_areas', 0):.6f}")
        print(f"  Union area: {stats.get('union_area', 0):.6f}")
        print(f"  Shared area: {stats.get('shared_area', 0):.6f}")
        print(f"  Overlap ratio: {stats['overlap_ratio']:.4f} ({stats['overlap_ratio']*100:.1f}%)")
        print()

    # Overall statistics
    overlap_ratios = []
    for img in per_image_data:
        overlap_stats = img.get('overlap_stats', {})
        if overlap_stats.get('has_overlap'):
            overlap_ratios.append(overlap_stats.get('overlap_ratio', 0.0))

    if overlap_ratios:
        import numpy as np
        print(f"Images with overlap: {len(overlap_ratios)}/{len(per_image_data)}")
        print(f"Overlap ratio range: {min(overlap_ratios):.4f} - {max(overlap_ratios):.4f}")
        print(f"Mean overlap ratio: {np.mean(overlap_ratios):.4f} ± {np.std(overlap_ratios):.4f}")
        print(f"All ratios ≤ 1.0: {'✓ YES' if all(r <= 1.0 for r in overlap_ratios) else '✗ NO'}")

    print()
    print("Done! Overlap ratios now use correct geometric union (no double-counting).")
    print()


if __name__ == '__main__':
    main()
