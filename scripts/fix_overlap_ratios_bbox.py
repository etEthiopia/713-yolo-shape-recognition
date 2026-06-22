#!/usr/bin/env python3
"""
Fix Overlap Ratios Using Bounding Box Intersection (Symmetric)

Correctly calculates overlap ratios based on bounding box intersection.
Both shapes in an overlapping pair share the same intersection area.

Usage:
    python fix_overlap_ratios_bbox.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict
import shutil


def calculate_bbox_intersection(box1: Dict, box2: Dict) -> float:
    """Calculate intersection area between two bounding boxes (normalized coordinates)."""
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

    Each shape's overlap = sum of intersections with all other shapes / shape's bbox area
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

        # Overlap ratio = total intersection area / shape area
        overlap_ratio = total_intersection / shape_area
        overlap_ratios.append(overlap_ratio)

    return overlap_ratios


def fix_image_overlap_stats(image_result: Dict) -> Dict:
    """Fix overlap statistics using symmetric bounding box intersection."""
    ground_truth = image_result.get('ground_truth', [])

    if not ground_truth:
        return image_result

    # Calculate symmetric overlap ratios
    overlap_ratios = calculate_symmetric_overlap_ratios(ground_truth)

    # Update ground truth with overlap ratios
    for gt, overlap_ratio in zip(ground_truth, overlap_ratios):
        gt['overlap_ratio'] = overlap_ratio
        gt['visibility_ratio'] = 1.0 - min(overlap_ratio, 1.0)

    # Calculate image-level overlap statistics
    total_area = sum(gt['bbox']['width'] * gt['bbox']['height'] for gt in ground_truth)
    total_intersection_area = 0.0

    # Calculate total intersection area (avoiding double counting)
    for i in range(len(ground_truth)):
        for j in range(i + 1, len(ground_truth)):
            intersection = calculate_bbox_intersection(
                ground_truth[i]['bbox'],
                ground_truth[j]['bbox']
            )
            total_intersection_area += intersection

    # Image overlap ratio = intersection / total_area
    # Cap at 1.0 because pairwise intersections can triple-count overlapping regions
    overlap_ratio = total_intersection_area / total_area if total_area > 0 else 0.0
    overlap_ratio = min(overlap_ratio, 1.0)  # Cap at 100%

    image_result['overlap_stats'] = {
        'has_overlap': total_intersection_area > 0,
        'overlap_ratio': overlap_ratio,
        'num_overlapping_shapes': sum(1 for ratio in overlap_ratios if ratio > 0),
        'metadata_available': False,  # Using bbox calculation, not pixel metadata
        'method': 'bounding_box',
        'total_area': total_area,
        'total_intersection_area': total_intersection_area
    }

    # Update predictions with matched ground truth overlap ratios
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
        description='Fix overlap ratios using symmetric bounding box intersection'
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
    print("Fix Overlap Ratios (Symmetric Bounding Box Intersection)")
    print("="*70)
    print()

    # Backup
    if args.backup:
        backup_file = eval_dir / 'per_image_results.json.backup_v2'
        if (eval_dir / 'per_image_results.json.backup').exists():
            shutil.copy(eval_dir / 'per_image_results.json.backup', backup_file)
            print(f"Saved old backup as: {backup_file.name}")

        backup_file = eval_dir / 'per_image_results.json.backup'
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
    print("Fixing overlap ratios with symmetric bounding box intersection...")
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

    # Check test_0000 specifically
    test_0000 = next((img for img in per_image_data if 'test_0000' in img['image_path']), None)
    if test_0000:
        print("test_0000.jpg (2 shapes with bounding box intersection):")
        for i, gt in enumerate(test_0000['ground_truth']):
            print(f"  Shape {i} ({gt['category']}): {gt['overlap_ratio']:.4f} ({gt['overlap_ratio']*100:.1f}%)")
        print(f"  Image overlap: {test_0000['overlap_stats']['overlap_ratio']:.4f}")
        print()

    # Check overall statistics
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

        if not all(r <= 1.0 for r in overlap_ratios):
            max_ratio = max(overlap_ratios)
            print(f"  Warning: Max ratio is {max_ratio:.4f}")
            print(f"  This can happen when bbox intersections exceed individual bbox areas")

    print()
    print("Done! Overlap ratios now use symmetric bounding box intersection.")
    print()


if __name__ == '__main__':
    main()
