#!/usr/bin/env python3
"""
Fix Overlap Ratios V2 - Using Symmetric Intersection

Correctly calculates overlap ratios based on INTERSECTION, not occlusion.
Both shapes in an overlapping pair have the same intersection area.

For test_0000 with shapes A and B sharing 8,382 pixels:
  Shape A overlap = 8,382 / area_A
  Shape B overlap = 8,382 / area_B
  (Different ratios because different sizes, same intersection)

Usage:
    python fix_overlap_ratios_v2.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import shutil


def calculate_pairwise_intersections_from_metadata(overlap_metadata: Dict) -> Dict[Tuple[int, int], float]:
    """
    Calculate intersection area for each pair of shapes from overlap metadata.

    Returns:
        Dict mapping (shape_i, shape_j) to intersection area in pixels
    """
    shapes = overlap_metadata.get('shapes', [])
    intersections = {}

    # For each shape that overlaps with others
    for i, shape in enumerate(shapes):
        overlaps_with = shape.get('overlaps_with', [])

        for j in overlaps_with:
            # Create canonical pair (smaller index first)
            pair = tuple(sorted([i, j]))

            if pair not in intersections:
                # The intersection equals the occluded area of the shape that's below
                # In metadata, shape i lists j in 'overlaps_with' means i is BELOW j
                # So shape i's occluded_area is the intersection
                intersections[pair] = float(shape['occluded_area'])

    return intersections


def calculate_per_shape_overlap_ratios_symmetric(overlap_metadata: Dict) -> List[float]:
    """
    Calculate symmetric overlap ratio for each shape.

    Each shape's overlap = sum of intersections with all other shapes / shape's total area
    """
    shapes = overlap_metadata.get('shapes', [])
    pairwise_intersections = calculate_pairwise_intersections_from_metadata(overlap_metadata)

    overlap_ratios = []

    for i, shape in enumerate(shapes):
        total_area = shape['total_area']

        if total_area == 0:
            overlap_ratios.append(0.0)
            continue

        # Sum intersections with all other shapes
        total_intersection = 0.0

        for j in range(len(shapes)):
            if i == j:
                continue

            pair = tuple(sorted([i, j]))
            intersection = pairwise_intersections.get(pair, 0.0)
            total_intersection += intersection

        overlap_ratio = total_intersection / total_area
        overlap_ratios.append(overlap_ratio)

    return overlap_ratios


def fix_image_overlap_stats(image_result: Dict, overlap_dir: Path, split: str) -> Dict:
    """
    Fix overlap statistics using symmetric intersection calculation.
    """
    image_name = image_result.get('image_path', '')
    base_name = Path(image_name).stem
    overlap_file = overlap_dir / split / f"{base_name}.json"

    if not overlap_file.exists():
        print(f"  Warning: No overlap metadata for {image_name}")
        return image_result

    # Load overlap metadata
    with open(overlap_file, 'r') as f:
        overlap_metadata = json.load(f)

    # Calculate symmetric overlap ratios
    overlap_ratios = calculate_per_shape_overlap_ratios_symmetric(overlap_metadata)

    # Update ground truth
    ground_truth = image_result.get('ground_truth', [])
    for i, gt in enumerate(ground_truth):
        if i < len(overlap_ratios):
            gt['overlap_ratio'] = overlap_ratios[i]
            gt['visibility_ratio'] = 1.0 - min(overlap_ratios[i], 1.0)

    # Image-level stats (same as before - correct)
    total_intersection = overlap_metadata.get('statistics', {}).get('total_overlap_area', 0)
    sum_of_areas = sum(s['total_area'] for s in overlap_metadata.get('shapes', []))
    union_area = sum_of_areas - total_intersection

    image_result['overlap_stats'] = {
        'has_overlap': total_intersection > 0,
        'overlap_ratio': total_intersection / union_area if union_area > 0 else 0.0,
        'num_overlapping_shapes': sum(1 for r in overlap_ratios if r > 0),
        'metadata_available': True,
        'union_area': union_area,
        'total_intersection_area': total_intersection
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
        description='Fix overlap ratios using symmetric intersection'
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

    # Find overlap directory
    experiment_dir = eval_dir.parent
    overlap_dir = experiment_dir / 'dataset' / 'overlap'

    if not overlap_dir.exists():
        print(f"Error: Overlap directory not found: {overlap_dir}")
        return

    print("="*70)
    print("Fix Overlap Ratios V2 (Symmetric Intersection)")
    print("="*70)
    print()

    # Backup
    if args.backup:
        backup_file = eval_dir / 'per_image_results.json.backup_v1'
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

    # Determine split
    first_image = per_image_data[0]['image_path'] if per_image_data else ''
    split = 'train' if 'train_' in first_image else 'test'
    print(f"Detected split: {split}")
    print()

    # Fix each image
    print("Fixing overlap ratios with symmetric intersection...")
    for i, image_result in enumerate(per_image_data):
        per_image_data[i] = fix_image_overlap_stats(image_result, overlap_dir, split)

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
        print("test_0000.jpg (2 shapes with 8,382 pixel intersection):")
        for i, gt in enumerate(test_0000['ground_truth']):
            print(f"  Shape {i} ({gt['category']}): {gt['overlap_ratio']:.4f} ({gt['overlap_ratio']*100:.1f}%)")
        print(f"  Image overlap: {test_0000['overlap_stats']['overlap_ratio']:.4f}")
        print()

    print("Done!")
    print()


if __name__ == '__main__':
    main()
