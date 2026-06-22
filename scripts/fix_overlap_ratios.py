#!/usr/bin/env python3
"""
Fix Overlap Ratio Calculations Using Actual Shape Pixel Areas

Corrects overlap ratio calculations in existing evaluation results using actual shape areas
from overlap metadata rather than bounding box areas.

Correct calculation:
- Union area = Sum of actual shape areas - Total intersection area
- Image overlap ratio = Total intersection area / Union area
- Per-shape overlap ratio = Intersection with others / Shape's actual area

Usage:
    python fix_overlap_ratios.py experiments/solid_background_with_overlap_splitted/detailed_evaluation
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional
import shutil


def load_overlap_metadata(overlap_file: Path) -> Optional[Dict]:
    """Load overlap metadata JSON file."""
    if not overlap_file.exists():
        return None

    with open(overlap_file, 'r') as f:
        return json.load(f)


def calculate_corrected_overlap_stats(overlap_metadata: Dict) -> Dict:
    """
    Calculate corrected overlap statistics using actual pixel areas.

    Args:
        overlap_metadata: Metadata with actual shape areas

    Returns:
        Dict with per-shape overlap ratios and image-level stats
    """
    shapes = overlap_metadata.get('shapes', [])

    if not shapes:
        return {
            'per_shape_overlap_ratios': [],
            'image_overlap_ratio': 0.0,
            'has_overlap': False,
            'num_overlapping_shapes': 0,
            'union_area': 0.0,
            'total_intersection_area': 0.0
        }

    # Calculate per-shape overlap ratios
    per_shape_overlap_ratios = []

    for shape in shapes:
        total_area = shape['total_area']
        visible_area = shape['visible_area']

        if total_area == 0:
            overlap_ratio = 0.0
        else:
            # Intersection area = total_area - visible_area
            intersection_area = total_area - visible_area
            overlap_ratio = intersection_area / total_area

        per_shape_overlap_ratios.append(overlap_ratio)

    # Calculate total intersection area (avoid double counting)
    # From statistics in metadata
    total_intersection_area = overlap_metadata.get('statistics', {}).get('total_overlap_area', 0)

    # Calculate union area: sum of actual areas - total intersection
    sum_of_areas = sum(s['total_area'] for s in shapes)
    union_area = sum_of_areas - total_intersection_area

    # Image-level overlap ratio
    if union_area > 0:
        image_overlap_ratio = total_intersection_area / union_area
    else:
        image_overlap_ratio = 0.0

    # Count overlapping shapes
    num_overlapping_shapes = sum(1 for ratio in per_shape_overlap_ratios if ratio > 0)

    return {
        'per_shape_overlap_ratios': per_shape_overlap_ratios,
        'image_overlap_ratio': image_overlap_ratio,
        'has_overlap': total_intersection_area > 0,
        'num_overlapping_shapes': num_overlapping_shapes,
        'union_area': union_area,
        'total_intersection_area': total_intersection_area
    }


def fix_image_overlap_stats(image_result: Dict, overlap_dir: Path, split: str) -> Dict:
    """
    Fix overlap statistics for a single image using actual pixel areas.

    Args:
        image_result: Image result dict
        overlap_dir: Base overlap directory
        split: 'train' or 'test'

    Returns:
        Updated image_result with corrected overlap stats
    """
    image_name = image_result.get('image_path', '')

    # Construct overlap metadata path
    base_name = Path(image_name).stem
    overlap_file = overlap_dir / split / f"{base_name}.json"

    # Load overlap metadata
    overlap_metadata = load_overlap_metadata(overlap_file)

    if overlap_metadata is None:
        print(f"  Warning: No overlap metadata for {image_name}")
        return image_result

    # Calculate corrected stats
    corrected = calculate_corrected_overlap_stats(overlap_metadata)

    # Update ground truth with corrected overlap ratios
    ground_truth = image_result.get('ground_truth', [])
    per_shape_ratios = corrected['per_shape_overlap_ratios']

    for i, gt in enumerate(ground_truth):
        if i < len(per_shape_ratios):
            gt['overlap_ratio'] = per_shape_ratios[i]
            gt['visibility_ratio'] = 1.0 - per_shape_ratios[i]

    # Update image-level overlap stats
    image_result['overlap_stats'] = {
        'has_overlap': corrected['has_overlap'],
        'overlap_ratio': corrected['image_overlap_ratio'],
        'num_overlapping_shapes': corrected['num_overlapping_shapes'],
        'metadata_available': True,
        'union_area': corrected['union_area'],
        'total_intersection_area': corrected['total_intersection_area']
    }

    # Update predictions with matched ground truth overlap ratios
    predictions = image_result.get('predictions', [])
    matches = image_result.get('matching', {}).get('matches', [])

    for pred in predictions:
        pred['overlap_ratio'] = 0.0  # Default

    for match in matches:
        pred_idx = match['prediction_idx']
        gt_idx = match['ground_truth_idx']
        if pred_idx < len(predictions) and gt_idx < len(ground_truth):
            predictions[pred_idx]['overlap_ratio'] = ground_truth[gt_idx]['overlap_ratio']

    return image_result


def main():
    parser = argparse.ArgumentParser(
        description='Fix overlap ratio calculations using actual pixel areas'
    )
    parser.add_argument('eval_dir', type=str,
                       help='Path to detailed evaluation directory')
    parser.add_argument('--backup', action='store_true', default=True,
                       help='Create backup before fixing (default: True)')

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

    print("=" * 70)
    print("Fix Overlap Ratio Calculations (Using Actual Pixel Areas)")
    print("=" * 70)
    print()
    print(f"Evaluation dir: {eval_dir}")
    print(f"Overlap dir: {overlap_dir}")
    print()

    # Backup if requested
    if args.backup:
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

    # Determine split (train or test)
    # Check first image name to determine split
    first_image = per_image_data[0]['image_path'] if per_image_data else ''
    split = 'train' if 'train_' in first_image else 'test'
    print(f"Detected split: {split}")
    print()

    # Fix overlap stats for each image
    print("Fixing overlap ratios using actual pixel areas...")
    errors = 0
    for i, image_result in enumerate(per_image_data):
        try:
            per_image_data[i] = fix_image_overlap_stats(image_result, overlap_dir, split)
        except Exception as e:
            print(f"  Error processing image {i}: {e}")
            errors += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(per_image_data)} images...")

    print(f"  ✓ Fixed {len(per_image_data) - errors}/{len(per_image_data)} images")
    if errors > 0:
        print(f"  ✗ {errors} errors encountered")
    print()

    # Save corrected data
    print(f"Saving corrected data to: {per_image_file.name}")
    with open(per_image_file, 'w') as f:
        json.dump(per_image_data, f, indent=2)
    print("  ✓ Saved")
    print()

    # Statistics
    print("=" * 70)
    print("Statistics After Correction")
    print("=" * 70)
    print()

    overlap_ratios = []
    union_areas = []
    intersection_areas = []

    for img in per_image_data:
        overlap_stats = img.get('overlap_stats', {})
        if overlap_stats.get('has_overlap'):
            overlap_ratios.append(overlap_stats.get('overlap_ratio', 0.0))
            union_areas.append(overlap_stats.get('union_area', 0.0))
            intersection_areas.append(overlap_stats.get('total_intersection_area', 0.0))

    if overlap_ratios:
        import numpy as np
        print(f"Images with overlap: {len(overlap_ratios)}/{len(per_image_data)}")
        print(f"Overlap ratio range: {min(overlap_ratios):.4f} - {max(overlap_ratios):.4f}")
        print(f"Mean overlap ratio: {np.mean(overlap_ratios):.4f} ± {np.std(overlap_ratios):.4f}")
        print(f"All ratios ≤ 1.0: {'✓ YES' if all(r <= 1.0 for r in overlap_ratios) else '✗ NO'}")
        print()
        print(f"Mean union area: {np.mean(union_areas):.1f} pixels")
        print(f"Mean intersection area: {np.mean(intersection_areas):.1f} pixels")
    else:
        print("No images with overlap found")

    print()
    print("Done! You can now re-run plotting scripts to see corrected results.")
    print()


if __name__ == '__main__':
    main()
