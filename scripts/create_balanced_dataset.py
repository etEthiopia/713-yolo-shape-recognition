#!/usr/bin/env python3
"""
Create balanced dataset by randomly sampling 390 shapes per category.
Output: dataset/selected_shapes.json
"""
# IMPORTANT: Import SSL config FIRST before any network operations
from ssl_config import setup_ssl_certificates
setup_ssl_certificates()

import json
import random
import glob
import os
from pathlib import Path

def create_balanced_dataset(shapes_dir: str, output_json: str, samples_per_category: int = 390, categories: list = None):
    """
    Randomly sample equal number of shapes from each category.

    Args:
        shapes_dir: Path to Shapes directory containing cat1-5 folders
        output_json: Output path for JSON file
        samples_per_category: Number of shapes to sample per category (default: 390)
        categories: List of categories to include (e.g., ['cat1', 'cat2', 'cat4', 'cat5']). If None, includes all (cat1-5).
    """
    print("=" * 60)
    print("Creating Balanced Shape Dataset")
    print("=" * 60)

    # Default to all categories if not specified
    if categories is None:
        categories = [f'cat{i}' for i in range(1, 6)]

    print(f"\nCategories to include: {', '.join(categories)}")
    excluded = [f'cat{i}' for i in range(1, 6) if f'cat{i}' not in categories]
    if excluded:
        print(f"Excluded categories: {', '.join(excluded)}")

    shapes_by_category = {}
    category_counts = {}

    # Process specified categories
    for cat_name in categories:
        cat_dir = os.path.join(shapes_dir, cat_name)

        # Get all PNG files
        all_shapes = sorted(glob.glob(os.path.join(cat_dir, 'img*.png')))

        if not all_shapes:
            raise ValueError(f"No images found in {cat_dir}")

        category_counts[cat_name] = len(all_shapes)
        print(f"\n{cat_name}: Found {len(all_shapes)} shapes")

        # Check if we have enough shapes
        if len(all_shapes) < samples_per_category:
            raise ValueError(
                f"{cat_name} has only {len(all_shapes)} shapes, "
                f"but {samples_per_category} requested"
            )

        # Random sample
        random.seed(42)  # For reproducibility
        selected = random.sample(all_shapes, samples_per_category)

        # Store as relative paths from project root
        shapes_by_category[cat_name] = selected
        print(f"  → Sampled {len(selected)} shapes")

    # Create output directory if needed
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump({
            'metadata': {
                'samples_per_category': samples_per_category,
                'total_shapes': samples_per_category * len(categories),
                'categories_included': categories,
                'categories_excluded': excluded,
                'original_counts': category_counts,
                'random_seed': 42
            },
            'shapes': shapes_by_category
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Balanced dataset saved to: {output_json}")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total categories: {len(categories)}")
    print(f"  Shapes per category: {samples_per_category}")
    print(f"  Total selected shapes: {samples_per_category * len(categories)}")
    if excluded:
        print(f"  Excluded: {', '.join(excluded)}")

    # Verify
    with open(output_json, 'r') as f:
        data = json.load(f)

    print(f"\n✓ Verification: JSON contains {len(data['shapes'])} categories")
    for cat, paths in data['shapes'].items():
        print(f"  {cat}: {len(paths)} shapes")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create balanced shape dataset')
    parser.add_argument('--shapes-dir', type=str, default='Shapes',
                       help='Directory containing shape category folders')
    parser.add_argument('--output-json', type=str, default='dataset/selected_shapes.json',
                       help='Output JSON file path')
    parser.add_argument('--samples-per-category', type=int, default=390,
                       help='Number of shapes to sample per category')
    parser.add_argument('--categories', nargs='+', default=None,
                       help='Categories to include (e.g., cat1 cat2 cat4 cat5). Default: all')

    args = parser.parse_args()

    # Get project directory
    PROJECT_DIR = Path(__file__).parent.parent
    shapes_dir = PROJECT_DIR / args.shapes_dir
    output_json = PROJECT_DIR / args.output_json

    # Create output directory if needed
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Create balanced dataset
    create_balanced_dataset(
        shapes_dir=str(shapes_dir),
        output_json=str(output_json),
        samples_per_category=args.samples_per_category,
        categories=args.categories
    )
