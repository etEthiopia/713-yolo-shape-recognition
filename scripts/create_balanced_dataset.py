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

def create_balanced_dataset(shapes_dir: str, output_json: str, samples_per_category: int = 390):
    """
    Randomly sample equal number of shapes from each category.

    Args:
        shapes_dir: Path to Shapes directory containing cat1-5 folders
        output_json: Output path for JSON file
        samples_per_category: Number of shapes to sample per category (default: 390)
    """
    print("=" * 60)
    print("Creating Balanced Shape Dataset")
    print("=" * 60)

    shapes_by_category = {}
    category_counts = {}

    # Process each category
    for cat_id in range(1, 6):
        cat_dir = os.path.join(shapes_dir, f'cat{cat_id}')

        # Get all PNG files
        all_shapes = sorted(glob.glob(os.path.join(cat_dir, 'img*.png')))

        if not all_shapes:
            raise ValueError(f"No images found in {cat_dir}")

        category_counts[f'cat{cat_id}'] = len(all_shapes)
        print(f"\nCategory {cat_id}: Found {len(all_shapes)} shapes")

        # Check if we have enough shapes
        if len(all_shapes) < samples_per_category:
            raise ValueError(
                f"Category {cat_id} has only {len(all_shapes)} shapes, "
                f"but {samples_per_category} requested"
            )

        # Random sample
        random.seed(42)  # For reproducibility
        selected = random.sample(all_shapes, samples_per_category)

        # Store as relative paths from project root
        shapes_by_category[f'cat{cat_id}'] = selected
        print(f"  → Sampled {len(selected)} shapes")

    # Create output directory if needed
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to JSON
    with open(output_json, 'w') as f:
        json.dump({
            'metadata': {
                'samples_per_category': samples_per_category,
                'total_shapes': samples_per_category * 5,
                'original_counts': category_counts,
                'random_seed': 42
            },
            'shapes': shapes_by_category
        }, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✓ Balanced dataset saved to: {output_json}")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total categories: 5")
    print(f"  Shapes per category: {samples_per_category}")
    print(f"  Total selected shapes: {samples_per_category * 5}")

    # Verify
    with open(output_json, 'r') as f:
        data = json.load(f)

    print(f"\n✓ Verification: JSON contains {len(data['shapes'])} categories")
    for cat, paths in data['shapes'].items():
        print(f"  {cat}: {len(paths)} shapes")

if __name__ == '__main__':
    # Paths
    PROJECT_DIR = Path(__file__).parent.parent
    SHAPES_DIR = PROJECT_DIR / 'Shapes'
    OUTPUT_JSON = PROJECT_DIR / 'dataset' / 'selected_shapes.json'

    # Create balanced dataset
    create_balanced_dataset(
        shapes_dir=str(SHAPES_DIR),
        output_json=str(OUTPUT_JSON),
        samples_per_category=390
    )
