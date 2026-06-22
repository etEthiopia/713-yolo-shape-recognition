#!/usr/bin/env python3
"""
Line Background with Overlap (Shape-Split) - Create Balanced Dataset

Reads from shared dataset/balanced_shapes.json and creates experiment-specific dataset
with STRICT train/test shape separation (80/20 split).
"""
import json
from pathlib import Path
from typing import List, Tuple

def split_shapes_train_test(shapes: List[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """
    Split shapes into train and test sets.

    Args:
        shapes: List of shape file paths
        train_ratio: Fraction of shapes for training (default 0.8 = 80%)

    Returns:
        (train_shapes, test_shapes) tuple
    """
    train_size = int(len(shapes) * train_ratio)
    train_shapes = shapes[:train_size]
    test_shapes = shapes[train_size:]

    return train_shapes, test_shapes

def create_experiment_dataset():
    """Create experiment-specific balanced dataset reference."""
    print("=" * 70)
    print("Creating Experiment Balanced Dataset")
    print("=" * 70)

    # Paths
    PROJECT_DIR = Path(__file__).parent.parent.parent.parent
    SHARED_JSON = PROJECT_DIR / 'dataset' / 'balanced_shapes.json'
    EXPERIMENT_DIR = Path(__file__).parent.parent
    OUTPUT_JSON = EXPERIMENT_DIR / 'dataset' / 'selected_shapes.json'

    print(f"\nReading shared balanced dataset: {SHARED_JSON.relative_to(PROJECT_DIR)}")

    # Read shared balanced dataset
    with open(SHARED_JSON, 'r') as f:
        shared_data = json.load(f)

    # Extract metadata
    metadata = shared_data['metadata']
    shapes = shared_data['shapes']

    print(f"\nShared dataset info:")
    print(f"  Total shapes: {metadata['total_shapes']}")
    print(f"  Categories: {', '.join(metadata['categories'])}")
    print(f"  Excluded: {', '.join(metadata['excluded'])}")
    print(f"  Samples per category: {metadata['samples_per_category']}")

    # Verify categories match experiment needs
    required_categories = ['cat1', 'cat2', 'cat4', 'cat5']
    for cat in required_categories:
        if cat not in shapes:
            raise ValueError(f"Required category {cat} not in shared dataset!")

    # Split shapes into train and test sets
    train_ratio = 0.8
    samples_per_category = metadata['samples_per_category']
    train_per_category = int(samples_per_category * train_ratio)
    test_per_category = samples_per_category - train_per_category

    print(f"\nSplitting shapes per category:")
    shapes_split = {}
    for cat in required_categories:
        cat_shapes = shapes[cat]
        train_shapes, test_shapes = split_shapes_train_test(cat_shapes, train_ratio)
        shapes_split[cat] = {
            'train': train_shapes,
            'test': test_shapes
        }
        print(f"  {cat}: {len(train_shapes)} train, {len(test_shapes)} test")

    # Create experiment-specific dataset with train/test split
    experiment_data = {
        'metadata': {
            'experiment': 'line_background_with_overlap_splitted',
            'source': 'Shared balanced dataset with train/test split',
            'samples_per_category': samples_per_category,
            'train_per_category': train_per_category,
            'test_per_category': test_per_category,
            'split_ratio': train_ratio,
            'total_shapes': samples_per_category * len(required_categories),
            'total_train_shapes': train_per_category * len(required_categories),
            'total_test_shapes': test_per_category * len(required_categories),
            'categories': required_categories,
            'excluded': metadata['excluded'],
            'random_seed': metadata['random_seed'],
            'shape_separation': 'strict train/test split with zero overlap'
        },
        'shapes': shapes_split
    }

    # Save to experiment directory
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\n✓ Experiment dataset saved to: {OUTPUT_JSON.relative_to(EXPERIMENT_DIR)}")

    # Verification
    print(f"\n✓ Verification:")
    print(f"  Total shapes: {samples_per_category * len(required_categories)}")
    print(f"  Train shapes: {train_per_category * len(required_categories)} ({train_per_category} per category)")
    print(f"  Test shapes: {test_per_category * len(required_categories)} ({test_per_category} per category)")
    print(f"  Shape separation: STRICT (zero overlap)")
    print("\nReady for image generation!")

if __name__ == '__main__':
    create_experiment_dataset()
