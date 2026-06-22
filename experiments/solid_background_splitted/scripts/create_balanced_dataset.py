#!/usr/bin/env python3
"""
Solid Background Experiment - Create Balanced Dataset

Reads from shared dataset/balanced_shapes.json and creates experiment-specific reference.
"""
import json
import shutil
from pathlib import Path

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

    # Create experiment-specific dataset (same as shared for this experiment)
    experiment_data = {
        'metadata': {
            'experiment': 'solid_background_splitted',
            'source': 'Shared balanced dataset',
            'samples_per_category': metadata['samples_per_category'],
            'total_shapes': metadata['total_shapes'],
            'categories': metadata['categories'],
            'excluded': metadata['excluded'],
            'random_seed': metadata['random_seed']
        },
        'shapes': shapes
    }

    # Save to experiment directory
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(experiment_data, f, indent=2)

    print(f"\n✓ Experiment dataset saved to: {OUTPUT_JSON.relative_to(EXPERIMENT_DIR)}")

    # Verification
    print(f"\n✓ Verification:")
    for cat in required_categories:
        count = len(shapes[cat])
        print(f"  {cat}: {count} shapes")

    print(f"\nTotal: {sum(len(paths) for paths in shapes.values())} shapes")
    print("\nReady for image generation!")

if __name__ == '__main__':
    create_experiment_dataset()
