#!/usr/bin/env python3
"""
Create experiments/solid_background_splitted from experiments/solid_background

Splits the shapes into distinct train and test sets to ensure the model sees
completely new shapes during testing.

Usage:
    python create_solid_background_splitted.py
"""

import json
import shutil
import random
from pathlib import Path
from typing import Dict, List

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SOURCE_EXP = PROJECT_ROOT / 'experiments' / 'solid_background'
TARGET_EXP = PROJECT_ROOT / 'experiments' / 'solid_background_splitted'

# Split ratio
TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def split_shapes(shapes_dict: Dict[str, List[str]], train_ratio: float = 0.8,
                seed: int = 42) -> Dict[str, Dict[str, List[str]]]:
    """
    Split shapes into train and test sets.

    Args:
        shapes_dict: Dict of category -> list of shape paths
        train_ratio: Ratio for train split (default 0.8)
        seed: Random seed for reproducibility

    Returns:
        Dict of category -> {"train": [...], "test": [...]}
    """
    random.seed(seed)

    split_shapes = {}

    for category, shape_list in shapes_dict.items():
        # Shuffle shapes
        shuffled = shape_list.copy()
        random.shuffle(shuffled)

        # Split
        n_total = len(shuffled)
        n_train = int(n_total * train_ratio)

        split_shapes[category] = {
            "train": shuffled[:n_train],
            "test": shuffled[n_train:]
        }

    return split_shapes


def create_splitted_experiment():
    """Create solid_background_splitted experiment from solid_background."""

    print("=" * 70)
    print("Create experiments/solid_background_splitted")
    print("=" * 70)
    print()

    # Check source exists
    if not SOURCE_EXP.exists():
        print(f"Error: Source experiment not found: {SOURCE_EXP}")
        return 1

    # Check if target already exists
    if TARGET_EXP.exists():
        response = input(f"\n{TARGET_EXP} already exists. Overwrite? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return 0
        print(f"\nRemoving existing {TARGET_EXP.name}...")
        shutil.rmtree(TARGET_EXP)

    print(f"Source: {SOURCE_EXP.name}")
    print(f"Target: {TARGET_EXP.name}")
    print()

    # Create target directory structure
    print("Creating directory structure...")
    TARGET_EXP.mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'dataset' / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'dataset' / 'images' / 'test').mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'dataset' / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'dataset' / 'labels' / 'test').mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'config').mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'metadata').mkdir(parents=True, exist_ok=True)
    (TARGET_EXP / 'scripts').mkdir(parents=True, exist_ok=True)
    print("  ✓ Directory structure created")
    print()

    # Use the same split as other splitted experiments for consistency
    print("Copying shapes split from solid_background_with_overlap_splitted...")
    reference_split = PROJECT_ROOT / 'experiments' / 'solid_background_with_overlap_splitted' / 'dataset' / 'selected_shapes.json'

    if reference_split.exists():
        # Copy the exact same split
        with open(reference_split, 'r') as f:
            reference_data = json.load(f)

        # Update experiment name
        reference_data['metadata']['experiment'] = 'solid_background_splitted'

        # Save
        target_shapes_file = TARGET_EXP / 'dataset' / 'selected_shapes.json'
        with open(target_shapes_file, 'w') as f:
            json.dump(reference_data, f, indent=2)

        # Extract counts for display
        metadata = reference_data['metadata']
        total_train = metadata['total_train_shapes']
        total_test = metadata['total_test_shapes']
        train_per_cat = metadata['train_per_category']
        test_per_cat = metadata['test_per_category']

        print(f"  ✓ Using same split as other experiments")
        print(f"  Train shapes: {total_train} ({train_per_cat} per category)")
        print(f"  Test shapes: {total_test} ({test_per_cat} per category)")
        print(f"  ✓ Saved: {target_shapes_file}")
    else:
        print(f"  Warning: Reference split not found at {reference_split}")
        print(f"  Creating new split...")

        # Fallback: create new split
        source_shapes_file = SOURCE_EXP / 'dataset' / 'selected_shapes.json'
        with open(source_shapes_file, 'r') as f:
            source_data = json.load(f)

        shapes_dict = source_data['shapes']
        metadata = source_data['metadata']

        split_shapes_dict = split_shapes(shapes_dict, TRAIN_RATIO, RANDOM_SEED)

        train_counts = {cat: len(shapes['train']) for cat, shapes in split_shapes_dict.items()}
        test_counts = {cat: len(shapes['test']) for cat, shapes in split_shapes_dict.items()}
        total_train = sum(train_counts.values())
        total_test = sum(test_counts.values())

        target_shapes_data = {
            "metadata": {
                "experiment": "solid_background_splitted",
                "source": "Shared balanced dataset with train/test split",
                "samples_per_category": metadata['samples_per_category'],
                "train_per_category": train_counts['cat1'],
                "test_per_category": test_counts['cat1'],
                "split_ratio": TRAIN_RATIO,
                "total_shapes": metadata['total_shapes'],
                "total_train_shapes": total_train,
                "total_test_shapes": total_test,
                "categories": metadata['categories'],
                "excluded": metadata['excluded'],
                "random_seed": RANDOM_SEED,
                "shape_separation": "strict train/test split with zero overlap"
            },
            "shapes": split_shapes_dict
        }

        target_shapes_file = TARGET_EXP / 'dataset' / 'selected_shapes.json'
        with open(target_shapes_file, 'w') as f:
            json.dump(target_shapes_data, f, indent=2)
        print(f"  ✓ Saved: {target_shapes_file}")

    print()

    # Copy config
    print("Copying configuration files...")
    source_config = SOURCE_EXP / 'config' / 'shapes.yaml'
    target_config = TARGET_EXP / 'config' / 'shapes.yaml'

    if source_config.exists():
        # Read and update paths
        with open(source_config, 'r') as f:
            config_content = f.read()

        # Replace experiment name in paths
        config_content = config_content.replace(
            'experiments/solid_background',
            'experiments/solid_background_splitted'
        )

        with open(target_config, 'w') as f:
            f.write(config_content)
        print(f"  ✓ Copied and updated: shapes.yaml")
    print()

    # Copy scripts
    print("Copying scripts...")
    source_scripts = SOURCE_EXP / 'scripts'
    if source_scripts.exists():
        for script_file in source_scripts.glob('*.py'):
            target_script = TARGET_EXP / 'scripts' / script_file.name

            # Read script and update paths
            with open(script_file, 'r') as f:
                script_content = f.read()

            # Replace experiment name
            script_content = script_content.replace(
                'solid_background',
                'solid_background_splitted'
            )

            with open(target_script, 'w') as f:
                f.write(script_content)

            # Make executable
            target_script.chmod(0o755)

        # Copy shell scripts
        for script_file in source_scripts.glob('*.sh'):
            target_script = TARGET_EXP / 'scripts' / script_file.name

            with open(script_file, 'r') as f:
                script_content = f.read()

            script_content = script_content.replace(
                'solid_background',
                'solid_background_splitted'
            )

            with open(target_script, 'w') as f:
                f.write(script_content)

            target_script.chmod(0o755)

        print(f"  ✓ Copied scripts")
    print()

    # Copy documentation
    print("Copying documentation...")
    for doc_file in SOURCE_EXP.glob('*.md'):
        target_doc = TARGET_EXP / doc_file.name

        with open(doc_file, 'r') as f:
            doc_content = f.read()

        # Update experiment name
        doc_content = doc_content.replace(
            'solid_background',
            'solid_background_splitted'
        )
        doc_content = doc_content.replace(
            'Solid Background',
            'Solid Background (Splitted)'
        )

        with open(target_doc, 'w') as f:
            f.write(doc_content)

    print(f"  ✓ Copied documentation")
    print()

    # Create README
    readme_content = """# Solid Background (Splitted) Experiment

This experiment is a variant of `solid_background` where shapes are strictly separated
into train and test sets. This ensures the model is evaluated on completely unseen shapes.

## Key Differences from solid_background

1. **Shape Separation**: Train and test use different shapes (80/20 split)
   - Train: 312 shapes per category (1,248 total)
   - Test: 78 shapes per category (312 total)

2. **Purpose**: Evaluate model generalization to new shape instances

## Dataset Statistics

- **Categories**: cat1, cat2, cat4, cat5 (cat3 excluded)
- **Train Images**: 1,000
- **Test Images**: 250
- **Background**: Solid grey (128, 128, 128)
- **Shapes per Image**: 2-5

## Usage

1. Generate dataset:
   ```bash
   cd experiments/solid_background_splitted
   bash scripts/run_experiment.sh
   ```

2. Or run individual steps:
   ```bash
   # Generate images
   python scripts/generate_composite_images.py

   # Train model
   python scripts/train_model.py

   # Evaluate
   python scripts/evaluate_model.py
   ```

## Comparison with solid_background

The original `solid_background` experiment may have used overlapping shapes between
train and test sets. This splitted version ensures zero overlap for more rigorous
evaluation of generalization capability.
"""

    with open(TARGET_EXP / 'README.md', 'w') as f:
        f.write(readme_content)

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"✓ Created: {TARGET_EXP}")
    print(f"  Train shapes: {total_train} (across 4 categories)")
    print(f"  Test shapes: {total_test} (across 4 categories)")
    print(f"  Split ratio: {TRAIN_RATIO}")
    print()
    print("Next steps:")
    print(f"  1. cd {TARGET_EXP}")
    print(f"  2. Generate images: python scripts/generate_composite_images.py")
    print(f"  3. Train model: python scripts/train_model.py")
    print(f"  4. Evaluate: python scripts/evaluate_model.py")
    print()

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(create_splitted_experiment())
