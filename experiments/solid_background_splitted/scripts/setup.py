#!/usr/bin/env python3
"""
Solid Background Experiment - Setup Script

Creates directory structure, YOLO config, and metadata.
Categories: cat1, cat2, cat4, cat5 (excluding cat3)
"""
import os
import json
import yaml
from pathlib import Path

# Experiment configuration
EXPERIMENT_NAME = "solid_background_splitted"
BASE_DIR = Path(__file__).parent.parent  # experiments/solid_background_splitted/

# Categories (excluding cat3)
CATEGORIES = ["cat1", "cat2", "cat4", "cat5"]

# Critical: Sequential class mapping (not formula-based!)
CLASS_MAPPING = {
    "cat1": 0,
    "cat2": 1,
    "cat4": 2,  # Class 2, NOT 3!
    "cat5": 3   # Class 3, NOT 4!
}

# Official color scheme
COLOR_SCHEME = {
    "cat1": {
        "display_name": "Unconstrained",
        "hex": "#00AADC",
        "rgb": [0, 170, 220]
    },
    "cat2": {
        "display_name": "Local (Var)",
        "hex": "#00539B",
        "rgb": [0, 83, 155]
    },
    "cat4": {
        "display_name": "Local (Matched)",
        "hex": "#7F2F8D",
        "rgb": [127, 47, 141]
    },
    "cat5": {
        "display_name": "Natural",
        "hex": "#B01F23",
        "rgb": [176, 31, 35]
    }
}

def create_directory_structure():
    """Create all necessary directories."""
    print("Creating directory structure...")

    directories = [
        BASE_DIR / "dataset" / "images" / "train",
        BASE_DIR / "dataset" / "images" / "test",
        BASE_DIR / "dataset" / "labels" / "train",
        BASE_DIR / "dataset" / "labels" / "test",
        BASE_DIR / "config",
        BASE_DIR / "training",
        BASE_DIR / "evaluation" / "figures",
        BASE_DIR / "metadata"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory.relative_to(BASE_DIR.parent)}")

    return BASE_DIR

def create_yolo_config():
    """Create YOLO dataset configuration YAML."""
    print("\nCreating YOLO configuration...")

    config_path = BASE_DIR / "config" / "shapes.yaml"

    # Get absolute path to dataset
    dataset_path = (BASE_DIR / "dataset").resolve()

    config = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/test",
        "nc": len(CATEGORIES),  # 4 classes
        "names": {CLASS_MAPPING[cat]: cat for cat in CATEGORIES}
    }

    # Add display names as comments
    config["# Category Mapping"] = "Sequential IDs 0-3"
    config["# Class 0"] = f"{CATEGORIES[0]} - {COLOR_SCHEME[CATEGORIES[0]]['display_name']}"
    config["# Class 1"] = f"{CATEGORIES[1]} - {COLOR_SCHEME[CATEGORIES[1]]['display_name']}"
    config["# Class 2"] = f"{CATEGORIES[2]} - {COLOR_SCHEME[CATEGORIES[2]]['display_name']}"
    config["# Class 3"] = f"{CATEGORIES[3]} - {COLOR_SCHEME[CATEGORIES[3]]['display_name']}"

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ {config_path.relative_to(BASE_DIR.parent)}")
    print(f"  Number of classes: {len(CATEGORIES)}")
    print(f"  Class mapping:")
    for cat in CATEGORIES:
        print(f"    {CLASS_MAPPING[cat]}: {cat} ({COLOR_SCHEME[cat]['display_name']})")

    return config_path

def save_experiment_metadata():
    """Save experiment configuration to metadata JSON."""
    print("\nSaving experiment metadata...")

    metadata_path = BASE_DIR / "metadata" / "experiment_config.json"

    metadata = {
        "experiment_name": "Solid Background",
        "description": "YOLO shape recognition with 4 categories on solid grey background",
        "categories": CATEGORIES,
        "excluded_categories": ["cat3"],
        "rationale": "Excluding cat3 to focus on well-separated categories",
        "class_mapping": CLASS_MAPPING,
        "parameters": {
            "shapes_per_image_range": [2, 7],
            "background_color": "solid grey (128, 128, 128)",
            "train_images": 1000,
            "test_images": 250,
            "image_size": 640,
            "epochs": 50,
            "batch_size": 16,
            "model": "yolov8n.pt",
            "random_seed": 42
        },
        "color_scheme": COLOR_SCHEME,
        "directory_structure": {
            "dataset": "Generated images and YOLO labels",
            "config": "YOLO configuration",
            "training": "Model weights and training logs",
            "evaluation": "Metrics and visualizations",
            "metadata": "Experiment parameters"
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ {metadata_path.relative_to(BASE_DIR.parent)}")
    return metadata_path

def main():
    print("=" * 70)
    print("Solid Background Experiment - Setup")
    print("=" * 70)
    print()

    # Create directories
    exp_dir = create_directory_structure()

    # Create YOLO config
    config_path = create_yolo_config()

    # Save metadata
    metadata_path = save_experiment_metadata()

    print()
    print("=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print(f"\nExperiment directory: {exp_dir.relative_to(exp_dir.parent)}")
    print(f"\nNext steps:")
    print(f"  1. Run create_balanced_dataset.py")
    print(f"  2. Run generate_composite_images.py")
    print(f"  3. Train YOLO model")
    print()

if __name__ == "__main__":
    main()
