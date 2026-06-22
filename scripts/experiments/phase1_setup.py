#!/usr/bin/env python3
"""
Phase 1: Setup - Create experiment directory structure and configuration
"""
import os
import json
import yaml
from pathlib import Path

# Experiment configuration
EXPERIMENT_NAME = "solid_grey_background"
BASE_DIR = Path("experiments") / EXPERIMENT_NAME

# Categories (excluding cat3)
CATEGORIES = ["cat1", "cat2", "cat4", "cat5"]
CATEGORY_MAPPING = {
    0: "cat1",  # Unconstrained
    1: "cat2",  # Local (Var)
    2: "cat4",  # Local (Matched)
    3: "cat5"   # Natural
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
        BASE_DIR / "evaluation" / "predictions",
        BASE_DIR / "metadata"
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created: {directory}")

    return BASE_DIR

def create_yolo_config():
    """Create YOLO dataset configuration YAML."""
    print("\nCreating YOLO configuration...")

    config_path = BASE_DIR / "config" / "shapes.yaml"

    # Get absolute path to dataset
    dataset_path = (Path.cwd() / BASE_DIR / "dataset").resolve()

    config = {
        "path": str(dataset_path),
        "train": "images/train",
        "val": "images/test",
        "nc": 4,  # 4 categories (excluding cat3)
        "names": {
            0: "cat1",  # Unconstrained
            1: "cat2",  # Local (Var)
            2: "cat4",  # Local (Matched)
            3: "cat5"   # Natural
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Created: {config_path}")
    return config_path

def save_experiment_metadata():
    """Save experiment configuration to metadata JSON."""
    print("\nSaving experiment metadata...")

    metadata_path = BASE_DIR / "metadata" / "experiment_config.json"

    metadata = {
        "experiment_name": "Solid Grey Background",
        "description": "YOLO shape recognition with 4 categories on solid grey background",
        "categories": CATEGORIES,
        "excluded_categories": ["cat3"],
        "rationale": "Excluding cat3 due to poor baseline performance (18.6% mAP)",
        "parameters": {
            "shapes_per_image_range": [2, 7],
            "background_color": "grey (128, 128, 128)",
            "train_images": 1000,
            "test_images": 250,
            "image_size": 640,
            "epochs": 200,
            "batch_size": 16,
            "model": "yolov8n.pt",
            "random_seed": 42
        },
        "color_scheme": {
            "cat1": {"display_name": "Unconstrained", "hex": "#00AADC", "rgb": [0, 170, 220]},
            "cat2": {"display_name": "Local (Var)", "hex": "#00539B", "rgb": [0, 83, 155]},
            "cat4": {"display_name": "Local (Matched)", "hex": "#7F2F8D", "rgb": [127, 47, 141]},
            "cat5": {"display_name": "Natural", "hex": "#B01F23", "rgb": [176, 31, 35]}
        },
        "directory_structure": {
            "dataset": "Generated images and labels",
            "config": "YOLO configuration files",
            "training": "Model weights and training logs",
            "evaluation": "Metrics, figures, and analysis",
            "metadata": "Experiment parameters and statistics"
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Created: {metadata_path}")
    return metadata_path

def main():
    print("=" * 70)
    print("Phase 1: Experiment Setup")
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
    print(f"\nExperiment directory: {exp_dir}")
    print(f"YOLO config: {config_path}")
    print(f"Metadata: {metadata_path}")
    print()
    print("Ready for Phase 2: Dataset Creation")
    print()

if __name__ == "__main__":
    main()
