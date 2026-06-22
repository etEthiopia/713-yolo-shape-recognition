# Detailed YOLO Evaluation Script

## Overview

The `detailed_evaluation.py` script provides comprehensive per-image evaluation metrics for YOLO models with support for:
- **Cross-experiment evaluation** (train on one background, test on another)
- **Overlap statistics** (performance on overlapping vs non-overlapping shapes)
- **Per-shape overlap ratios** (visibility/occlusion metrics for each detected shape)
- **Detailed JSON outputs** for downstream plotting and analysis

## Features

### Per-Image Metrics
Each image evaluation includes:
- **Ground truth** with overlap ratios and visibility ratios for each shape
- **Predictions** with confidence scores and matched overlap ratios
- **Matching results** (true positives, false positives, false negatives)
- **Image-level overlap statistics** (overlap ratio, number of overlapping shapes)
- **Per-category counts** (ground truth, predictions, true positives)

### Aggregate Metrics
- Overall precision, recall, F1 score
- Per-category metrics (cat1, cat2, cat4, cat5)
- Confusion matrix
- Performance comparison: overlapping vs non-overlapping images
- Image-level statistics (mean, std, min, max, median)

### Overlap Analysis
- **Image overlap_ratio**: Total intersection area / Total shape area
- **Shape overlap_ratio**: Occluded area / Shape area (0 = fully visible, 1 = fully occluded)
- **Performance stratification**: Metrics split by overlap presence

## Usage

### 1. Evaluate model on its own test data

```bash
python scripts/detailed_evaluation.py experiments/solid_background_with_overlap_splitted
```

**Output**:
```
experiments/solid_background_with_overlap_splitted/detailed_evaluation/
├── per_image_results.json       # Detailed per-image metrics
├── aggregate_metrics.json        # Aggregated statistics
└── evaluation_summary.json       # Complete evaluation report
```

### 2. Cross-experiment evaluation

Train on natural scenes, test on line backgrounds:

```bash
python scripts/detailed_evaluation.py experiments/natural_scenes_background_with_overlap_splitted \
    --test-data experiments/line_background_with_overlap_splitted
```

**Output**:
```
experiments/natural_scenes_background_with_overlap_splitted/
    detailed_evaluation_on_line_background_with_overlap_splitted/
    ├── per_image_results.json
    ├── aggregate_metrics.json
    └── evaluation_summary.json
```

### 3. Custom confidence and IoU thresholds

```bash
python scripts/detailed_evaluation.py experiments/solid_background_with_overlap_splitted \
    --conf 0.3 \
    --iou 0.6
```

### 4. Custom output directory

```bash
python scripts/detailed_evaluation.py experiments/solid_background_with_overlap_splitted \
    --output-dir custom_results/my_evaluation
```

## Command-Line Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model_experiment` | str | required | Path to trained experiment directory |
| `--test-data` | str | same as model | Path to experiment containing test data |
| `--conf` | float | 0.25 | Confidence threshold for predictions |
| `--iou` | float | 0.5 | IoU threshold for matching predictions to ground truth |
| `--output-dir` | str | auto | Custom output directory |

## Output Files

### 1. per_image_results.json

Array of per-image evaluation results:

```json
[
  {
    "image_path": "test_0042.jpg",
    "image_size": {"width": 640, "height": 640},
    "overlap_stats": {
      "has_overlap": true,
      "overlap_ratio": 0.23,
      "num_overlapping_shapes": 2,
      "metadata_available": true
    },
    "ground_truth": [
      {
        "class_id": 0,
        "category": "cat1",
        "bbox": {"x_center": 0.5, "y_center": 0.5, "width": 0.3, "height": 0.3},
        "overlap_ratio": 0.15,
        "visibility_ratio": 0.85
      }
    ],
    "predictions": [
      {
        "class_id": 0,
        "category": "cat1",
        "confidence": 0.92,
        "bbox": {"x_center": 0.51, "y_center": 0.49, "width": 0.29, "height": 0.31},
        "overlap_ratio": 0.15
      }
    ],
    "matching": {
      "matches": [
        {
          "prediction_idx": 0,
          "ground_truth_idx": 0,
          "iou": 0.87,
          "class_id": 0,
          "confidence": 0.92,
          "correct_class": true
        }
      ],
      "false_positives": [],
      "false_negatives": []
    },
    "metrics": {
      "num_ground_truth": 1,
      "num_predictions": 1,
      "true_positives": 1,
      "false_positives": 0,
      "false_negatives": 0,
      "precision": 1.0,
      "recall": 1.0,
      "f1_score": 1.0,
      "mean_iou": 0.87
    },
    "per_category": {
      "ground_truth_counts": {"cat1": 1, "cat2": 0, "cat4": 0, "cat5": 0},
      "prediction_counts": {"cat1": 1, "cat2": 0, "cat4": 0, "cat5": 0},
      "true_positive_counts": {"cat1": 1, "cat2": 0, "cat4": 0, "cat5": 0}
    }
  }
]
```

### 2. aggregate_metrics.json

Aggregated statistics across all images:

```json
{
  "overall": {
    "total_images": 250,
    "total_ground_truth": 1248,
    "total_predictions": 1205,
    "true_positives": 1102,
    "false_positives": 103,
    "false_negatives": 146,
    "precision": 0.9145,
    "recall": 0.8830,
    "f1_score": 0.8985,
    "images_with_overlap": 125,
    "images_without_overlap": 125
  },
  "per_category": {
    "cat1": {
      "ground_truth": 312,
      "predictions": 305,
      "true_positives": 290,
      "false_positives": 15,
      "false_negatives": 22,
      "precision": 0.9508,
      "recall": 0.9295,
      "f1_score": 0.9401
    },
    "cat2": {...},
    "cat4": {...},
    "cat5": {...}
  },
  "overlap_analysis": {
    "with_overlap": {
      "num_images": 125,
      "precision": 0.88,
      "recall": 0.82,
      "f1_score": 0.85
    },
    "without_overlap": {
      "num_images": 125,
      "precision": 0.95,
      "recall": 0.94,
      "f1_score": 0.945
    }
  },
  "confusion_matrix": [[290, 0, 0, 0], [0, 280, 0, 0], ...],
  "image_statistics": {
    "precision": {
      "mean": 0.9145,
      "std": 0.1234,
      "min": 0.5000,
      "max": 1.0000,
      "median": 0.9500
    },
    "recall": {...},
    "f1_score": {...}
  }
}
```

### 3. evaluation_summary.json

Complete evaluation metadata + metrics:

```json
{
  "evaluation_info": {
    "timestamp": "2026-05-31T20:45:00",
    "model_experiment": "solid_background_with_overlap_splitted",
    "test_experiment": "solid_background_with_overlap_splitted",
    "model_weights": "experiments/.../training/run_1/weights/best.pt",
    "test_images_dir": "experiments/.../dataset/images/test",
    "num_test_images": 250,
    "conf_threshold": 0.25,
    "iou_threshold": 0.5,
    "is_cross_experiment": false
  },
  "metrics": { ... }
}
```

## Use Cases

### 1. Compare all three experiments on their own test data

```bash
# Evaluate each experiment
for exp in solid_background_with_overlap_splitted \
           line_background_with_overlap_splitted \
           natural_scenes_background_with_overlap_splitted; do
    python scripts/detailed_evaluation.py experiments/$exp
done
```

### 2. Cross-experiment evaluation matrix (3×3 comparisons)

```bash
# Train on solid, test on all backgrounds
python scripts/detailed_evaluation.py experiments/solid_background_with_overlap_splitted \
    --test-data experiments/solid_background_with_overlap_splitted

python scripts/detailed_evaluation.py experiments/solid_background_with_overlap_splitted \
    --test-data experiments/line_background_with_overlap_splitted

python scripts/detailed_evaluation.py experiments/solid_background_with_overlap_splitted \
    --test-data experiments/natural_scenes_background_with_overlap_splitted

# Repeat for line and natural scenes models...
```

### 3. Analyze overlap impact

Extract overlap analysis from aggregate_metrics.json:

```bash
cat experiments/solid_background_with_overlap_splitted/detailed_evaluation/aggregate_metrics.json | \
    jq '.overlap_analysis'
```

### 4. Extract per-category precision for all test images

```bash
cat experiments/solid_background_with_overlap_splitted/detailed_evaluation/per_image_results.json | \
    jq '[.[] | {image: .image_path, cat1_tp: .per_category.true_positive_counts.cat1}]'
```

## Downstream Analysis

The JSON outputs are designed for easy plotting with Python:

```python
import json
import matplotlib.pyplot as plt

# Load results
with open('experiments/.../detailed_evaluation/per_image_results.json') as f:
    results = json.load(f)

# Plot precision vs overlap ratio
overlap_ratios = [r['overlap_stats']['overlap_ratio'] for r in results]
precisions = [r['metrics']['precision'] for r in results]

plt.scatter(overlap_ratios, precisions)
plt.xlabel('Image Overlap Ratio')
plt.ylabel('Precision')
plt.title('Detection Precision vs Shape Overlap')
plt.show()
```

## Requirements

- Trained YOLO model (best.pt weights)
- Test images and labels
- Overlap metadata JSON files (optional, for overlap statistics)

## Notes

- **Overlap metadata**: If overlap metadata is not available, overlap statistics will default to 0
- **Cross-experiment**: Works seamlessly - just point to different test data
- **Performance**: ~0.5-1 second per image on CPU, faster on GPU
- **Memory**: Handles 1000+ images without issues

## Examples

See the three shape-split experiments for complete examples:
- `experiments/solid_background_with_overlap_splitted`
- `experiments/line_background_with_overlap_splitted`
- `experiments/natural_scenes_background_with_overlap_splitted`
