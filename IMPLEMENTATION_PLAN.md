# YOLO Shape Recognition Experiment Plan

## Context

**Objective**: Compare YOLO's ability to recognize synthetic vs. natural silhouette shapes across 5 categories (cat1-4: synthetic with varying curvature constraints; cat5: natural animal silhouettes).

**Available Data**:
- 2822 total shapes: cat1(600), cat2(720), cat3(720), cat4(391), cat5(391)
- Format: 2048×2048 transparent PNG with white foreground
- Location: `/Shapes/cat{1-5}/img*.png`

**Reference**: Assignment 3 notebook at `/Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/assignment_3/Assignment3_pro.ipynb` shows complete YOLO workflow (Darknet setup, dataset preparation, training, evaluation).

---

## Implementation Steps

### 1. Shape Selection & Sampling
**Script**: `scripts/create_balanced_dataset.py`
**Output**: `dataset/selected_shapes.json`

Create balanced dataset by randomly sampling 390 shapes per category (minimum available in cat4/cat5):

```python
import json, random, glob

shapes_by_category = {}
for cat_id in range(1, 6):
    all_shapes = glob.glob(f'Shapes/cat{cat_id}/img*.png')
    selected = random.sample(all_shapes, 390)
    shapes_by_category[f'cat{cat_id}'] = selected

# Save for reproducibility
with open('dataset/selected_shapes.json', 'w') as f:
    json.dump(shapes_by_category, f, indent=2)
```

**Total**: 1950 shapes (390 × 5 categories), balanced across all categories

### 2. Composite Image Generation
**Script**: `scripts/generate_composite_images.py`
**Input**: `dataset/selected_shapes.json`
**Output**: Training/test images + YOLO labels

Create composite images by sampling from the balanced JSON:

- **Image specs**: 640×640 grey background (RGB: 128,128,128)
- **Shapes per image**: Random 2-5 shapes
- **Sampling strategy**: Load paths from `selected_shapes.json`, randomly pick category → pick random shape from that category
- **Balance**: Equal category representation across all generated images (track counts)
- **Augmentation per shape**: Random scale (0.1-0.3 of image size), rotation (0-360°), position (non-overlapping)
- **Dataset split**: Generate train/test by referencing the JSON
  - Train: 500 images
  - Test: 125 images
- **Output format**:
  - Images: `dataset/images/train/*.jpg`, `dataset/images/test/*.jpg`
  - YOLO labels: `dataset/labels/train/*.txt`, `dataset/labels/test/*.txt` (one line per shape: `<class_id> <x_center> <y_center> <width> <height>` in normalized coordinates)

**Key functions**:
- `load_selected_shapes(json_path)`: Load balanced shape paths from JSON
- `sample_shape(shapes_dict)`: Randomly pick category, then shape from that category
- `load_shape_image(path)`: Load PNG, extract silhouette mask from alpha channel
- `paste_shape_augmented(canvas, shape_mask, scale, rotation, position)`: Apply transformations and paste
- `compute_bounding_box(mask, transform_params)`: Compute YOLO bbox from binary mask + transforms
- `ensure_no_overlap(existing_boxes, new_box, min_separation)`: Collision detection

### 3. YOLO Configuration
**Files**: `config/shapes.yaml`, `config/yolov8n-shapes.yaml`

Based on Assignment 3 structure, create:
- **Dataset config** (`shapes.yaml`):
  ```yaml
  path: /absolute/path/to/dataset
  train: images/train
  val: images/test
  nc: 5
  names: ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']
  ```
- **Model config**: Use YOLOv8-nano pretrained weights (lighter than v4-tiny from Assignment 3)
- **Training config**: 
  - Input size: 640×640 (standard for YOLOv8)
  - Epochs: 100-150
  - Batch size: 16 (adjust based on GPU)
  - IoU threshold: 0.5 for mAP calculation

**Why YOLOv8**: The reference assignment uses YOLOv4-tiny via Darknet. YOLOv8 (via Ultralytics) is more modern with better Python API, easier setup, and comparable performance for research tasks.

### 4. Training Script
**Script**: `scripts/train_yolo.py`

Using Ultralytics YOLO Python API:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load pretrained nano model
results = model.train(
    data='config/shapes.yaml',
    epochs=150,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    project='runs/shapes',
    name='experiment_1'
)
```

**Outputs**: Saved to `runs/shapes/experiment_1/`:
- `weights/best.pt`, `weights/last.pt`
- Training curves: `results.png`, `confusion_matrix.png`
- Per-class metrics: `results.csv`

### 5. Evaluation & Analysis
**Script**: `scripts/evaluate_model.py`

**Metrics to compute**:
1. **Per-category mAP@0.5**: How well YOLO detects each shape category
2. **Per-category precision/recall**: At 0.5 IoU threshold
3. **Confusion matrix**: Which categories get confused (especially synthetic vs. natural)
4. **Detection rate vs. shape scale**: Does performance degrade for smaller shapes?
5. **Curvature complexity correlation**: Do cat1 (max entropy) shapes perform worse than cat4 (constrained)?

**Visualization**:
- Sample predictions grid: 20 test images with bounding boxes color-coded by category
- Precision-Recall curves per category
- mAP bar chart: cat1-5 comparison
- Error analysis: Show false positives/negatives with shape complexity annotation

**Script structure**:
```python
model = YOLO('runs/shapes/experiment_1/weights/best.pt')
metrics = model.val(data='config/shapes.yaml')  # Computes mAP, P, R

# Custom analysis
for category in ['cat1', 'cat2', 'cat3', 'cat4', 'cat5']:
    category_images = filter_test_images_by_category(category)
    results = model.predict(category_images)
    analyze_performance(results, category)
```

### 6. Results Reporting
**Output**: `results/experiment_report.md` + plots in `results/figures/`

**Structure**:
- **Summary table**: mAP@0.5, precision, recall for each category
- **Key findings**:
  - Synthetic (cat1-4) vs. natural (cat5) performance gap
  - Effect of curvature constraints (cat1→cat4) on detection accuracy
  - Common failure modes (shape overlaps, scale extremes)
- **Statistical tests**: One-way ANOVA to test if mAP differences across categories are significant
- **Figures**: 
  - `mAP_by_category.png`
  - `precision_recall_curves.png`
  - `sample_predictions.png` (4×5 grid: one row per category)
  - `confusion_matrix.png`

---

## Critical Files

**To create**:
1. `scripts/create_balanced_dataset.py` — Balanced sampling to JSON
2. `scripts/generate_composite_images.py` — Dataset generation from JSON
3. `scripts/train_yolo.py` — Model training
4. `scripts/evaluate_model.py` — Metrics & analysis
5. `config/shapes.yaml` — Dataset config for YOLO
6. `requirements.txt` — Dependencies (ultralytics, opencv-python, numpy, matplotlib, scipy)

**Generated artifacts**:
1. `dataset/selected_shapes.json` — Balanced shape paths (390 per category)

**Existing to reference**:
- Assignment 3 notebook for Darknet workflow patterns
- `/Shapes/cat{1-5}/` for source PNG images

---

## Verification

**End-to-end test**:
1. Run `create_balanced_dataset.py` → verify JSON has exactly 390 shapes per category
2. Generate 10 sample composite images → verify balanced category distribution by counting shapes
3. Train for 5 epochs on tiny subset → confirm no errors
4. Run evaluation script → verify all metrics compute without errors
5. Visual check: At least 3 sample predictions should have correct bounding boxes

**Success criteria**:
- JSON contains exactly 390 unique paths per category (1950 total)
- Generated dataset has approximately equal category representation (±5%)
- mAP@0.5 > 0.7 overall (YOLO should handle simple silhouettes well)
- Statistical difference between at least one pair of categories (e.g., cat1 vs cat4)
- Report answers: "Does curvature complexity affect detectability?" with quantitative evidence
