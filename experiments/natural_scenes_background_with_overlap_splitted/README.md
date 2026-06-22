# Natural Scenes Background with Overlap (Shape-Split) Experiment

## Overview

This experiment tests YOLO object detection with:
- **Natural scene backgrounds** (15 pre-loaded grayscale images)
- **Controlled overlap** (50% images with chain overlaps, 25-75% visibility constraint)
- **STRICT shape-level train/test separation** (zero overlap between train and test shape pools)

## Shape-Level Train/Test Separation

This experiment implements **strict shape separation** between training and testing:

- **Train pool**: 312 shapes per category (1,248 total)
  - cat1: 312, cat2: 312, cat4: 312, cat5: 312
- **Test pool**: 78 shapes per category (312 total)
  - cat1: 78, cat2: 78, cat4: 78, cat5: 78
- **Guarantee**: Zero overlap between pools

If shape `cat1/img012.png` appears in any training image, it will NEVER appear in any test image.

### Why This Matters

**Previous experiments** (without shape split):
- Same shapes could appear in both train and test images
- Model may memorize specific shapes → inflated test performance
- Shape-level data leakage

**This experiment** (with shape split):
- Training shapes and test shapes are completely disjoint
- Tests true generalization to unseen shapes
- More realistic evaluation

### Expected Impact

- **Train mAP**: No change (~same as without split)
- **Test mAP**: Expected to be **5-10% lower** (harder task, more realistic)
- **Better measure**: True detection capability on novel shapes

## Categories

- **cat1** (Unconstrained): Shapes with no statistical constraints
- **cat2** (Local Var): Shapes with local variance matching
- **cat4** (Local Matched): Shapes with full local statistics matching
- **cat5** (Natural): Shapes derived from natural silhouettes

**Excluded**: cat3 (skew/kurtosis matched) due to poor performance in baseline experiments

## Directory Structure

```
natural_scenes_background_with_overlap_splitted/
├── background/              # 15 natural scene JPG images (640x640)
├── scripts/
│   ├── setup.py
│   ├── create_balanced_dataset.py   # Creates train/test split
│   ├── generate_composite_images.py  # Uses separate pools
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── run_experiment.sh
│   └── verify_split.py              # Verification script
├── config/
│   └── shapes.yaml
├── dataset/
│   ├── selected_shapes.json         # Nested structure: train/test
│   ├── images/
│   │   ├── train/                   # 1,000 images (train shapes only)
│   │   └── test/                    # 250 images (test shapes only)
│   └── labels/
│       ├── train/
│       └── test/
├── training/
│   └── run_1/
│       ├── weights/
│       │   ├── best.pt
│       │   └── last.pt
│       └── results.csv
├── evaluation/
│   └── metrics.json
└── metadata/
    └── experiment_config.json
```

## Quick Start

```bash
cd experiments/natural_scenes_background_with_overlap_splitted/scripts

# 1. Setup
python setup.py

# 2. Create dataset with train/test split
python create_balanced_dataset.py

# 3. Verify zero overlap
python verify_split.py

# 4. Generate images (uses separate pools)
python generate_composite_images.py --num-train 1000 --num-test 250 --seed 42

# 5. Train
python train_model.py

# 6. Evaluate
python evaluate_model.py
```

Or run everything:

```bash
./run_experiment.sh
```

## Verification

To verify strict shape separation:

```bash
python verify_split.py
```

Expected output:
```
cat1:
  Train: 312 shapes
  Test: 78 shapes
  Overlap: 0 shapes
  ✅ PASSED: Zero overlap

cat2:
  Train: 312 shapes
  Test: 78 shapes
  Overlap: 0 shapes
  ✅ PASSED: Zero overlap

...

✅ ALL CATEGORIES PASSED: Strict train/test separation verified
```

## Parameters

- **Image size**: 640x640
- **Shapes per image**: 2-7 (random)
- **Training images**: 1,000
- **Test images**: 250
- **Overlap percentage**: 50% of images
- **Visibility constraint**: 25-75% for occluded shapes
- **Model**: YOLOv8-nano
- **Epochs**: 50
- **Batch size**: 16
- **Random seed**: 42

## Expected Results

Compared to experiments without shape split:

| Metric | Without Split | With Split (this) | Difference |
|--------|---------------|-------------------|------------|
| Train mAP | ~0.80 | ~0.80 | No change |
| Test mAP | ~0.75 | **~0.65-0.70** | -5 to -10% |

The test mAP drop is expected and desirable - it represents a more realistic measure of the model's ability to detect truly novel shapes.
