# Solid Background with Overlap Experiment

**Status**: ✅ Ready to run  
**Categories**: cat1, cat2, cat4, cat5 (excluding cat3)  
**Overlap**: 50% images with chain overlaps, 25-75% visibility constraint  
**Created**: 2026-05-29

---

## Overview

This experiment evaluates YOLO's ability to detect and classify shapes under **controlled occlusion conditions**. It extends the baseline `solid_background` experiment by introducing overlapping shapes with strict visibility constraints.

**Key Features**:
- ✅ 50% images with overlaps, 50% without (for direct comparison)
- ✅ Chain overlap pattern (A overlaps B, B overlaps C, etc.)
- ✅ Visibility constraint: Occluded shapes must have 25-75% visible area
- ✅ Per-image overlap metadata (JSON files for analysis)
- ✅ Same training parameters as baseline (enables direct performance comparison)

---

## Quick Start

### Local Execution

```bash
cd experiments/solid_background_with_overlap_splitted/scripts

# Complete pipeline (recommended)
bash run_experiment.sh
```

### SLURM Cluster

```bash
# From project root
sbatch experiments/solid_background_with_overlap_splitted/scripts/run_slurm.sh

# Monitor
tail -f experiments/solid_background_with_overlap_splitted/slurm_output.out
```

### Manual Step-by-Step

```bash
cd experiments/solid_background_with_overlap_splitted/scripts

# 1. Setup
python setup.py

# 2. Dataset reference (uses shared balanced dataset)
python create_balanced_dataset.py

# 3. Generate images with overlap
python generate_composite_images.py

# 4. Train (10 epochs)
python train_model.py

# 5. Fix training visualization colors
python fix_training_colors.py

# 6. Evaluate
python evaluate_model.py
```

---

## Directory Structure

```
solid_background_with_overlap_splitted/
├── scripts/
│   ├── setup.py
│   ├── create_balanced_dataset.py
│   ├── generate_composite_images.py     # 🆕 Overlap logic
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── fix_training_colors.py
│   ├── run_experiment.sh
│   └── run_slurm.sh
├── config/
│   └── shapes.yaml                      # YOLO config (4 classes)
├── dataset/
│   ├── selected_shapes.json             # 1,560 balanced shapes
│   ├── images/
│   │   ├── train/                       # 1,000 images (500 overlap, 500 no-overlap)
│   │   └── test/                        # 250 images (125 overlap, 125 no-overlap)
│   ├── labels/
│   │   ├── train/                       # YOLO format labels
│   │   └── test/
│   └── overlap/                         # 🆕 Overlap metadata
│       ├── train/                       # JSON: train_0000.json, train_0001.json, ...
│       └── test/                        # JSON: test_0000.json, test_0001.json, ...
├── training/
│   └── run_1/                           # Training outputs
├── evaluation/                          # Evaluation outputs
└── metadata/
    └── experiment_config.json           # Experiment parameters + overlap config
```

---

## Overlap Configuration

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Overlap percentage** | 50% | Half of images contain overlaps |
| **Overlap pattern** | Chain | Shapes form overlapping chains (A→B→C) |
| **Visibility constraint** | 25-75% | Occluded shapes must have this much visible area |
| **Applies to** | Occluded shapes | Background shapes (being covered) |
| **Metadata tracking** | Per-image JSON | Detailed overlap statistics |

### Overlap Metadata Format

Each image has a corresponding JSON file with overlap statistics:

```json
{
  "image": "train_0042.jpg",
  "num_shapes": 5,
  "num_overlapping_shapes": 3,
  "shapes": [
    {
      "shape_idx": 0,
      "class_id": 0,
      "category": "cat1",
      "total_area": 15234,
      "visible_area": 15234,
      "occluded_area": 0,
      "visibility_ratio": 1.0,
      "overlaps_with": [],
      "overlap_type": "none"
    },
    {
      "shape_idx": 1,
      "class_id": 1,
      "category": "cat2",
      "total_area": 18456,
      "visible_area": 9228,
      "occluded_area": 9228,
      "visibility_ratio": 0.50,
      "overlaps_with": [0],
      "overlap_type": "partial"
    },
    ...
  ],
  "statistics": {
    "total_overlap_area": 12456,
    "avg_visibility_ratio": 0.85,
    "overlap_percentage": 60.0
  }
}
```

**Fields**:
- `total_area`: Shape area in pixels
- `visible_area`: Non-occluded pixels
- `occluded_area`: Pixels covered by other shapes
- `visibility_ratio`: `visible_area / total_area`
- `overlaps_with`: Indices of shapes this overlaps
- `overlap_type`: `none`, `partial`, or `chain`

---

## Overlap Algorithm

### Chain Overlap Strategy

1. **First shape**: Always placed without overlap
2. **Subsequent shapes**: 50% chance of attempting overlap
   - Try to position so it partially occludes previous shape
   - Constraint: Occluded shape must have 25-75% visible area
   - Max 100 placement attempts
   - If fails, fall back to no-overlap placement
3. **No-overlap fallback**: Uses standard separation (15px margin)

### Visibility Calculation

```
overlap_mask = binary mask of all previously placed shapes

For new shape at position (x, y):
  shape_region = shape's binary mask
  overlap_region = overlap_mask[y:y+h, x:x+w]
  
  occluded_pixels = (shape_region > 0) & (overlap_region > 0)
  visible_pixels = (shape_region > 0) & ~(overlap_region > 0)
  
  visibility_ratio = visible_pixels.sum() / shape_region.sum()
  
  Valid placement: 0.25 ≤ visibility_ratio ≤ 0.75
```

---

## Categories & Class IDs

**CRITICAL**: Sequential class mapping (not formula-based!)

| Category | Class ID | Display Name | Color Hex | RGB |
|----------|----------|--------------|-----------|-----|
| cat1 | 0 | Unconstrained | #00AADC | (0, 170, 220) |
| cat2 | 1 | Local (Var) | #00539B | (0, 83, 155) |
| cat4 | 2 | Local (Matched) | #7F2F8D | (127, 47, 141) |
| cat5 | 3 | Natural | #B01F23 | (176, 31, 35) |

**Note**: cat4 is class 2 (NOT 3!), cat5 is class 3 (NOT 4!)

---

## Parameters

- **Background**: Solid grey (128, 128, 128)
- **Shapes per image**: 2-7 (random)
- **Shape scale**: 0.1-0.3 of image size (same as baseline for fair comparison)
- **Image size**: 640×640
- **Training images**: 1,000 (500 overlap, 500 no-overlap)
- **Test images**: 250 (125 overlap, 125 no-overlap)
- **Epochs**: 10
- **Batch size**: 16
- **Model**: YOLOv8-nano
- **Random seed**: 42

---

## Expected Performance

**Hypothesis**: Overlap will reduce mAP due to occlusion challenges.

**Baseline** (`solid_background`): 96.9% mAP@0.5

**Expected** (`solid_background_with_overlap_splitted`): 70-85% mAP@0.5

**Breakdown**:
- No-overlap images: ~95%+ (similar to baseline)
- Overlap images: ~60-75% (degraded by occlusion)
- **Weighted average**: ~75-85%

**Analysis**:
- Compare overall performance to baseline
- Split metrics by overlap/no-overlap subsets
- Correlate visibility ratio with detection accuracy
- Identify which categories are robust to occlusion

---

## Analyzing Overlap Data

### Check Overlap Distribution

```bash
# Count overlap images
python -c "
import json
from pathlib import Path

overlap_dir = Path('dataset/overlap/train')
total = 0
has_overlap = 0

for json_file in overlap_dir.glob('*.json'):
    data = json.load(open(json_file))
    total += 1
    if data['num_overlapping_shapes'] > 0:
        has_overlap += 1

print(f'Overlap percentage: {has_overlap/total*100:.1f}%')
print(f'Target: 50%')
"
```

### Validate Visibility Constraints

```bash
# Check all occluded shapes meet 25-75% constraint
python -c "
import json
from pathlib import Path

overlap_dir = Path('dataset/overlap/train')
violations = 0
total_occluded = 0

for json_file in overlap_dir.glob('*.json'):
    data = json.load(open(json_file))
    for shape in data['shapes']:
        if shape['overlap_type'] != 'none':
            total_occluded += 1
            vis = shape['visibility_ratio']
            if not (0.25 <= vis <= 0.75):
                violations += 1
                print(f'{json_file.name}: shape {shape[\"shape_idx\"]} has {vis:.2%} visibility')

print(f'Total occluded shapes: {total_occluded}')
print(f'Constraint violations: {violations}')
"
```

### Visualize Overlap Statistics

```bash
# Average visibility per image
python -c "
import json
from pathlib import Path

overlap_dir = Path('dataset/overlap/train')
visibilities = []

for json_file in overlap_dir.glob('*.json'):
    data = json.load(open(json_file))
    visibilities.append(data['statistics']['avg_visibility_ratio'])

import numpy as np
print(f'Mean visibility: {np.mean(visibilities):.2%}')
print(f'Min visibility: {np.min(visibilities):.2%}')
print(f'Max visibility: {np.max(visibilities):.2%}')
"
```

---

## Comparison to Baseline

| Aspect | Baseline (solid_background) | This Experiment |
|--------|----------------------------|-----------------|
| Overlap | None (0%) | 50% images with overlap |
| Visibility | 100% | 25-75% for occluded shapes |
| Pattern | Separated shapes | Chain overlaps |
| Metadata | None | Per-image JSON |
| Expected mAP@0.5 | 96.9% | 70-85% |

**Use case**: Baseline establishes upper-bound performance (no occlusion). This experiment tests robustness to realistic occlusion scenarios.

---

## Verification Checklist

Before running full experiment:

- [ ] ✅ Setup complete (`metadata/experiment_config.json` exists)
- [ ] ✅ Dataset reference created (`dataset/selected_shapes.json`)
- [ ] ✅ Images generated (1,250 total)
- [ ] ✅ Overlap metadata exists (`dataset/overlap/train/*.json`)
- [ ] ✅ 50% ± 5% images have overlaps
- [ ] ✅ All occluded shapes have 25-75% visibility
- [ ] ✅ Labels use only class IDs 0-3 (no class 4!)

After training:

- [ ] Model converges (>50% mAP@0.5)
- [ ] Training visualizations use correct colors
- [ ] Evaluation metrics computed for all 4 categories
- [ ] Comparison to baseline documented

---

## Troubleshooting

**Problem**: Overlap generation slow  
**Cause**: 100 attempts per shape to find valid overlap position  
**Fix**: Reduce `max_attempts` in `generate_composite_images.py` (line 234)

**Problem**: No overlaps in generated images  
**Cause**: Constraints too strict for small shapes  
**Fix**: Widen visibility range to 0.15-0.85 (edit `target_visibility` in code)

**Problem**: Training similar to baseline  
**Cause**: Model may be learning from visible parts only  
**Analysis**: Check if overlap images have lower mAP in evaluation split

---

## Notes

- **Excluded category**: cat3 (for consistency with baseline)
- **Shared dataset**: Uses `dataset/balanced_shapes.json` (all experiments use this)
- **Self-contained**: All scripts in `experiments/solid_background_with_overlap_splitted/scripts/`
- **Relative paths**: All paths relative to project root for portability
- **Deterministic**: Random seed 42 ensures reproducibility

---

**Created**: 2026-05-29  
**Author**: Dagmawi Wube  
**Course**: 713 Applied ML  
**Baseline**: experiments/solid_background
