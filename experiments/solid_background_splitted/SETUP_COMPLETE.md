# Setup Complete: solid_background_splitted

## ✓ Dataset Generation Complete

Generated: May 31, 2026

### Dataset Statistics

**Images:**
- Training: 1,000 images
- Test: 250 images
- Image size: 640x640 pixels

**Shape Instances:**
- Total: 4,320 instances across 1,250 images
- cat1 (Unconstrained): 1,056 instances (24.4%)
- cat2 (Local: Variance): 1,078 instances (25.0%)
- cat4 (Local: Match All): 1,097 instances (25.4%)
- cat5 (Natural): 1,089 instances (25.2%)

**Shape Sources:**
- Train shapes: 1,248 shapes (312 per category)
- Test shapes: 312 shapes (78 per category)
- **Zero overlap** between train and test shape sets

**Sample Paths:**
- Train example: `Shapes/cat1/img201.png`
- Test example: `Shapes/cat1/img386.png` (different from train!)

### Key Features

1. **Strict Shape Separation**: Train and test use completely different shapes
2. **Solid Grey Background**: RGB(128, 128, 128) - eliminates background complexity
3. **No Overlap**: Shapes placed with 15px minimum separation
4. **Balanced Distribution**: All 4 categories equally represented (~25% each)
5. **Correct Class IDs**: 0-3 only (no class 4 bug!)

### Same Split as Other Experiments

This experiment uses the **exact same train/test shape split** as:
- `solid_background_with_overlap_splitted`
- `line_background_with_overlap_splitted`
- `natural_scenes_background_with_overlap_splitted`

This ensures fair comparison across different experimental conditions.

## Next Steps

### 1. Train Model
```bash
cd experiments/solid_background_splitted
python scripts/train_model.py
```

Expected training time:
- GPU: ~30-45 minutes (100 epochs)
- CPU: ~4-6 hours

### 2. Evaluate Model
```bash
python ../../scripts/detailed_evaluation.py . --conf 0.5
```

Generates:
- `detailed_evaluation/per_image_results.json`
- `detailed_evaluation/summary_stats.json`

### 3. Generate Plots
```bash
python ../../scripts/generate_all_plots.py .
```

Creates 5 visualization plots in `detailed_evaluation_stats_and_plots/`

### 4. Compare with Other Experiments

After training and evaluation, compare:
- **vs solid_background_with_overlap_splitted**: Impact of shape overlap
- **vs line_background_with_overlap_splitted**: Impact of line backgrounds
- **vs natural_scenes_background_with_overlap_splitted**: Impact of natural backgrounds

## Verification

All dataset files present:
- ✓ 1,000 train images
- ✓ 1,000 train labels
- ✓ 250 test images
- ✓ 250 test labels
- ✓ selected_shapes.json (train/test split)
- ✓ Training scripts updated
- ✓ Configuration files updated

## Purpose

This experiment serves as the **baseline** for evaluating:
1. Model's ability to generalize to unseen shape instances
2. Impact of simple vs. complex backgrounds (when compared to line/natural variants)
3. Impact of shape overlap (when compared to overlap_splitted variant)

The solid grey background and no-overlap design make this the simplest scenario, isolating the shape generalization challenge.
