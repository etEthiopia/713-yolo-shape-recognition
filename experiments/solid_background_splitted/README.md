# Solid Background (Splitted) Experiment

Evaluates YOLO detection with strict train/test shape separation on solid grey backgrounds.

## Key Differences from solid_background

1. **Shape Separation**: Train and test use different shapes (80/20 split)
   - Train: 312 shapes per category (1,248 total)
   - Test: 78 shapes per category (312 total)
   - Zero overlap between train and test shapes

2. **Purpose**: Evaluate model generalization to new shape instances

3. **Same split** as other `_splitted` experiments for fair comparison

## Usage

```bash
# Generate images (if needed)
python scripts/generate_composite_images.py

# Train
python scripts/train_model.py

# Evaluate  
python ../../scripts/detailed_evaluation.py . --conf 0.5

# Generate plots
python ../../scripts/generate_all_plots.py .
```

## Comparison

This is the **baseline** for comparison with:
- `solid_background_with_overlap_splitted` (same shapes, with overlap)
- `line_background_with_overlap_splitted` (complex backgrounds)
- `natural_scenes_background_with_overlap_splitted` (real-world)

All use the same train/test shape split for fair comparison.
