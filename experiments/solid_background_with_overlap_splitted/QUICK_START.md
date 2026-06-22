# Quick Start: Solid Background with Overlap

Fast reference for running the overlap experiment.

---

## One-Command Execution

```bash
cd experiments/solid_background_with_overlap_splitted/scripts
bash run_experiment.sh
```

**What it does**: Setup → Dataset → Images (with overlap) → Train → Fix colors → Evaluate

---

## SLURM Cluster

```bash
# From project root
sbatch experiments/solid_background_with_overlap_splitted/scripts/run_slurm.sh
```

---

## Manual Steps

```bash
cd experiments/solid_background_with_overlap_splitted/scripts

python setup.py                    # Create directories
python create_balanced_dataset.py  # Use shared 1,560 shapes
python generate_composite_images.py # 1,000 train + 250 test (50% overlap)
python train_model.py              # 10 epochs
python fix_training_colors.py      # Apply color scheme
python evaluate_model.py           # Evaluate
```

---

## Key Parameters

| Setting | Value |
|---------|-------|
| Images | 1,000 train, 250 test |
| Overlap | 50% of images |
| Visibility | 25-75% for occluded shapes |
| Overlap pattern | Chain (A→B→C) |
| Categories | cat1, cat2, cat4, cat5 (no cat3) |
| Epochs | 10 |

---

## Verify Overlap

```bash
# Check first image
cat dataset/overlap/train/train_0000.json | jq '.num_overlapping_shapes, .statistics'

# Count overlap images
ls dataset/overlap/train/*.json | wc -l
# Should be 1,000
```

---

## Results Location

```
experiments/solid_background_with_overlap_splitted/
├── training/run_1/weights/best.pt    # Trained model
├── evaluation/evaluation_summary.json # Metrics
└── dataset/overlap/                   # Overlap metadata (JSON)
```

---

## Compare to Baseline

```bash
# Baseline (no overlap)
cat ../solid_background/evaluation/evaluation_summary.json | jq '.overall.mAP@0.5'

# This experiment (with overlap)
cat evaluation/evaluation_summary.json | jq '.overall.mAP@0.5'
```

Expected: **70-85%** vs baseline **96.9%**

---

## Troubleshooting

**Slow generation**: Reduce `--num-train` for testing  
**Wrong paths**: Run from `experiments/solid_background_with_overlap_splitted/scripts/`  
**Out of memory**: Use `--batch 8` in `train_model.py`
