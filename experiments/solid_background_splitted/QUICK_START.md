# Quick Start Guide

Fast reference for running the solid_background_splitted experiment.

---

## Local Execution

```bash
cd experiments/solid_background_splitted/scripts

# Complete pipeline (recommended)
bash run_experiment.sh
```

**What it does**:
1. ✅ Setup experiment structure
2. ✅ Create balanced dataset (1,560 shapes)
3. ✅ Generate images (1,000 train + 250 test)
4. ✅ Train YOLO (200 epochs)
5. ✅ **Fix training visualization colors**
6. ✅ Evaluate model

---

## SLURM Cluster

```bash
# From project root
sbatch experiments/solid_background_splitted/scripts/run_slurm.sh

# Monitor
tail -f experiments/solid_background_splitted/slurm_output.out
```

See [SLURM_GUIDE.md](SLURM_GUIDE.md) for cluster setup.

---

## Manual Step-by-Step

```bash
cd experiments/solid_background_splitted/scripts

# 1. Setup
python setup.py

# 2. Dataset
python create_balanced_dataset.py

# 3. Images
python generate_composite_images.py

# 4. Train (1-2 hours with GPU)
python train_model.py

# 5. Fix colors ⭐ IMPORTANT
python fix_training_colors.py

# 6. Evaluate
python evaluate_model.py
```

---

## Key Files

| File | Purpose |
|------|---------|
| `train_model.py` | Train YOLO (wrapper with metadata defaults) |
| `fix_training_colors.py` | Apply color scheme to training visualizations |
| `evaluate_model.py` | Evaluate model (wrapper with metadata) |
| `run_experiment.sh` | Complete pipeline (local) |
| `run_slurm.sh` | Complete pipeline (SLURM) |

---

## Results Location

```
experiments/solid_background_splitted/
├── training/run_1/
│   ├── weights/best.pt           # Trained model
│   ├── train_batch*.jpg          # Training visualizations (after fix)
│   └── results.csv               # Training metrics
└── evaluation/
    ├── evaluation_summary.json   # Metrics
    └── predictions/              # Sample predictions
```

---

## Common Tasks

### Change Training Epochs

```bash
# Edit metadata (affects all runs using train_model.py)
vim metadata/experiment_config.json
# Change "epochs": 200 to desired value

# Or override for single run
python train_model.py --epochs 100
```

### Retrain from Scratch

```bash
# Delete previous training
rm -rf ../training/run_1/

# Train again
python train_model.py
```

### Fix Colors on Existing Training

```bash
# If you already trained but need to fix colors
python fix_training_colors.py
```

### Generate More Training Batches

```bash
# Default: 3 batches
python fix_training_colors.py --num-batches 10
```

---

## Troubleshooting

**Problem**: Training visualizations have wrong colors  
**Fix**: Run `python fix_training_colors.py`

**Problem**: Out of memory during training  
**Fix**: Reduce batch size: `python train_model.py --batch 8`

**Problem**: Training takes too long  
**Fix**: Reduce epochs: `python train_model.py --epochs 50`

---

## Color Scheme

| Category | Color | Hex |
|----------|-------|-----|
| cat1 (Unconstrained) | Cyan | `#00AADC` |
| cat2 (Local Var) | Dark Blue | `#00539B` |
| cat4 (Local Matched) | Purple | `#7F2F8D` |
| cat5 (Natural) | Red | `#B01F23` |

Colors defined in: `metadata/experiment_config.json`

---

## More Info

- [README.md](README.md) - Full documentation
- [COLOR_SCHEME_GUIDE.md](COLOR_SCHEME_GUIDE.md) - Color usage details
- [SLURM_GUIDE.md](SLURM_GUIDE.md) - Cluster execution
