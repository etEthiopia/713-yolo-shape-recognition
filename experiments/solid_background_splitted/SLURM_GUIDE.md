# SLURM Execution Guide

This guide explains how to run the solid_background_splitted experiment on a SLURM cluster with GPU support.

---

## Prerequisites

1. **SLURM cluster access** with GPU partition
2. **Virtual environment** with all dependencies installed
3. **Shapes dataset** uploaded to the cluster at `Shapes/cat1/`, `Shapes/cat2/`, etc.
4. **Shared balanced dataset** at `dataset/balanced_shapes.json`

---

## Quick Start

```bash
# 1. Navigate to project root on cluster
cd /path/to/713-yolo-shape-recognition

# 2. Submit job to SLURM
sbatch experiments/solid_background_splitted/scripts/run_slurm.sh
```

That's it! The job will:
- Auto-detect GPU
- Run all phases (setup → dataset → images → train → evaluate)
- Save results to `experiments/solid_background_splitted/`

---

## Configuration

### Adjust SLURM Resources

Edit [scripts/run_slurm.sh](scripts/run_slurm.sh) `#SBATCH` directives:

```bash
#SBATCH --partition=gpugrp1        # Your GPU partition name
#SBATCH --gres=gpu:1               # Number of GPUs (1 is sufficient)
#SBATCH --cpus-per-task=4          # CPUs for data loading
#SBATCH --mem=12G                  # Memory (12GB is safe)
#SBATCH --time=24:00:00            # Max runtime (adjust if needed)
```

**Timing estimates**:
- Image generation: ~5-10 minutes
- Training (200 epochs): ~1-2 hours with GPU, ~6-8 hours with CPU
- Evaluation: ~5-10 minutes
- **Total**: ~2-3 hours with GPU

### Adjust Training Parameters

To change epochs, batch size, etc., modify the metadata:

```bash
# Before submitting, edit:
vim experiments/solid_background_splitted/metadata/experiment_config.json

# Change "epochs": 200 to your desired value
```

Or override in the SLURM script directly:

```bash
# In run_slurm.sh, change:
python train_model.py --device cuda:0
# To:
python train_model.py --device cuda:0 --epochs 100 --batch 8
```

---

## Monitoring Your Job

### Check Job Status

```bash
# List your jobs
squeue -u $USER

# Job details
scontrol show job <JOB_ID>
```

### View Live Output

```bash
# Training progress
tail -f experiments/solid_background_splitted/slurm_output.out

# Errors (if any)
tail -f experiments/solid_background_splitted/slurm_error.err
```

### Cancel Job

```bash
scancel <JOB_ID>
```

---

## Results

After job completes, results are in `experiments/solid_background_splitted/`:

```bash
# Training outputs
ls experiments/solid_background_splitted/training/run_1/weights/
# → best.pt, last.pt

# Evaluation metrics
cat experiments/solid_background_splitted/evaluation/evaluation_summary.json

# Visualizations
ls experiments/solid_background_splitted/evaluation/predictions/
```

---

## Troubleshooting

### No GPU Detected

**Symptom**: `CUDA available: False`

**Fix**: Check partition and GPU request:
```bash
#SBATCH --partition=gpugrp1    # Verify this is your GPU partition
#SBATCH --gres=gpu:1           # Request at least 1 GPU
```

### Virtual Environment Not Found

**Symptom**: `ERROR: No virtual environment found`

**Fix**: Create venv on cluster:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Out of Memory

**Symptom**: `CUDA out of memory` or job killed

**Fix**: Reduce batch size:
```bash
# In run_slurm.sh, change:
python train_model.py --device cuda:0 --batch 8  # Reduce from 16 to 8
```

Or request more memory:
```bash
#SBATCH --mem=24G  # Increase from 12G
```

### Module Load Required

Some clusters require loading modules first:

```bash
# Uncomment in run_slurm.sh:
module load python/3.9 cuda/11.8
```

### Images Already Generated

If you need to regenerate images:
```bash
# Delete existing images before submitting
rm -rf experiments/solid_background_splitted/dataset/images/
```

Or comment out the image generation phase in `run_slurm.sh`.

---

## Running Individual Phases

To run only specific phases (e.g., just training):

```bash
# 1. Comment out unwanted phases in run_slurm.sh
#    (Comment out phases 1-3 if data is ready)

# 2. Submit job
sbatch experiments/solid_background_splitted/scripts/run_slurm.sh
```

Or run interactively on a GPU node:

```bash
# Request interactive GPU session
srun --partition=gpugrp1 --gres=gpu:1 --mem=12G --time=4:00:00 --pty bash

# Navigate and activate venv
cd /path/to/713-yolo-shape-recognition
source venv/bin/activate

# Run specific phase
cd experiments/solid_background_splitted/scripts
python train_model.py --device cuda:0
```

---

## File Transfer

### Upload to Cluster

```bash
# From your local machine
scp -r 713-yolo-shape-recognition/ username@cluster.edu:/path/to/destination/
```

### Download Results

```bash
# From your local machine
scp -r username@cluster.edu:/path/to/713-yolo-shape-recognition/experiments/solid_background_splitted/evaluation/ ./local_results/
```

---

## Checklist Before Submitting

- [ ] Virtual environment created and dependencies installed
- [ ] Shapes dataset uploaded (`Shapes/cat1/`, `cat2/`, `cat4/`, `cat5/`)
- [ ] Shared balanced dataset exists (`dataset/balanced_shapes.json`)
- [ ] SLURM partition name correct in `run_slurm.sh`
- [ ] Output directory writable
- [ ] Sufficient disk space (~5GB for images + models)

---

## Example Session

```bash
# SSH to cluster
ssh username@cluster.edu

# Navigate to project
cd ~/713-yolo-shape-recognition

# Verify setup
ls Shapes/  # Should show cat1, cat2, cat4, cat5
ls dataset/balanced_shapes.json  # Should exist

# Activate venv
source venv/bin/activate

# Verify dependencies
python -c "from ultralytics import YOLO; print('✓ YOLO installed')"

# Submit job
sbatch experiments/solid_background_splitted/scripts/run_slurm.sh
# → Submitted batch job 123456

# Monitor
squeue -u $USER
tail -f experiments/solid_background_splitted/slurm_output.out

# Wait for completion (1-3 hours)

# Check results
cat experiments/solid_background_splitted/evaluation/evaluation_summary.json
ls experiments/solid_background_splitted/training/run_1/weights/best.pt
```

---

**Created**: 2026-05-29  
**Author**: Dagmawi Wube
