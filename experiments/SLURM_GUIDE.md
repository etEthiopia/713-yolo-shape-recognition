# Running Solid Grey Background Experiment on SLURM

This guide explains how to run the solid grey background experiment on a remote GPU cluster using SLURM.

## Files Overview

### SLURM Batch Script
**File**: `slurm_solid_grey_experiment.sh`  
**Location**: Project root

This script runs all 6 phases sequentially:
1. Setup (create directories)
2. Dataset Creation (sample 1,560 shapes from 4 categories)
3. Image Generation (1,000 train + 250 test with 2-7 shapes each)
4. Training (YOLOv8-nano for 200 epochs)
5. Evaluation (comprehensive metrics and visualizations)
6. Documentation (generate report)

### Python Scripts
- `scripts/experiments/phase1_setup.py` - Setup phase
- `scripts/create_balanced_dataset.py` - Dataset creation (modified to support category filtering)
- `scripts/generate_composite_images.py` - Image generation (modified to support 2-7 shapes range)
- `scripts/train_yolo.py` - Training (existing)
- `scripts/experiments/phase5_evaluate_comprehensive.py` - Evaluation (to be created)
- `scripts/experiments/phase6_generate_report.py` - Documentation (to be created)

---

## Step-by-Step Instructions

### 1. Upload Project to Cluster

```bash
# On your local machine
cd /Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/
tar -czf 713-yolo-shape-recognition.tar.gz 713-yolo-shape-recognition/

# Upload to cluster
scp 713-yolo-shape-recognition.tar.gz user@cluster:/path/to/destination/

# SSH into cluster
ssh user@cluster

# Extract
cd /path/to/destination/
tar -xzf 713-yolo-shape-recognition.tar.gz
cd 713-yolo-shape-recognition
```

### 2. Setup Virtual Environment on Cluster

```bash
# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: Make sure the cluster has access to PyPI or use `--find-links` to specify a local package cache if needed.

### 3. Configure SSL Certificates (if needed)

If your cluster requires SSL certificates:

```bash
# Copy your certificates to cluster
scp /Users/dagmawi.wube/cacerts.txt user@cluster:/path/to/713-yolo-shape-recognition/

# Verify path in scripts/ssl_config.py matches
```

Or edit `scripts/ssl_config.py` to use cluster-specific certificate path.

### 4. Submit SLURM Job

```bash
# Make script executable
chmod +x slurm_solid_grey_experiment.sh

# Submit job
sbatch slurm_solid_grey_experiment.sh

# Check job status
squeue -u $USER

# View job ID
```

### 5. Monitor Progress

```bash
# Watch output in real-time
tail -f experiments/solid_grey_background/slurm_output.out

# Check for errors
tail -f experiments/solid_grey_background/slurm_error.err

# Check job status
scontrol show job <JOB_ID>
```

### 6. After Completion

```bash
# Download results to local machine
cd /Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/
scp -r user@cluster:/path/to/713-yolo-shape-recognition/experiments/solid_grey_background ./713-yolo-shape-recognition/experiments/
```

---

## SLURM Configuration Details

### Resource Requirements

```bash
#SBATCH --partition=gpugrp1        # GPU partition
#SBATCH --gres=gpu:1               # 1 GPU
#SBATCH --cpus-per-task=4          # 4 CPUs (for data loading)
#SBATCH --mem=16G                  # 16GB RAM
#SBATCH --time=24:00:00            # 24 hours
```

**Why these settings?**
- **GPU**: Required for training 200 epochs in reasonable time (~1.5-2 hours vs 6-8 hours on CPU)
- **4 CPUs**: Parallel data loading speeds up training
- **16GB RAM**: Image generation creates 1,250 images in memory, training needs batch loading
- **24 hours**: Conservative estimate (actual ~2-3 hours on GPU), allows for queue delays

### Adjust if Needed

**If your cluster has different partitions**:
```bash
# List available partitions
sinfo

# Update in script
#SBATCH --partition=<your-gpu-partition>
```

**If memory is limited**:
- Reduce batch size in phase 4: `--batch 8` instead of `--batch 16`
- This will increase training time but reduce memory usage

**If you want faster completion**:
- Request more GPUs: `#SBATCH --gres=gpu:2`
- But YOLOv8 training script uses only 1 GPU by default

---

## Expected Runtime

| Phase | GPU Time | CPU Time | Memory | Notes |
|-------|----------|----------|--------|-------|
| 1. Setup | <1 min | <1 min | <1GB | Directory creation |
| 2. Dataset | ~1 min | ~1 min | ~2GB | Sample 1,560 shapes |
| 3. Images | ~8 min | ~10 min | ~8GB | Generate 1,250 images |
| 4. Training | ~1.5 hrs | ~6 hrs | ~12GB | 200 epochs |
| 5. Evaluation | ~10 min | ~15 min | ~4GB | Metrics + figures |
| 6. Documentation | ~3 min | ~3 min | ~1GB | Report generation |
| **Total** | **~2 hours** | **~7 hours** | **~16GB peak** | |

---

## Output Files

After successful completion:

```
experiments/solid_grey_background/
├── dataset/
│   ├── selected_shapes.json          # 1,560 shapes
│   ├── images/train/                 # 1,000 JPEG images
│   ├── images/test/                  # 250 JPEG images
│   └── labels/{train,test}/          # YOLO labels
├── config/
│   └── shapes.yaml                   # 4-category config
├── training/
│   └── run_1/
│       ├── weights/best.pt           # ⭐ Trained model
│       ├── results.csv               # Per-epoch metrics
│       └── results.png               # Training curves
├── evaluation/
│   ├── metrics_comprehensive.json    # ⭐ All metrics
│   ├── figures/                      # ⭐ 8-10 PNG figures
│   └── report_evaluation.md
├── metadata/
│   └── experiment_config.json        # All parameters
├── EXPERIMENT_REPORT.md              # ⭐ Main report
├── README.md
├── slurm_output.out                  # SLURM stdout
└── slurm_error.err                   # SLURM stderr
```

Files marked with ⭐ are the most important to download.

---

## Troubleshooting

### Job Fails Immediately

**Check**:
```bash
cat experiments/solid_grey_background/slurm_error.err
```

**Common issues**:
1. **Module not found**: Virtual environment not activated
   - Solution: Ensure `source venv/bin/activate` is in script

2. **Permission denied**: Script not executable
   - Solution: `chmod +x slurm_solid_grey_experiment.sh`

3. **Out of memory**: Batch size too large
   - Solution: Edit script to use `--batch 8` instead of `--batch 16`

### GPU Not Found

**Check**:
```bash
# In slurm_output.out, look for:
# "CUDA available: True"
```

**If False**:
1. Wrong partition: `#SBATCH --partition=` should be a GPU partition
2. No GPU requested: `#SBATCH --gres=gpu:1` should be present
3. CUDA not installed on cluster node

### Training Too Slow

**Check GPU utilization**:
```bash
# SSH to compute node (while job is running)
ssh <compute-node>
nvidia-smi
```

**If GPU utilization is low**:
- Increase batch size: `--batch 32` (if memory allows)
- Check if data loading is bottleneck (increase `--cpus-per-task=8`)

### Phase 5 or 6 Fails

These phases don't exist yet in the codebase! You need to create:
- `scripts/experiments/phase5_evaluate_comprehensive.py`
- `scripts/experiments/phase6_generate_report.py`

See the workflow plan for what these should do, or run phases 1-4 only for now.

---

## Quick Start Template

```bash
# 1. Upload and extract
scp 713-yolo-shape-recognition.tar.gz user@cluster:/scratch/
ssh user@cluster
cd /scratch/
tar -xzf 713-yolo-shape-recognition.tar.gz
cd 713-yolo-shape-recognition

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Submit job
sbatch slurm_solid_grey_experiment.sh

# 4. Monitor
watch -n 10 squeue -u $USER
tail -f experiments/solid_grey_background/slurm_output.out

# 5. Download results (after completion)
scp -r user@cluster:/scratch/713-yolo-shape-recognition/experiments/solid_grey_background ./experiments/
```

---

## Comparison: Local vs SLURM

| Aspect | Local (Workflow) | SLURM (Batch) |
|--------|------------------|---------------|
| **Execution** | `./run_solid_grey_experiment.sh` | `sbatch slurm_solid_grey_experiment.sh` |
| **Orchestration** | Workflow engine (agent-based) | Sequential shell script |
| **Monitoring** | Real-time console output | Log files (tail -f) |
| **GPU** | Auto-detect local GPU | SLURM-allocated GPU |
| **Flexibility** | Easy to modify phases | Edit batch script |
| **Best for** | Interactive development | Batch processing, cluster resources |

---

## Next Steps After Completion

1. **Download results**: `scp` the `experiments/solid_grey_background/` directory
2. **Review report**: Open `EXPERIMENT_REPORT.md`
3. **Check metrics**: View `evaluation/metrics_comprehensive.json`
4. **Visualize**: Open figures in `evaluation/figures/`
5. **Compare to baseline**: See metrics improvement vs previous experiment

---

**Questions?** See main `experiments/README.md` or project `README.md`
