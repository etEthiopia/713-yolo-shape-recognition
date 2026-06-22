#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=yolo_solid_grey_ev
#SBATCH --output=experiments/solid_grey_background/slurm_output_ev.out
#SBATCH --error=experiments/solid_grey_background/slurm_error_ev.err
#SBATCH --partition=gpugrp1        # GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # Single task
#SBATCH --cpus-per-task=4          # 4 CPUs for data loading
#SBATCH --mem=12G                  # 16GB memory (for image generation and training)
#SBATCH --time=24:00:00            # 24 hours (plenty of time for 200 epochs)

# --- LOAD MODULES ---
# Uncomment if your cluster requires module loading
# module load python/3.9 cuda/11.8

echo "=========================================="
echo "YOLO Solid Grey Background Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job running on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# --- SETUP ---
# Navigate to project directory
cd $SLURM_SUBMIT_DIR || exit 1

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || source .venv/bin/activate

# Verify GPU is available
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# echo ""
# echo "=========================================="
# echo "Phase 1: Setup"
# echo "=========================================="
# python scripts/experiments/phase1_setup.py

# echo ""
# echo "=========================================="
# echo "Phase 2: Dataset Creation"
# echo "=========================================="
# python scripts/create_balanced_dataset.py \
#     --shapes-dir Shapes \
#     --output-json experiments/solid_grey_background/dataset/selected_shapes.json \
#     --samples-per-category 390 \
#     --categories cat1 cat2 cat4 cat5

# echo ""
# echo "=========================================="
# echo "Phase 3: Image Generation"
# echo "=========================================="
# python scripts/generate_composite_images.py \
#     --shapes-json experiments/solid_grey_background/dataset/selected_shapes.json \
#     --output-dir experiments/solid_grey_background/dataset \
#     --num-train 1000 \
#     --num-test 250 \
#     --img-size 640 \
#     --shapes-range 2 7 \
#     --seed 42

# echo ""
# echo "=========================================="
# echo "Phase 4: Training (200 epochs)"
# echo "=========================================="
# python scripts/train_yolo.py \
#     --config experiments/solid_grey_background/config/shapes.yaml \
#     --model yolov8n.pt \
#     --epochs 100 \
#     --batch 16 \
#     --img-size 640 \
#     --project experiments/solid_grey_background/training \
#     --name run_1 \
#     --patience 50

echo ""
echo "=========================================="
echo "Phase 5: Evaluation"
echo "=========================================="
python scripts/experiments/phase5_evaluate_comprehensive.py \
    --weights experiments/solid_grey_background/training/run_1/weights/best.pt \
    --config experiments/solid_grey_background/config/shapes.yaml \
    --output-dir experiments/solid_grey_background/evaluation \
    --categories cat1 cat2 cat4 cat5

echo ""
echo "=========================================="
echo "Phase 6: Documentation"
echo "=========================================="
python scripts/experiments/phase6_generate_report.py \
    --experiment-dir experiments/solid_grey_background

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: experiments/solid_grey_background/"
echo "Main report: experiments/solid_grey_background/EXPERIMENT_REPORT.md"
echo "Model weights: experiments/solid_grey_background/training/run_1/weights/best.pt"
echo "Evaluation: experiments/solid_grey_background/evaluation/"
echo ""
