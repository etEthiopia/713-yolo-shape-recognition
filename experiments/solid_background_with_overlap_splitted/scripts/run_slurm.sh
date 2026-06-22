#!/bin/bash

# --- SLURM CONFIGURATION ---
#SBATCH --job-name=yolo_overlap
#SBATCH --output=experiments/solid_background_with_overlap_splitted/slurm_output.out
#SBATCH --error=experiments/solid_background_with_overlap_splitted/slurm_error.err
#SBATCH --partition=gpugrp1        # GPU partition (adjust to your cluster)
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --ntasks=1                 # Single task
#SBATCH --cpus-per-task=4          # 4 CPUs for data loading
#SBATCH --mem=12G                  # 12GB memory
#SBATCH --time=24:00:00            # 24 hours max

# --- LOAD MODULES (if needed) ---
# Uncomment and adjust if your cluster requires module loading
# module load python/3.9 cuda/11.8

echo "=========================================="
echo "YOLO Overlap Experiment (SLURM)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job running on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# --- SETUP ---
# Navigate to project directory (where job was submitted from)
cd $SLURM_SUBMIT_DIR || exit 1
echo "Working directory: $(pwd)"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: No virtual environment found (venv/ or .venv/)"
    exit 1
fi

# Verify GPU is available
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Navigate to experiment scripts
cd experiments/solid_background_with_overlap_splitted/scripts || exit 1

# --- PHASE 1: SETUP ---
echo ""
echo "=========================================="
echo "Phase 1: Setup"
echo "=========================================="
if [ ! -f "../metadata/experiment_config.json" ]; then
    python setup.py
    echo "✓ Setup complete"
else
    echo "✓ Setup already done (metadata exists)"
fi

# --- PHASE 2: DATASET CREATION ---
echo ""
echo "=========================================="
echo "Phase 2: Create Balanced Dataset"
echo "=========================================="
if [ ! -f "../dataset/selected_shapes.json" ]; then
    python create_balanced_dataset.py
    echo "✓ Dataset reference created"
else
    echo "✓ Dataset reference already exists"
fi

# --- PHASE 3: IMAGE GENERATION WITH OVERLAP ---
echo ""
echo "=========================================="
echo "Phase 3: Generate Composite Images (with Overlap)"
echo "=========================================="
if [ ! -d "../dataset/images/train" ] || [ -z "$(ls -A ../dataset/images/train 2>/dev/null)" ]; then
    echo "Generating 1,000 training + 250 test images..."
    echo "  - 50% with chain overlaps"
    echo "  - 25-75% visibility constraint"
    echo "  - Overlap metadata saved to dataset/overlap/"
    python generate_composite_images.py
    echo "✓ Images generated"
else
    echo "✓ Images already generated"
    echo "   (To regenerate, delete ../dataset/images/ directory)"
fi

# --- PHASE 4: TRAINING ---
echo ""
echo "=========================================="
echo "Phase 4: Train YOLO Model"
echo "=========================================="
echo "Training with:"
echo "  - Model: YOLOv8-nano"
echo "  - Epochs: 10 (from metadata)"
echo "  - Batch size: 16"
echo "  - Device: Auto-detected (GPU if available)"
echo ""

if [ ! -f "../training/run_1/weights/best.pt" ]; then
    echo "Starting training (this may take 1-2 hours with GPU)..."
    python train_model.py --device cuda:0
    echo "✓ Training complete"
else
    echo "✓ Model already trained"
    echo "   Weights: ../training/run_1/weights/best.pt"
    echo "   (To retrain, delete ../training/run_1/ directory)"
fi

# --- PHASE 5: FIX COLORS ---
echo ""
echo "=========================================="
echo "Phase 5: Fix Training Visualization Colors"
echo "=========================================="
if [ -f "../training/run_1/train_batch0.jpg" ]; then
    echo "Applying experiment color scheme..."
    python fix_training_colors.py
    echo "✓ Colors fixed (train_batch*.jpg now use official colors)"
else
    echo "⊘ No training visualizations found (skipping)"
fi

# --- PHASE 6: EVALUATION ---
echo ""
echo "=========================================="
echo "Phase 6: Evaluate Model"
echo "=========================================="
echo "Running comprehensive evaluation..."
python evaluate_model.py --device cuda:0
echo "✓ Evaluation complete"

# --- COMPLETION ---
echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Results saved to: experiments/solid_background_with_overlap_splitted/"
echo "  - Model weights: training/run_1/weights/best.pt"
echo "  - Training curves: training/run_1/results.csv"
echo "  - Evaluation metrics: evaluation/evaluation_summary.json"
echo "  - Visualizations: evaluation/predictions/"
echo "  - Overlap metadata: dataset/overlap/"
echo ""
echo "To analyze overlap data:"
echo "  cat ../dataset/overlap/train/train_0000.json"
echo "  ls -lh ../dataset/overlap/train/ | wc -l"
echo ""
echo "Compare to baseline:"
echo "  experiments/solid_background/evaluation/evaluation_summary.json"
echo ""
