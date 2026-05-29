#!/bin/bash
# Complete pipeline to run all steps of the YOLO shape recognition experiment

set -e  # Exit on error

PROJECT_DIR="/Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/713-yolo-shape-recognition"
cd "$PROJECT_DIR"

echo "=========================================="
echo "YOLO Shape Recognition - Full Pipeline"
echo "=========================================="

# Activate virtual environment
echo ""
echo "Step 0: Activating virtual environment..."
source venv/bin/activate

# Step 1: Create balanced dataset
echo ""
echo "=========================================="
echo "Step 1: Creating balanced dataset"
echo "=========================================="
python scripts/create_balanced_dataset.py

# Step 2: Generate composite images
echo ""
echo "=========================================="
echo "Step 2: Generating composite images"
echo "=========================================="
python scripts/generate_composite_images.py \
    --num-train 500 \
    --num-test 125 \
    --img-size 640

# Step 3: Train YOLO model
echo ""
echo "=========================================="
echo "Step 3: Training YOLO model"
echo "=========================================="
python scripts/train_yolo.py \
    --model yolov8n.pt \
    --epochs 150 \
    --batch 16 \
    --name experiment_1

# Step 4: Evaluate model
echo ""
echo "=========================================="
echo "Step 4: Evaluating trained model"
echo "=========================================="
python scripts/evaluate_model.py \
    --weights runs/shapes/experiment_1/weights/best.pt \
    --output-dir results

echo ""
echo "=========================================="
echo "✓ Pipeline Complete!"
echo "=========================================="
echo ""
echo "Results available in:"
echo "  - runs/shapes/experiment_1/ (training outputs)"
echo "  - results/ (evaluation metrics and visualizations)"
echo ""
