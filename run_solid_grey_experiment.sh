#!/bin/bash
# Launcher for Solid Grey Background Experiment
# This script runs the complete workflow for the experiment

set -e  # Exit on error

PROJECT_DIR="/Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/713-yolo-shape-recognition"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Solid Grey Background Experiment"
echo "=========================================="
echo ""
echo "Experiment parameters:"
echo "  - Categories: cat1, cat2, cat4, cat5 (excluding cat3)"
echo "  - Shapes per image: 2-7 (random)"
echo "  - Training images: 1,000"
echo "  - Test images: 250"
echo "  - Epochs: 200"
echo "  - Background: Grey (128, 128, 128)"
echo ""
echo "Estimated runtime:"
echo "  - GPU: ~1.5-2 hours"
echo "  - CPU: ~7-9 hours"
echo ""
read -p "Press Enter to start the workflow..."

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Run the workflow
echo ""
echo "=========================================="
echo "Starting workflow..."
echo "=========================================="
echo ""

# Use Claude Code's workflow feature
claude workflow run experiments/workflows/solid_grey_background.js

echo ""
echo "=========================================="
echo "Workflow Complete!"
echo "=========================================="
echo ""
echo "Results are available in:"
echo "  experiments/solid_grey_background/"
echo ""
echo "Key outputs:"
echo "  - Trained model: experiments/solid_grey_background/training/run_1/weights/best.pt"
echo "  - Experiment report: experiments/solid_grey_background/EXPERIMENT_REPORT.md"
echo "  - Evaluation metrics: experiments/solid_grey_background/evaluation/metrics_comprehensive.json"
echo "  - Figures: experiments/solid_grey_background/evaluation/figures/"
echo ""
