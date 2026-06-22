#!/bin/bash
#
# Complete solid_background_with_overlap_splitted experiment pipeline
#
# Orchestrates all steps from setup through evaluation.
#

set -e  # Exit on error

# Get script directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Solid Background with Overlap - Pipeline"
echo "=========================================="
echo ""
echo "Experiment: solid_background_with_overlap_splitted"
echo "Categories: cat1, cat2, cat4, cat5 (excluding cat3)"
echo "Overlap: 50% images with chain overlaps (25-75% visibility)"
echo "Output: $(dirname "$SCRIPT_DIR")"
echo ""

# Step 1: Setup (if not already done)
echo "----------------------------------------"
echo "Step 1: Setup"
echo "----------------------------------------"
if [ ! -f "../metadata/experiment_config.json" ]; then
    echo "Running setup.py..."
    python setup.py
    echo "✓ Setup complete"
else
    echo "✓ Setup already done (metadata exists)"
fi
echo ""

# Step 2: Create dataset reference
echo "----------------------------------------"
echo "Step 2: Create Balanced Dataset"
echo "----------------------------------------"
if [ ! -f "../dataset/selected_shapes.json" ]; then
    echo "Running create_balanced_dataset.py..."
    python create_balanced_dataset.py
    echo "✓ Dataset reference created"
else
    echo "✓ Dataset reference already exists"
fi
echo ""

# Step 3: Generate images with overlap
echo "----------------------------------------"
echo "Step 3: Generate Composite Images (with Overlap)"
echo "----------------------------------------"
if [ ! -d "../dataset/images/train" ] || [ -z "$(ls -A ../dataset/images/train 2>/dev/null)" ]; then
    echo "Generating 1,000 training + 250 test images..."
    echo "  - 50% with chain overlaps"
    echo "  - 25-75% visibility constraint"
    python generate_composite_images.py
    echo "✓ Images generated"
else
    echo "✓ Images already generated"
    echo "   (To regenerate, delete ../dataset/images/ directory)"
fi
echo ""

# Step 4: Train
echo "----------------------------------------"
echo "Step 4: Train YOLO Model"
echo "----------------------------------------"
if [ ! -f "../training/run_1/weights/best.pt" ]; then
    echo "Running train_model.py..."
    echo "(This may take 1-2 hours with GPU, 6-8 hours with CPU)"
    python train_model.py
    echo "✓ Training complete"
else
    echo "✓ Model already trained"
    echo "   (To retrain, delete ../training/run_1/ directory)"
fi
echo ""

# Step 5: Fix Training Colors
echo "----------------------------------------"
echo "Step 5: Fix Training Visualization Colors"
echo "----------------------------------------"
if [ -f "../training/run_1/train_batch0.jpg" ]; then
    echo "Applying experiment color scheme to training visualizations..."
    python fix_training_colors.py
    echo "✓ Colors fixed"
else
    echo "⊘ No training visualizations found (skipping)"
fi
echo ""

# Step 6: Evaluate
echo "----------------------------------------"
echo "Step 6: Evaluate Model"
echo "----------------------------------------"
echo "Running evaluate_model.py..."
python evaluate_model.py
echo "✓ Evaluation complete"
echo ""

echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Training:   $(dirname "$SCRIPT_DIR")/training/run_1/"
echo "  Evaluation: $(dirname "$SCRIPT_DIR")/evaluation/"
echo "  Overlap metadata: $(dirname "$SCRIPT_DIR")/dataset/overlap/"
echo ""
echo "Next steps:"
echo "  - Review metrics:        cat ../evaluation/evaluation_summary.json"
echo "  - View visualizations:   open ../evaluation/predictions/"
echo "  - Analyze overlap data:  ls ../dataset/overlap/train/"
echo "  - Compare to baseline:   Compare with solid_background experiment"
echo ""
