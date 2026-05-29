# Quick Start Guide

Run the complete YOLO shape recognition experiment in 3 commands.

## Prerequisites

- Python 3.11+ installed
- SSL certificate at `/Users/dagmawi.wube/cacerts.txt`
- ~2GB disk space for dataset and models
- GPU recommended (CPU works but slower)

## Step 1: Setup (One-time)

```bash
cd /Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/713-yolo-shape-recognition
./setup_env.sh
```

**Time**: ~2 minutes  
**Output**: Virtual environment with all dependencies installed

## Step 2: Run Experiment

```bash
source venv/bin/activate
./run_all.sh
```

**Time**: 
- GPU: ~35 minutes
- CPU: ~4.5 hours

**What it does**:
1. Creates balanced dataset (390 shapes × 5 categories)
2. Generates 625 composite images (500 train, 125 test)
3. Trains YOLOv8-nano model (150 epochs)
4. Evaluates model and generates report

## Step 3: View Results

```bash
# View evaluation report
cat results/experiment_report.md

# View summary JSON
cat results/evaluation_summary.json

# Open visualizations
open results/figures/mAP_by_category.png
open results/figures/synthetic_vs_natural.png
open results/figures/sample_predictions.png
```

## Manual Step-by-Step (Alternative)

If you prefer to run each step individually:

```bash
# Activate environment
source venv/bin/activate

# Step 1: Create balanced dataset
python scripts/create_balanced_dataset.py
# Output: dataset/selected_shapes.json

# Step 2: Generate composite images
python scripts/generate_composite_images.py --num-train 500 --num-test 125
# Output: dataset/images/ and dataset/labels/

# Step 3: Train model
python scripts/train_yolo.py --epochs 150 --batch 16
# Output: runs/shapes/experiment_1/weights/best.pt

# Step 4: Evaluate
python scripts/evaluate_model.py --weights runs/shapes/experiment_1/weights/best.pt
# Output: results/
```

## Common Adjustments

### Use CPU instead of GPU
```bash
python scripts/train_yolo.py --device cpu
```

### Reduce memory usage
```bash
python scripts/train_yolo.py --batch 8
```

### Quick test run (fewer epochs)
```bash
python scripts/train_yolo.py --epochs 50
```

### Larger model (better accuracy)
```bash
python scripts/train_yolo.py --model yolov8s.pt
```

### More training data
```bash
python scripts/generate_composite_images.py --num-train 1000 --num-test 250
```

## Troubleshooting

### "SSL certificate error"
Verify certificate file exists:
```bash
ls -l /Users/dagmawi.wube/cacerts.txt
```

### "ModuleNotFoundError"
Activate the virtual environment:
```bash
source venv/bin/activate
```

### "Out of memory"
Reduce batch size:
```bash
python scripts/train_yolo.py --batch 4
```

### Training too slow on CPU
Use a smaller number of epochs for testing:
```bash
python scripts/train_yolo.py --epochs 10
```

## File Locations

| What | Where |
|------|-------|
| Balanced shape selection | `dataset/selected_shapes.json` |
| Training images | `dataset/images/train/` |
| Test images | `dataset/images/test/` |
| Labels (YOLO format) | `dataset/labels/train/` and `dataset/labels/test/` |
| Trained model | `runs/shapes/experiment_1/weights/best.pt` |
| Training curves | `runs/shapes/experiment_1/results.png` |
| Evaluation metrics | `results/evaluation_summary.json` |
| Full report | `results/experiment_report.md` |
| Visualizations | `results/figures/*.png` |

## Expected Results

The model should achieve:
- **Overall mAP@0.5**: 0.70 - 0.90 (target: >0.70)
- **Per-category mAP**: Varies by curvature complexity
- **Training time**: 30-60 min on GPU, 3-5 hours on CPU
- **Dataset size**: ~50MB for 625 images

## Next Steps After Results

1. **Analyze the report**: Read `results/experiment_report.md`
2. **Answer research questions**:
   - Does curvature complexity affect detection?
   - Synthetic vs natural performance?
   - Category confusion patterns?
3. **Experiment further**:
   - Try different model sizes (yolov8s, yolov8m)
   - Adjust training data amount
   - Modify image generation parameters

## Documentation

- **Full README**: [README.md](README.md)
- **Setup Guide**: [SETUP.md](SETUP.md)
- **Implementation Plan**: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **Implementation Status**: [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)

---

**Ready to start?** Run `./setup_env.sh` to begin!
