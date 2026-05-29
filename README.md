# YOLO Shape Recognition Experiment

Comparing YOLO's ability to recognize synthetic vs. natural silhouette shapes across 5 categories.

## Project Structure

```
713-yolo-shape-recognition/
├── Shapes/                      # Source shape images (2048x2048 PNGs)
│   ├── cat1/                   # Synthetic - unconstrained (max entropy)
│   ├── cat2/                   # Synthetic - variance matched
│   ├── cat3/                   # Synthetic - skew/kurtosis matched
│   ├── cat4/                   # Synthetic - all stats matched
│   └── cat5/                   # Natural - animal silhouettes
├── scripts/                     # Python scripts
│   ├── ssl_config.py           # SSL certificate configuration
│   ├── create_balanced_dataset.py
│   ├── generate_composite_images.py
│   ├── train_yolo.py
│   └── evaluate_model.py
├── config/
│   └── shapes.yaml             # YOLO dataset configuration
├── dataset/                     # Generated dataset (created by scripts)
│   ├── selected_shapes.json    # Balanced shape selection
│   ├── images/
│   │   ├── train/              # Training images
│   │   └── test/               # Test images
│   └── labels/
│       ├── train/              # YOLO format labels
│       └── test/
├── runs/                        # Training outputs (created during training)
│   └── shapes/
│       └── experiment_1/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── results/                     # Evaluation outputs (created during eval)
│   ├── evaluation_summary.json
│   ├── experiment_report.md
│   └── figures/
├── venv/                        # Python virtual environment
├── requirements.txt
├── IMPLEMENTATION_PLAN.md       # Detailed implementation plan
├── SETUP.md                     # Setup instructions
└── README.md                    # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Run automated setup
./setup_env.sh

# Activate virtual environment
source venv/bin/activate
```

See [SETUP.md](SETUP.md) for detailed setup instructions.

### 2. Run Complete Pipeline

**Option A: Automated (recommended)**

```bash
./run_all.sh
```

This runs all 4 steps sequentially:
1. Create balanced dataset (390 shapes per category)
2. Generate composite images (500 train, 125 test)
3. Train YOLOv8 model (150 epochs)
4. Evaluate and generate report

**Option B: Manual Step-by-Step**

```bash
# Step 1: Create balanced shape selection
python scripts/create_balanced_dataset.py

# Step 2: Generate composite images
python scripts/generate_composite_images.py \
    --num-train 500 \
    --num-test 125 \
    --img-size 640

# Step 3: Train YOLO model
python scripts/train_yolo.py \
    --model yolov8n.pt \
    --epochs 150 \
    --batch 16

# Step 4: Evaluate model
python scripts/evaluate_model.py \
    --weights runs/shapes/experiment_1/weights/best.pt
```

## Script Details

### 1. create_balanced_dataset.py

Creates balanced sampling of 390 shapes per category.

**Output**: `dataset/selected_shapes.json`

**Usage**:
```bash
python scripts/create_balanced_dataset.py
```

### 2. generate_composite_images.py

Generates composite images with 2-5 shapes each on grey background.

**Output**: 
- `dataset/images/train/*.jpg` (500 images)
- `dataset/images/test/*.jpg` (125 images)
- `dataset/labels/train/*.txt` (YOLO format)
- `dataset/labels/test/*.txt`

**Usage**:
```bash
python scripts/generate_composite_images.py \
    --num-train 500 \
    --num-test 125 \
    --img-size 640 \
    --seed 42
```

**Options**:
- `--num-train`: Number of training images (default: 500)
- `--num-test`: Number of test images (default: 125)
- `--img-size`: Image size in pixels (default: 640)
- `--seed`: Random seed for reproducibility (default: 42)

### 3. train_yolo.py

Trains YOLOv8 model on generated dataset.

**Output**: 
- `runs/shapes/experiment_1/weights/best.pt` (best model)
- `runs/shapes/experiment_1/weights/last.pt` (last epoch)
- Training curves, confusion matrix, metrics

**Usage**:
```bash
python scripts/train_yolo.py \
    --model yolov8n.pt \
    --epochs 150 \
    --batch 16 \
    --device cuda:0
```

**Options**:
- `--model`: Pretrained model (yolov8n/s/m/l/x.pt) - default: yolov8n.pt
- `--epochs`: Training epochs (default: 150)
- `--batch`: Batch size (default: 16)
- `--device`: Device (cuda:0, cpu, or auto-detect)
- `--name`: Experiment name (default: experiment_1)
- `--patience`: Early stopping patience (default: 50)

**Note**: First run downloads pretrained weights (~6MB for nano model)

### 4. evaluate_model.py

Comprehensive evaluation with metrics and visualizations.

**Output**:
- `results/evaluation_summary.json` (metrics)
- `results/experiment_report.md` (full report)
- `results/figures/mAP_by_category.png`
- `results/figures/synthetic_vs_natural.png`
- `results/figures/sample_predictions.png`

**Usage**:
```bash
python scripts/evaluate_model.py \
    --weights runs/shapes/experiment_1/weights/best.pt \
    --conf 0.25 \
    --iou 0.5
```

**Options**:
- `--weights`: Path to trained weights
- `--conf`: Confidence threshold (default: 0.25)
- `--iou`: IoU threshold for evaluation (default: 0.5)

## Experimental Design

### Dataset
- **Categories**: 5 (cat1-4: synthetic, cat5: natural)
- **Balanced sampling**: 390 shapes per category (1950 total)
- **Training images**: 500 composite images
- **Test images**: 125 composite images
- **Shapes per image**: Random 2-5
- **Background**: Grey (RGB: 128, 128, 128)
- **Augmentation**: Random scale (10-30%), rotation (0-360°), position

### Model
- **Architecture**: YOLOv8-nano (lightweight, 3.2M parameters)
- **Input size**: 640×640
- **Pretrained**: COCO weights (transfer learning)
- **Training epochs**: 150 (with early stopping)
- **Batch size**: 16

### Evaluation Metrics
1. **mAP@0.5**: Primary metric (IoU threshold 0.5)
2. **mAP@0.5:0.95**: Strict metric (IoU 0.5 to 0.95)
3. **Per-category metrics**: Precision, recall, mAP
4. **Synthetic vs Natural**: Average performance comparison
5. **Confusion matrix**: Category misclassifications

## Research Questions

1. **How does curvature complexity affect YOLO detection?**
   - Compare cat1 (max entropy) vs cat4 (constrained) performance

2. **Synthetic vs Natural: Which performs better?**
   - Compare average mAP of cat1-4 vs cat5

3. **Are there category confusions?**
   - Analyze confusion matrix for systematic errors

## Expected Runtime

| Step | GPU | CPU |
|------|-----|-----|
| 1. Create balanced dataset | ~5 sec | ~5 sec |
| 2. Generate images (625 total) | ~2 min | ~5 min |
| 3. Train (150 epochs) | ~30 min | ~4 hours |
| 4. Evaluate | ~2 min | ~5 min |
| **Total** | **~35 min** | **~4.5 hours** |

*Note: GPU times assume NVIDIA GPU with CUDA. CPU times are estimates.*

## Troubleshooting

### SSL Certificate Errors
All scripts automatically use company SSL certificates at `/Users/dagmawi.wube/cacerts.txt`. If you see certificate errors, verify this file exists.

### Out of Memory (OOM)
Reduce batch size:
```bash
python scripts/train_yolo.py --batch 8  # or 4
```

### CUDA Not Available
Training will automatically use CPU if GPU not detected. To force CPU:
```bash
python scripts/train_yolo.py --device cpu
```

### Import Errors
Ensure virtual environment is activated:
```bash
source venv/bin/activate
```

## Results Interpretation

After running the full pipeline, check `results/experiment_report.md` for:
- Overall model performance (mAP, precision, recall)
- Per-category breakdown
- Synthetic vs natural comparison
- Sample predictions visualization
- Key findings and conclusions

## References

- **YOLO**: [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- **Shape Data Source**: Elder, Oleskiw, & Fruend (2018) - Natural shape curvature statistics
- **Assignment Reference**: [assignment_3/Assignment3_pro.ipynb](../assignment_3/Assignment3_pro.ipynb)

## License

Academic use only - Course project for 713 Applied ML.
