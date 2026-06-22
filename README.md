# Can YOLO Learn Statistical Curvature Properties of Synthetic Shape Stimuli?

This study investigates whether YOLOv8, a modern object detection architecture, can learn to classify and localize shapes defined solely by their statistical curvature properties. Using maximum-entropy shape stimuli, this research bridges computational vision with neuroscience research on shape perception through a controlled 2×2 factorial experimental design.

## Research Aim

**Central Question**: Can a state-of-the-art object detection model (YOLOv8) learn to distinguish and accurately localize shapes that are defined solely by the statistical properties of their curvature distributions?

This study employs maximum-entropy shape stimuli from Oleskiw et al. (2018) to test whether YOLOv8 can:
1. **Classify** shapes into categories differing only in curvature variance, skewness, and kurtosis
2. **Localize** irregular, non-convex shapes with high IoU (≥ 0.5)
3. **Generalize** under challenging conditions: overlapping shapes and complex backgrounds

## Shape Categories

Four categories representing different levels of statistical constraint on curvature:

- **Category 1 (Unconstrained)**: Purely random closed loops maximizing entropy, highly irregular "scribble-like" shapes
- **Category 2 (Variance-Matched)**: Match 2nd statistical moment (variance) of natural shape curvature
- **Category 4 (All-Matched)**: Match variance, skewness, AND kurtosis of natural curvature distributions
- **Category 5 (Natural)**: Ground-truth animal silhouettes from Hemera Photo-Objects dataset

*Note: Category 3 (skewness+kurtosis matched, variance unconstrained) was excluded from this study.*

## Experimental Design

### 2×2 Factorial Design

**Factor 1: Overlap Condition**
- **No Overlap**: Bounding boxes guaranteed not to intersect (Experiment 1 baseline)
- **With Overlap**: Random placement allows bounding box intersection (Experiments 2-4)

**Factor 2: Background Complexity**
- **Solid Grey**: Uniform grey background (RGB: 128, 128, 128) — minimal complexity
- **Line Patterns**: Random line patterns creating edge clutter — medium complexity
- **Natural Scenes**: Real photographs from LHQ-1024 dataset — maximum complexity

### Four Experiments

| Experiment | Background | Overlap | Purpose |
|------------|-----------|---------|---------|
| **Exp 1** | Solid Grey | No | Baseline performance ceiling |
| **Exp 2** | Solid Grey | Yes | Isolate overlap effect |
| **Exp 3** | Line Patterns | Yes | Test edge clutter robustness |
| **Exp 4** | Natural Scenes | Yes | Test realistic complexity |

## Key Results

- **Baseline Performance**: 98.66% mAP@0.5 (solid background, no overlap)
- **Overlap Penalty**: -17.86% average mAP drop when shapes overlap
- **Background Effect**: -4.98% maximum mAP drop from background complexity
- **Overlap Dominates**: Overlap is 3.6× more impactful than background complexity
- **Constraint Hierarchy**: Tighter curvature constraints suffer 4× more degradation under overlap
- **Surprising Finding**: Natural scenes (79.11%) outperform line backgrounds (76.89%)

## Project Structure

```
713-yolo-shape-recognition/
├── Shapes/                      # Source shape images (2048x2048 PNGs)
│   ├── cat1/                   # Unconstrained (max entropy) - 390 shapes
│   ├── cat2/                   # Variance-matched - 390 shapes
│   ├── cat4/                   # All-matched - 390 shapes
│   └── cat5/                   # Natural (animals) - 390 shapes
│
├── experiments/                 # Experimental results and datasets
│   ├── solid_background_splitted/           # Exp 1: Baseline
│   │   ├── train/ (images + labels)
│   │   ├── test/ (images + labels)
│   │   ├── runs/shapes/weights/best.pt
│   │   └── results/
│   ├── solid_background_with_overlap_splitted/    # Exp 2
│   ├── line_background_with_overlap_splitted/     # Exp 3
│   └── natural_scenes_background_with_overlap_splitted/  # Exp 4
│
├── scripts/                     # Analysis and training scripts
│   ├── train_yolo.py           # YOLOv8 training script
│   ├── evaluate_model.py       # Model evaluation with metrics
│   ├── detailed_evaluation.py  # Comprehensive analysis across all experiments
│   ├── generate_all_plots.py   # Generate all publication figures
│   ├── plot_ap_across_experiments.py  # Cross-experiment comparison
│   └── DETAILED_EVALUATION_README.md  # Evaluation documentation
│
├── figures/                     # Publication-ready figures
│   ├── fig_categories.png      # Shape category examples
│   ├── fig_solid_ap_by_category.jpg
│   ├── fig_ap_across_experiments.jpg  # Main result figure
│   └── [additional visualization files]
│
├── results/                     # Consolidated results
│   ├── evaluation_summary.json
│   └── experiment_report.md
│
├── config/
│   └── shapes.yaml             # YOLO dataset configuration
│
├── main.tex                     # LaTeX paper source
├── references.bib              # Bibliography
├── paper_skeleton.md           # Paper structure template
└── README.md                    # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Reproduce Experiments

Each experiment has its own directory under `experiments/`. To reproduce results:

**Experiment 1 (Baseline): Solid Background, No Overlap**
```bash
cd experiments/solid_background_splitted
python ../../scripts/train_yolo.py --data config.yaml --epochs 200 --batch 16
python ../../scripts/evaluate_model.py --weights runs/shapes/weights/best.pt
```

**Experiment 2: Solid Background with Overlap**
```bash
cd experiments/solid_background_with_overlap_splitted
python ../../scripts/train_yolo.py --data config.yaml --epochs 200 --batch 16
python ../../scripts/evaluate_model.py --weights runs/shapes/weights/best.pt
```

**Experiment 3: Line Background with Overlap**
```bash
cd experiments/line_background_with_overlap_splitted
python ../../scripts/train_yolo.py --data config.yaml --epochs 200 --batch 16
python ../../scripts/evaluate_model.py --weights runs/shapes/weights/best.pt
```

**Experiment 4: Natural Scenes with Overlap**
```bash
cd experiments/natural_scenes_background_with_overlap_splitted
python ../../scripts/train_yolo.py --data config.yaml --epochs 200 --batch 16
python ../../scripts/evaluate_model.py --weights runs/shapes/weights/best.pt
```

### 3. Generate Analysis and Figures

**Comprehensive evaluation across all experiments:**
```bash
python scripts/detailed_evaluation.py
```

**Generate all publication figures:**
```bash
python scripts/generate_all_plots.py
```

**Cross-experiment comparison plot:**
```bash
python scripts/plot_ap_across_experiments.py
```

## Key Files and Their Locations

### Trained Models

Each experiment directory contains trained weights:
```
experiments/<experiment_name>/runs/shapes/weights/
├── best.pt          # Best model checkpoint (by validation mAP)
└── last.pt          # Final epoch checkpoint
```

**Example**: `experiments/solid_background_splitted/runs/shapes/weights/best.pt`

### Datasets

Training and test data for each experiment:
```
experiments/<experiment_name>/
├── train/
│   ├── images/      # 1,000 composite images per experiment
│   └── labels/      # YOLO format labels (.txt)
└── test/
    ├── images/      # 250 composite images per experiment
    └── labels/      # YOLO format labels (.txt)
```

### Results and Metrics

```
experiments/<experiment_name>/results/
├── confusion_matrix.png
├── F1_curve.png
├── P_curve.png
├── PR_curve.png
├── R_curve.png
└── results.csv      # Per-class metrics
```

### Publication Figures

All publication-ready figures are located in `figures/`:
- `fig_categories.png` - Visual examples of the four shape categories
- `fig_solid_ap_by_category.jpg` - Baseline performance by category
- `fig_ap_across_experiments.jpg` - Main comparative result figure
- `fig_solid_shape_count_scatter.jpg` - Predicted vs actual shape counts
- `fig_overlap_TP_FP_example.jpg` - Overlap calculation methodology

### Analysis Scripts

**[scripts/train_yolo.py](scripts/train_yolo.py)**
- Trains YOLOv8-nano on shape datasets
- Uses COCO pretrained weights for transfer learning
- Hyperparameters: 200 epochs, batch size 16, SGD optimizer with momentum

**[scripts/evaluate_model.py](scripts/evaluate_model.py)**
- Evaluates trained model on test set
- Computes mAP@0.5, mAP@0.5:0.95, precision, recall, F1
- Generates confusion matrices and prediction visualizations

**[scripts/detailed_evaluation.py](scripts/detailed_evaluation.py)**
- Comprehensive cross-experiment analysis
- Overlap stratification (no overlap vs. with overlap)
- Per-category performance breakdown across all conditions
- See [scripts/DETAILED_EVALUATION_README.md](scripts/DETAILED_EVALUATION_README.md) for details

**[scripts/generate_all_plots.py](scripts/generate_all_plots.py)**
- Generates all publication-ready figures
- Category-specific performance plots
- Shape count scatter plots for each experiment
- Overlap analysis visualizations

**[scripts/plot_ap_across_experiments.py](scripts/plot_ap_across_experiments.py)**
- Comparative analysis across all 4 experiments
- Tracks each category's performance trajectory
- Identifies curvature constraint hierarchy

## Dataset Specifications

### Per-Experiment Dataset
- **Training images**: 1,000 composite images
- **Test images**: 250 composite images
- **Shapes per image**: Random 2-7 shapes
- **Image size**: 640×640 pixels
- **Format**: YOLO format (.txt labels)

### Shape Distribution
- **Total unique shapes**: 1,560 (390 per category)
- **Training split**: 80% (312 shapes per category)
- **Test split**: 20% (78 shapes per category)
- **Augmentation**: Random scale (10-30% of image), rotation (0-360°), translation

### Background Types
1. **Solid Grey**: RGB (128, 128, 128) uniform background
2. **Line Patterns**: Random line grids with varying orientations
3. **Natural Scenes**: Real photographs from LHQ-1024 dataset

## Model Configuration

- **Architecture**: YOLOv8-nano (3.2M parameters)
- **Input resolution**: 640×640
- **Pretrained weights**: COCO dataset
- **Optimizer**: SGD with momentum (0.937)
- **Learning rate**: 0.01 (initial) with cosine annealing
- **Training epochs**: 200
- **Batch size**: 16
- **Data augmentation**: Mosaic, HSV jitter, rotation, translation, scale, flips

## Evaluation Metrics

1. **mAP@0.5**: Mean Average Precision at IoU threshold 0.5 (primary metric)
2. **mAP@0.5:0.95**: Mean Average Precision averaged across IoU thresholds 0.5-0.95
3. **Precision**: TP / (TP + FP)
4. **Recall**: TP / (TP + FN)
5. **F1 Score**: Harmonic mean of precision and recall
6. **Per-category AP**: Average Precision for each shape category
7. **Confusion Matrix**: Category-level classification errors

## Research Findings Summary

### RQ1: Can YOLO Learn Curvature Statistics?
**Answer: Yes** — YOLOv8 achieves 98.66% mAP@0.5 under ideal conditions, with **zero cross-category classification errors** across all experiments.

### RQ2: How Does Overlap Affect Detection?
- **Overlap penalty**: -17.86% average mAP drop
- **Recall degradation**: 5.3× more severe than precision loss
- **Constraint scaling**: Tightly constrained shapes suffer 4× more degradation than unconstrained shapes

### Key Insights

1. **Overlap Dominates Background**: Overlap causes 3.6× more performance degradation than background complexity
2. **Non-Monotonic Complexity**: Natural scenes (79.11% mAP) unexpectedly outperform line backgrounds (76.89%)
3. **Statistical Alignment Effect**: All-matched shapes improve +9.17% on natural backgrounds vs. line patterns
4. **Perfect Category Separation**: Zero cross-category confusion demonstrates genuine curvature-based learning

## Expected Runtime

| Task | GPU (NVIDIA V100) | CPU |
|------|-------------------|-----|
| Train single experiment (200 epochs) | ~2-3 hours | ~12 hours |
| Evaluate single experiment | ~5 min | ~15 min |
| Generate all plots | ~2 min | ~5 min |
| Complete 4-experiment study | ~10 hours | ~2 days |

## Paper and Documentation

- **Main Paper**: [main.tex](main.tex) — Full LaTeX source
- **Paper Skeleton**: [paper_skeleton.md](paper_skeleton.md) — Structure template
- **Bibliography**: [references.bib](references.bib)
- **Evaluation Guide**: [scripts/DETAILED_EVALUATION_README.md](scripts/DETAILED_EVALUATION_README.md)
- **Plotting Guide**: [scripts/PLOTTING_README.md](scripts/PLOTTING_README.md)

## Connections to Research

This work builds on:
- **Oleskiw et al. (2018)**: Maximum-entropy shape synthesis framework
- **Wube et al. (2026)**: Neural representations of curvature statistics (VSS presentation)
- **CS 713 Assignment 3**: YOLOv4-tiny training on animal categories

The study represents the **first application of maximum-entropy visual stimuli to object detection**, bridging computational vision with neuroscience research on shape perception.

## References

- **Oleskiw, T. D., Nowack, A., & Pomplun, M. (2018)**. "Statistical curvature properties of natural and artificial closed shapes." Vision Research, 149, 26-36.
- **Wube, D., Oleskiw, T. D., & Elder, J. H. (2026)**. "Can neural networks learn statistical curvature properties of visual stimuli?" Vision Sciences Society Annual Meeting.
- **Ultralytics YOLOv8**: [https://docs.ultralytics.com/](https://docs.ultralytics.com/)
- **LHQ-1024 Dataset**: Natural scene backgrounds

## Paper
[Google Drive Link]([https://docs.ultralytics.com/](https://drive.google.com/file/d/13jGpM-6ONYDyiBopzCmPegZ7H5JwX_C7/view?usp=sharing ))


## Citation

If you use this work, please cite:

```bibtex
@article{wube2026yolo,
  title={Can YOLO Learn Statistical Curvature Properties of Synthetic Shape Stimuli?},
  author={Wube, Dagmawi N},
  journal={CS 713 Applied Machine Learning},
  year={2026},
  institution={University of Regina}
}
```

## License

Academic use only — Course project for CS 713 Applied Machine Learning, University of Regina.
