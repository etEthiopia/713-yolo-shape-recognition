# YOLO Shape Recognition Experiments

This directory contains multiple experiments studying YOLO's performance on shape recognition under various conditions.

## Experiment List

### 1. Solid Grey Background
**Status**: Ready to run  
**Directory**: `solid_grey_background/`  
**Workflow**: `workflows/solid_grey_background.js`

**Parameters**:
- **Categories**: cat1, cat2, cat4, cat5 (excluding cat3)
- **Rationale**: Previous baseline showed cat3 had poor performance (18.6% mAP). Excluding it establishes a cleaner baseline for future comparisons.
- **Shapes per image**: 2-7 (random, increased from 2-5)
- **Background**: Solid grey (128, 128, 128)
- **Dataset**:
  - Training: 1,000 images (increased from 500)
  - Test: 250 images (increased from 125)
  - Total shapes: 1,560 (390 × 4 categories)
- **Training**: 200 epochs (increased from 150)
- **Expected mAP@0.5**: > 75% (improved from 72.9%)

**Color Scheme** (consistent across all visualizations):
- Cat1 (Unconstrained): `#00AADC` (cyan blue)
- Cat2 (Local Var): `#00539B` (dark blue)
- Cat4 (Local Matched): `#7F2F8D` (purple)
- Cat5 (Natural): `#B01F23` (red)

**Run**:
```bash
# Option 1: Using launcher script
./run_solid_grey_experiment.sh

# Option 2: Direct workflow execution
source venv/bin/activate
claude workflow run experiments/workflows/solid_grey_background.js
```

**Outputs**:
```
solid_grey_background/
├── dataset/                    # Generated dataset
├── config/                     # YOLO configuration
├── training/                   # Model weights and training logs
├── evaluation/                 # Metrics and visualizations
├── metadata/                   # Experiment parameters
├── EXPERIMENT_REPORT.md        # Comprehensive report
└── README.md                   # Reproduction instructions
```

---

## Experiment Workflow Structure

Each experiment follows a 6-phase workflow:

1. **Setup**: Create directory structure and configuration
2. **Dataset Creation**: Filter and sample shapes from source directories
3. **Image Generation**: Generate composite images with YOLO labels
4. **Training**: Train YOLOv8 model with comprehensive logging
5. **Evaluation**: Compute metrics, generate visualizations, run statistical tests
6. **Documentation**: Generate comprehensive reports and documentation

---

## Baseline Experiment (Reference)

**Location**: `../dataset/` (root-level, 5 categories)  
**Results**: `../results/`

**Key findings**:
- Overall mAP@0.5: 72.9%
- Best category: cat1 (67.5%)
- Worst category: cat3 (18.6%) ← excluded in solid_grey_background
- Natural (cat5): 59.5%

---

## Future Experiments

Planned experiments using the same workflow framework:

### 2. Textured Background
- Introduce patterned backgrounds (noise, gradients, textures)
- Compare to solid grey baseline
- Test robustness to background variation

### 3. Varying Shape Counts
- Test 1-3, 4-6, 7-10 shapes per image
- Analyze performance vs. crowding

### 4. Multi-Background Comparison
- Mix of grey, white, black, colored backgrounds
- Study background color effects

### 5. Scale Variation Study
- Test different shape scale ranges
- Analyze performance on small vs large shapes

---

## Common Parameters Across Experiments

### Fixed Parameters
- **Model**: YOLOv8-nano (pretrained on COCO)
- **Image size**: 640×640
- **Batch size**: 16
- **Early stopping**: Patience 50 epochs
- **Random seed**: 42 (reproducibility)
- **Category balance**: Equal sampling across included categories

### Variable Parameters (experiment-specific)
- Background type/color
- Number of shapes per image
- Shape scale range
- Included categories
- Training image count
- Number of epochs

---

## Metrics & Evaluation Standards

All experiments compute:
- **Overall**: mAP@0.5, mAP@0.5:0.95, mAP@0.75, precision, recall, F1
- **Per-category**: All above metrics for each included category
- **Confusion matrix**: N×N for N categories
- **Statistical tests**: ANOVA, confidence intervals, pairwise comparisons
- **Visualizations**: 8-10 publication-quality figures

---

## Reproducibility

Each experiment includes:
- Complete parameter logging in `metadata/experiment_config.json`
- Dataset statistics and sha256 hashes
- Training logs with per-epoch metrics
- README with exact reproduction commands
- Workflow script for full automation

---

## Color Scheme Standard

All experiments use consistent category colors:

| Category | Display Name | Hex Color | RGB | Use Case |
|----------|--------------|-----------|-----|----------|
| cat1 | Unconstrained | `#00AADC` | (0, 170, 220) | Bars, lines, bboxes |
| cat2 | Local (Var) | `#00539B` | (0, 83, 155) | Bars, lines, bboxes |
| cat4 | Local (Matched) | `#7F2F8D` | (127, 47, 141) | Bars, lines, bboxes |
| cat5 | Natural | `#B01F23` | (176, 31, 35) | Bars, lines, bboxes |

*(cat3 excluded from current experiments)*

---

## Contributing New Experiments

To add a new experiment:

1. Create workflow script in `workflows/experiment_name.js`
2. Define experiment parameters in workflow `meta` block
3. Follow 6-phase structure: Setup → Dataset → Images → Training → Evaluation → Documentation
4. Use consistent color scheme for visualizations
5. Include statistical tests and comparisons
6. Generate comprehensive report (2,000-3,000 words)
7. Update this README with experiment description

---

## Questions or Issues

See main project README: `../README.md`
