# Solid Grey Background Experiment - Quick Start

## Run the Experiment

```bash
cd /Users/dagmawi.wube/Documents/School/Courses/713_Applied_ML/713-yolo-shape-recognition
./run_solid_grey_experiment.sh
```

That's it! The workflow handles everything.

---

## What It Does

1. **Creates** experiment directory at `experiments/solid_grey_background/`
2. **Samples** 1,560 shapes (390 from cat1, cat2, cat4, cat5)
3. **Generates** 1,250 composite images (1,000 train, 250 test, 2-7 shapes each)
4. **Trains** YOLOv8-nano for 200 epochs
5. **Evaluates** with comprehensive metrics and 8-10 visualizations
6. **Documents** everything in detailed report

---

## Expected Runtime

- **GPU**: 1.5-2 hours
- **CPU**: 7-9 hours

---

## Key Differences from Baseline

| Parameter | Baseline | Solid Grey | Change |
|-----------|----------|------------|--------|
| Categories | 5 | 4 | Exclude cat3 |
| Shapes/img | 2-5 | 2-7 | More complexity |
| Train imgs | 500 | 1,000 | 2× more data |
| Test imgs | 125 | 250 | 2× more data |
| Epochs | 10-150 | 200 | More thorough |

---

## Expected Results

- **Overall mAP@0.5**: > 75% (up from 72.9%)
- **All categories**: > 45% mAP
- **Precision**: > 70%
- **Recall**: > 85%

---

## Where to Find Results

After completion, see:
- **Main report**: `experiments/solid_grey_background/EXPERIMENT_REPORT.md`
- **Metrics**: `experiments/solid_grey_background/evaluation/metrics_comprehensive.json`
- **Figures**: `experiments/solid_grey_background/evaluation/figures/*.png`
- **Model**: `experiments/solid_grey_background/training/run_1/weights/best.pt`

---

## Color Scheme (Consistent Across All Visualizations)

- **Cat1** (Unconstrained): #00AADC (cyan blue)
- **Cat2** (Local Var): #00539B (dark blue)
- **Cat4** (Local Matched): #7F2F8D (purple)
- **Cat5** (Natural): #B01F23 (red)

---

## Need Help?

See detailed documentation:
- `experiments/README.md` - Full experiment guide
- `experiments/WORKFLOW_IMPLEMENTATION_COMPLETE.md` - Implementation details
- `experiments/workflows/solid_grey_background.js` - Workflow source code
