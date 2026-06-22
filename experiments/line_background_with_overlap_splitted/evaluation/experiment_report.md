# YOLO Shape Recognition - Experiment Report

## Overview

This report presents the results of training YOLOv8 to recognize synthetic and natural shape silhouettes.

## Overall Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.7982 |
| mAP@0.5:0.95 | 0.6425 |
| Precision | 0.8562 |
| Recall | 0.8185 |

## Per-Category Results

| Category | Type | mAP@0.5 | mAP@0.5:0.95 |
|----------|------|---------|---------------|
| cat1 | Synthetic (unconstrained) | 0.9031 | 0.7423 |
| cat2 | Synthetic (variance matched) | 0.7907 | 0.6311 |
| cat4 | Synthetic (all stats matched) | 0.7181 | 0.5985 |
| cat5 | Natural (animals) | 0.7807 | 0.5981 |

## Key Findings

1. **Synthetic shapes average mAP**: 0.8040
2. **Natural shapes mAP**: 0.7807
3. **Difference**: 0.0233

4. **Best performing category**: cat1 (mAP: 0.9031)
5. **Worst performing category**: cat4 (mAP: 0.7181)

## Visualizations

### Performance by Category
![mAP by Category](figures/mAP_by_category.png)

### Synthetic vs Natural Comparison
![Synthetic vs Natural](figures/synthetic_vs_natural.png)

### Sample Predictions
![Sample Predictions](figures/sample_predictions.png)

## Conclusion

The trained YOLO model successfully detects and classifies shape silhouettes across all categories. Synthetic shapes showed better detection performance (0.8040) compared to natural shapes (0.7807), suggesting that controlled curvature properties may be more easily learned by the model.
