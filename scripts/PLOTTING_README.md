# mAP Plotting Scripts

## Overview

Publication-quality plotting scripts for YOLO evaluation results with per-category metrics and SEM (Standard Error of the Mean).

## Scripts

### plot_map_by_category.py

Creates bar charts from detailed evaluation results with:
- Per-category performance (Unconstrained, Local: Variance, Local: All, Natural)
- SEM error bars calculated from per-image metrics
- Publication-ready PDF and high-res JPG outputs
- Consistent color scheme across all plots

## Usage

### Basic Usage

```bash
python scripts/plot_map_by_category.py \
    experiments/solid_background_with_overlap_splitted/detailed_evaluation
```

**Output**:
```
experiments/solid_background_with_overlap_splitted/detailed_evaluation_stats_and_plots/
├── aggregated_stats.json          # Computed statistics
├── precision_bar_chart_with_yaxis.pdf
├── precision_bar_chart_with_yaxis.jpg
├── recall_bar_chart_with_yaxis.pdf
├── recall_bar_chart_with_yaxis.jpg
├── f1_score_bar_chart_with_yaxis.pdf
├── f1_score_bar_chart_with_yaxis.jpg
├── all_metrics_combined_with_yaxis.pdf
└── all_metrics_combined_with_yaxis.jpg
```

### Without Y-Axis (for multi-panel figures)

```bash
python scripts/plot_map_by_category.py \
    experiments/solid_background_with_overlap_splitted/detailed_evaluation \
    --no-yaxis
```

Generates additional `*_no_yaxis.pdf/jpg` versions.

### Custom Output Directory

```bash
python scripts/plot_map_by_category.py \
    experiments/solid_background_with_overlap_splitted/detailed_evaluation \
    --output-name custom_plots
```

## Outputs

### 1. aggregated_stats.json

Complete statistics for all categories:

```json
{
  "experiment_info": {
    "total_images": 250,
    "total_ground_truth": 1248,
    "total_predictions": 1205
  },
  "overall_metrics": {
    "precision": 0.9145,
    "recall": 0.8830,
    "f1_score": 0.8985
  },
  "per_category": {
    "cat1": {
      "display_name": "Unconstrained",
      "color": "#00AADC",
      "aggregate_metrics": {
        "precision": 0.9508,
        "recall": 0.9295,
        "f1_score": 0.9401,
        "ground_truth": 312,
        "predictions": 305,
        "true_positives": 290
      },
      "per_image_stats": {
        "precision": {
          "mean": 0.9512,
          "sem": 0.0023,
          "n": 250
        },
        "recall": {
          "mean": 0.9298,
          "sem": 0.0019,
          "n": 250
        },
        "f1_score": {
          "mean": 0.9403,
          "sem": 0.0020,
          "n": 250
        }
      }
    },
    ...
  }
}
```

### 2. Bar Charts (PDF + JPG)

**Individual Metrics**:
- `precision_bar_chart_*.pdf/jpg` - Precision by category
- `recall_bar_chart_*.pdf/jpg` - Recall by category
- `f1_score_bar_chart_*.pdf/jpg` - F1 Score by category

**Combined**:
- `all_metrics_combined_*.pdf/jpg` - All three metrics in one plot

All plots include:
- Category-specific colors
- SEM error bars
- Publication-quality formatting
- Illustrator-compatible PDFs (editable text)

## Color Scheme

Consistent across all plots:

| Category | Name | Color |
|----------|------|-------|
| cat1 | Unconstrained | <span style="color:#00AADC">**#00AADC**</span> (cyan blue) |
| cat2 | Local: Variance | <span style="color:#00539B">**#00539B**</span> (dark blue) |
| cat4 | Local: All | <span style="color:#7F2F8D">**#7F2F8D**</span> (purple) |
| cat5 | Natural | <span style="color:#B01F23">**#B01F23**</span> (red) |

## How SEM is Calculated

The script computes SEM from **per-image metrics**, not aggregate metrics:

```python
# For each image, calculate per-category precision/recall/F1
for each image:
    for each category:
        precision = TP / (TP + FP) if predictions > 0
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)

# Then compute mean and SEM across all images
mean = np.mean(values)
sem = np.std(values, ddof=1) / np.sqrt(n)
```

This gives a more accurate representation of variability across test images.

## Plot Styling

### Publication-Ready Format

- **PDF output**: Vector graphics with editable text (Type 42 fonts)
- **JPG output**: 500 DPI for high-quality printing
- **Transparent backgrounds**: Easy to overlay on presentations
- **Bold error bars**: Cap size 22, line width 7.5
- **Clean axes**: Only left and bottom visible
- **Italic labels**: Consistent with academic standards

### Bar Chart Specifications

- **Bar width**: 0.6 (60% of category spacing)
- **Y-range**: 0-100% with 20% increments
- **Font sizes**: Labels 36pt, ticks 30pt
- **Error bars**: SEM (not SD or confidence intervals)
- **No X-axis labels**: Categories identified by color (add legend in figure caption)

## Batch Processing

Process all three experiments:

```bash
for exp in solid_background_with_overlap_splitted \
           line_background_with_overlap_splitted \
           natural_scenes_background_with_overlap_splitted; do
    python scripts/plot_map_by_category.py \
        experiments/$exp/detailed_evaluation
done
```

## Example Console Output

```
======================================================================
mAP Aggregator and Plotter
======================================================================

Input: experiments/solid_background_with_overlap_splitted/detailed_evaluation
Output: experiments/solid_background_with_overlap_splitted/detailed_evaluation_stats_and_plots

Loading evaluation data...
  ✓ Loaded 250 images

Computing aggregated statistics...
  ✓ Saved: aggregated_stats.json

Generating plots...

  Precision:
  ✓ Saved: precision_bar_chart_with_yaxis.pdf
  ✓ Saved: precision_bar_chart_with_yaxis.jpg

  Recall:
  ✓ Saved: recall_bar_chart_with_yaxis.pdf
  ✓ Saved: recall_bar_chart_with_yaxis.jpg

  F1_score:
  ✓ Saved: f1_score_bar_chart_with_yaxis.pdf
  ✓ Saved: f1_score_bar_chart_with_yaxis.jpg

  All metrics combined:
  ✓ Saved: all_metrics_combined_with_yaxis.pdf
  ✓ Saved: all_metrics_combined_with_yaxis.jpg

======================================================================
Summary
======================================================================

Per-category metrics (mean ± SEM):

Unconstrained        :
  precision : 95.12% ± 0.23%  (n=250)
  recall    : 92.98% ± 0.19%  (n=250)
  f1_score  : 94.03% ± 0.20%  (n=250)

Local: Variance      :
  precision : 89.45% ± 0.31%  (n=250)
  recall    : 87.21% ± 0.28%  (n=250)
  f1_score  : 88.31% ± 0.29%  (n=250)

Local: All           :
  precision : 86.78% ± 0.35%  (n=250)
  recall    : 84.12% ± 0.32%  (n=250)
  f1_score  : 85.43% ± 0.33%  (n=250)

Natural              :
  precision : 91.23% ± 0.27%  (n=250)
  recall    : 89.45% ± 0.24%  (n=250)
  f1_score  : 90.33% ± 0.25%  (n=250)

Results saved to: experiments/.../detailed_evaluation_stats_and_plots
```

## Integration with Papers/Presentations

### For LaTeX Documents

```latex
\begin{figure}[h]
  \centering
  \includegraphics[width=0.8\textwidth]{precision_bar_chart_with_yaxis.pdf}
  \caption{Detection precision by shape category. Error bars represent SEM.}
  \label{fig:precision}
\end{figure}
```

### For PowerPoint/Keynote

Use the JPG files (500 DPI) for high-quality display.

### For Multi-Panel Figures

```bash
# Create three plots without Y-axis for panels 2-3
python scripts/plot_map_by_category.py exp1/detailed_evaluation
python scripts/plot_map_by_category.py exp2/detailed_evaluation --no-yaxis
python scripts/plot_map_by_category.py exp3/detailed_evaluation --no-yaxis
```

Then combine the `*_with_yaxis.pdf` (panel 1) and `*_no_yaxis.pdf` (panels 2-3) in Illustrator.

## Requirements

- Python 3.7+
- matplotlib
- numpy
- Detailed evaluation results (from `detailed_evaluation.py`)

## Notes

- SEM is calculated from per-image metrics for statistical rigor
- All plots use consistent colors for categories
- PDF outputs are vector graphics (scalable, editable)
- Transparent backgrounds make overlay easy
