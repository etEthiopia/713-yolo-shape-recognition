# Color Scheme Guide

This experiment uses a consistent color scheme across **all** visualizations to represent each shape category.

---

## Official Color Scheme

| Category | Class ID | Display Name | Hex Color | RGB | Purpose |
|----------|----------|--------------|-----------|-----|---------|
| cat1 | 0 | Unconstrained | `#00AADC` | (0, 170, 220) | Cyan - no curvature constraints |
| cat2 | 1 | Local (Var) | `#00539B` | (0, 83, 155) | Dark Blue - local curvature variance |
| cat4 | 2 | Local (Matched) | `#7F2F8D` | (127, 47, 141) | Purple - local matched |
| cat5 | 3 | Natural | `#B01F23` | (176, 31, 35) | Red - natural shapes |

**Defined in**: [`metadata/experiment_config.json`](metadata/experiment_config.json)

---

## Where Colors Are Applied

### 🔧 How It Works

YOLO uses its own default color palette for training visualizations. To apply your custom colors, we use **post-processing**:

1. **Training** runs normally with YOLO's default colors
2. **After training**, run `fix_training_colors.py` to regenerate visualizations with correct colors
3. **Evaluation** scripts load colors from metadata and apply them

### ✅ Files That Use Custom Colors

**Post-Training (after running `fix_training_colors.py`)**:
1. **Training Visualizations**
   - `training/run_1/train_batch0.jpg`
   - `training/run_1/train_batch1.jpg`
   - `training/run_1/train_batch2.jpg`

**During Evaluation**:
2. **Evaluation Predictions** (when implemented)
   - `evaluation/predictions/*.jpg`

**Note**: YOLO's built-in plots (confusion matrix, PR curves) use YOLO's colors and can't be easily changed. These are generated during training and are reference only.

---

## Applying Colors to Training Visualizations

### After Training

**Always run this after training completes:**

```bash
cd experiments/solid_background_splitted/scripts

# Regenerate training batch visualizations with correct colors
python fix_training_colors.py
```

This **overwrites** existing `train_batch*.jpg` files with corrected versions.

### Options

```bash
# For different training run
python fix_training_colors.py --training-dir ../training/run_2

# Generate more batch visualizations
python fix_training_colors.py --num-batches 5
```

### Automatic (via pipeline scripts)

The pipeline scripts (`run_experiment.sh`, `run_slurm.sh`) automatically run the fix script after training:

```bash
# Runs: train → fix colors → evaluate
bash run_experiment.sh
```

### For Custom Visualizations

If you're creating custom plots (e.g., in evaluation scripts):

```python
import json
from pathlib import Path

# Load color scheme
with open('experiments/solid_background_splitted/metadata/experiment_config.json') as f:
    metadata = json.load(f)

# Extract colors in order of class ID
class_mapping = metadata['class_mapping']
color_scheme = metadata['color_scheme']

colors_hex = [None] * len(class_mapping)
for cat_name, class_id in class_mapping.items():
    colors_hex[class_id] = color_scheme[cat_name]['hex']

# Use in matplotlib
import matplotlib.pyplot as plt

categories = ['cat1', 'cat2', 'cat4', 'cat5']
mAP_values = [0.85, 0.78, 0.72, 0.80]

plt.bar(categories, mAP_values, color=colors_hex)
plt.show()
```

For OpenCV (BGR format):
```python
def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)  # OpenCV uses BGR

colors_bgr = [hex_to_bgr(c) for c in colors_hex]

# Draw bounding box
cv2.rectangle(img, (x1, y1), (x2, y2), colors_bgr[class_id], 2)
```

---

## Verification

### Check Training Visualizations

After training, verify colors are correct:

```bash
# View training batches
open experiments/solid_background_splitted/training/run_1/train_batch0.jpg

# Expected:
# - cat1 boxes: Cyan (#00AADC)
# - cat2 boxes: Dark Blue (#00539B)
# - cat4 boxes: Purple (#7F2F8D)
# - cat5 boxes: Red (#B01F23)
```

### Check Metadata

```bash
# View color scheme
cat experiments/solid_background_splitted/metadata/experiment_config.json | jq '.color_scheme'
```

---

## Troubleshooting

### Problem: Training visualizations show default YOLO colors

**Symptoms**: `train_batch*.jpg` shows rainbow/default colors instead of official scheme

**Cause**: Haven't run the color fix script yet

**Fix**:
```bash
cd experiments/solid_background_splitted/scripts
python fix_training_colors.py
```

This is **expected** - YOLO always generates visualizations with default colors. Run the fix script after training to apply your colors.

### Problem: Colors don't match across different plots

**Cause**: Custom plots not using metadata color scheme

**Fix**: Load colors from `experiment_config.json` as shown in "Manual Color Application" section

### Problem: Import error when setting colors

**Symptom**: `Warning: Could not set YOLO colors: ...`

**Fix**: This is non-fatal. Training proceeds with default colors. To fix:
```bash
# Verify ultralytics is installed
pip install ultralytics

# Or run fix script after training
python fix_training_colors.py
```

---

## Advanced: Modifying the Color Scheme

To change colors for a new experiment:

1. **Create new experiment** (e.g., `experiments/textured_background/`)

2. **Edit `metadata/experiment_config.json`**:
```json
{
  "color_scheme": {
    "cat1": {
      "display_name": "Unconstrained",
      "hex": "#FF5733",  // Change to your color
      "rgb": [255, 87, 51]
    },
    ...
  }
}
```

3. **Run training**:
```bash
python scripts/train_model.py
```

Colors automatically apply!

---

## Color Consistency Checklist

When creating visualizations, ensure:

- [ ] Colors loaded from `metadata/experiment_config.json`
- [ ] Class ID used to index into color array (not category name)
- [ ] Hex colors used for matplotlib/seaborn
- [ ] BGR colors used for OpenCV (not RGB!)
- [ ] All plots use same color for same category
- [ ] Legend shows display names ("Unconstrained", not "cat1")

---

## Scripts Reference

| Script | Purpose | When to Use |
|--------|---------|-------------|
| `train_model.py` | Train YOLO model | For training |
| `fix_training_colors.py` | Apply colors to training visualizations | **After every training run** |
| `set_yolo_colors.py` | Helper for color conversion | (Optional, not needed for post-processing) |

---

**Summary**: After training completes, run `fix_training_colors.py` to apply your custom color scheme to the training batch visualizations!
