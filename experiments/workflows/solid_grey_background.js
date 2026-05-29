#!/usr/bin/env node
/**
 * Solid Grey Background Experiment - Complete YOLO Training Workflow
 *
 * Categories: cat1, cat2, cat4, cat5 (excluding cat3)
 * Shapes per image: 2-7
 * Background: Grey (128, 128, 128)
 * Training: 1,000 images, 200 epochs
 * Test: 250 images
 */

export const meta = {
  name: 'solid-grey-background-experiment',
  description: 'Complete YOLO training pipeline excluding cat3, 2-7 shapes per image',
  phases: [
    { title: 'Setup', detail: 'Create experiment directory and configuration' },
    { title: 'Dataset Creation', detail: 'Filter and sample shapes (cat1,2,4,5 only)' },
    { title: 'Image Generation', detail: 'Generate 2-7 shapes per image, grey background' },
    { title: 'Training', detail: 'Train YOLOv8-nano for 200 epochs' },
    { title: 'Evaluation', detail: 'Comprehensive metrics with statistical analysis' },
    { title: 'Documentation', detail: 'Generate complete experiment report' }
  ]
}

// Define JSON schemas for structured outputs
const SETUP_SCHEMA = {
  type: 'object',
  properties: {
    experiment_dir: { type: 'string' },
    categories_included: { type: 'array', items: { type: 'string' } },
    shapes_per_image_range: { type: 'array', items: { type: 'integer' } },
    background_color: { type: 'string' },
    config_paths: { type: 'object' }
  },
  required: ['experiment_dir', 'categories_included', 'config_paths']
}

const DATASET_SCHEMA = {
  type: 'object',
  properties: {
    total_shapes: { type: 'integer' },
    shapes_per_category: { type: 'object' },
    excluded_categories: { type: 'array' },
    output_path: { type: 'string' }
  },
  required: ['total_shapes', 'shapes_per_category', 'output_path']
}

const IMAGE_GEN_SCHEMA = {
  type: 'object',
  properties: {
    train_images: { type: 'integer' },
    test_images: { type: 'integer' },
    avg_shapes_per_image: { type: 'number' },
    category_distribution: { type: 'object' },
    output_paths: { type: 'object' }
  },
  required: ['train_images', 'test_images', 'output_paths']
}

const TRAINING_SCHEMA = {
  type: 'object',
  properties: {
    model_type: { type: 'string' },
    total_epochs: { type: 'integer' },
    best_epoch: { type: 'integer' },
    final_map50: { type: 'number' },
    weights_path: { type: 'string' }
  },
  required: ['total_epochs', 'weights_path']
}

const EVAL_SCHEMA = {
  type: 'object',
  properties: {
    overall_metrics: { type: 'object' },
    per_category_metrics: { type: 'object' },
    statistical_tests: { type: 'object' },
    visualization_paths: { type: 'array' },
    sample_predictions: { type: 'integer' }
  },
  required: ['overall_metrics', 'visualization_paths']
}

const DOCS_SCHEMA = {
  type: 'object',
  properties: {
    report_path: { type: 'string' },
    metrics_json_path: { type: 'string' },
    readme_path: { type: 'string' },
    total_figures: { type: 'integer' }
  },
  required: ['report_path']
}

// ========================================
// Phase 1: Setup
// ========================================
phase('Setup')
const setup = await agent(
  `Create experiment directory structure for "Solid Grey Background" experiment.

  Base path: experiments/solid_grey_background/

  Create the following directory structure:
  - dataset/ (for selected shapes, images, labels)
  - config/ (for experiment-specific YAML)
  - training/ (for model weights, training curves)
  - evaluation/ (for metrics, figures, reports)
  - metadata/ (for experiment configuration, parameters)

  Tasks:
  1. Create all directories using Python os.makedirs or Bash mkdir -p
  2. Create experiment config YAML at config/shapes.yaml with 4 categories:
     - cat1 (ID: 0, name: Unconstrained)
     - cat2 (ID: 1, name: Local (Var))
     - cat4 (ID: 2, name: Local (Matched))
     - cat5 (ID: 3, name: Natural)
     Note: cat3 is EXCLUDED
  3. Save experiment parameters to metadata/experiment_config.json:
     - experiment_name: "Solid Grey Background"
     - categories: ["cat1", "cat2", "cat4", "cat5"]
     - excluded_categories: ["cat3"]
     - shapes_per_image_range: [2, 7]
     - background_color: "grey (128, 128, 128)"
     - train_images: 1000
     - test_images: 250
     - epochs: 200

  Return the created paths and configuration.`,
  { schema: SETUP_SCHEMA }
)

log(`✓ Experiment directory created: ${setup.experiment_dir}`)
log(`✓ Categories: ${setup.categories_included.join(', ')}`)
log(`✓ Shapes per image: ${setup.shapes_per_image_range[0]}-${setup.shapes_per_image_range[1]}`)

// ========================================
// Phase 2: Dataset Creation
// ========================================
phase('Dataset Creation')
const dataset = await agent(
  `Create balanced dataset by filtering and sampling shapes.

  Source directory: Shapes/
  Output path: experiments/solid_grey_background/dataset/selected_shapes.json

  Categories to INCLUDE: cat1, cat2, cat4, cat5
  Category to EXCLUDE: cat3

  Tasks:
  1. Modify or run scripts/create_balanced_dataset.py with category filtering:
     - Only process cat1, cat2, cat4, cat5 directories
     - Sample exactly 390 shapes from each category
     - Total: 390 × 4 = 1,560 shapes
  2. Save to experiments/solid_grey_background/dataset/selected_shapes.json
  3. JSON structure should include:
     - metadata.samples_per_category: 390
     - metadata.total_shapes: 1560
     - metadata.excluded_categories: ["cat3"]
     - metadata.random_seed: 42
     - shapes: { "cat1": [...paths], "cat2": [...paths], "cat4": [...paths], "cat5": [...paths] }
  4. Verify counts are exactly 390 per category

  Return summary with total shapes and per-category breakdown.`,
  { schema: DATASET_SCHEMA }
)

log(`✓ Dataset created: ${dataset.total_shapes} shapes from 4 categories`)
log(`✓ Per-category: ${JSON.stringify(dataset.shapes_per_category)}`)
log(`✓ Excluded: ${dataset.excluded_categories.join(', ')}`)

// ========================================
// Phase 3: Image Generation
// ========================================
phase('Image Generation')
const images = await agent(
  `Generate composite images with multiple shapes.

  Input: experiments/solid_grey_background/dataset/selected_shapes.json
  Output directory: experiments/solid_grey_background/dataset/

  Parameters:
  - Training images: 1,000
  - Test images: 250
  - Shapes per image: Random 2-7 (configurable range)
  - Background: Grey (128, 128, 128) - solid color
  - Image size: 640×640 pixels
  - Format: JPEG for images, TXT for labels (YOLO format)
  - Random seed: 42 for reproducibility

  Tasks:
  1. Modify or run scripts/generate_composite_images.py with:
     --shapes-json experiments/solid_grey_background/dataset/selected_shapes.json
     --output-dir experiments/solid_grey_background/dataset
     --num-train 1000
     --num-test 250
     --shapes-range 2 7
     --img-size 640
     --seed 42

  2. Generate images/train/ and images/test/ with composite images
  3. Generate labels/train/ and labels/test/ with YOLO format labels
  4. Track category distribution to ensure balance across all images
  5. Calculate average shapes per image

  Output structure:
  - experiments/solid_grey_background/dataset/images/train/*.jpg (1000 files)
  - experiments/solid_grey_background/dataset/images/test/*.jpg (250 files)
  - experiments/solid_grey_background/dataset/labels/train/*.txt (1000 files)
  - experiments/solid_grey_background/dataset/labels/test/*.txt (250 files)

  Return summary with image counts and category distribution.`,
  { schema: IMAGE_GEN_SCHEMA }
)

log(`✓ Generated ${images.train_images} training images`)
log(`✓ Generated ${images.test_images} test images`)
log(`✓ Average shapes per image: ${images.avg_shapes_per_image.toFixed(2)}`)
log(`✓ Category distribution: ${JSON.stringify(images.category_distribution)}`)

// ========================================
// Phase 4: Training
// ========================================
phase('Training')
const training = await agent(
  `Train YOLOv8-nano model on the generated dataset.

  Config file: experiments/solid_grey_background/config/shapes.yaml
  Output directory: experiments/solid_grey_background/training/

  Training parameters:
  - Model: yolov8n.pt (nano, pretrained on COCO)
  - Epochs: 200
  - Batch size: 16
  - Image size: 640
  - Device: Auto-detect (CUDA if available, else CPU)
  - Early stopping patience: 50 epochs
  - Save checkpoints every: 10 epochs
  - Project name: solid_grey_background_training

  Tasks:
  1. Run scripts/train_yolo.py with:
     --config experiments/solid_grey_background/config/shapes.yaml
     --model yolov8n.pt
     --epochs 200
     --batch 16
     --img-size 640
     --project experiments/solid_grey_background/training
     --name run_1
     --patience 50

  2. Monitor training progress and log metrics
  3. Save outputs to experiments/solid_grey_background/training/run_1/:
     - weights/best.pt (best model)
     - weights/last.pt (last epoch)
     - weights/epoch_*.pt (checkpoints every 10 epochs)
     - results.csv (per-epoch metrics)
     - results.png (training curves)
     - confusion_matrix.png

  4. Extract final metrics:
     - Best epoch number
     - Final mAP@0.5
     - Training completion status

  Return summary with training metrics and weights path.`,
  { schema: TRAINING_SCHEMA, model: 'sonnet' }
)

log(`✓ Training complete: ${training.total_epochs} epochs`)
log(`✓ Best mAP@0.5: ${training.final_map50.toFixed(4)} at epoch ${training.best_epoch}`)
log(`✓ Weights saved: ${training.weights_path}`)

// ========================================
// Phase 5: Evaluation
// ========================================
phase('Evaluation')
const evaluation = await agent(
  `Comprehensive evaluation of trained model with statistical analysis.

  Model weights: ${training.weights_path}
  Config: experiments/solid_grey_background/config/shapes.yaml
  Output directory: experiments/solid_grey_background/evaluation/

  Color scheme (use consistently across ALL visualizations):
  - cat1 (Unconstrained): #00AADC (cyan blue) - RGB(0, 170, 220), BGR(220, 170, 0)
  - cat2 (Local Var): #00539B (dark blue) - RGB(0, 83, 155), BGR(155, 83, 0)
  - cat4 (Local Matched): #7F2F8D (purple) - RGB(127, 47, 141), BGR(141, 47, 127)
  - cat5 (Natural): #B01F23 (red) - RGB(176, 31, 35), BGR(35, 31, 176)

  Metrics to compute:
  1. Overall metrics:
     - mAP@0.5, mAP@0.5:0.95, mAP@0.75
     - Precision, Recall, F1-score

  2. Per-category metrics (for cat1, cat2, cat4, cat5 only):
     - mAP@0.5, mAP@0.5:0.95, mAP@0.75
     - Precision, Recall, F1-score

  3. Confusion matrix (4×4 for 4 categories)

  4. IoU threshold sensitivity (0.5 to 0.95 in 0.05 steps)

  5. Statistical tests:
     - One-way ANOVA for category differences
     - 95% confidence intervals per category
     - Pairwise t-tests: cat1 vs cat2, cat1 vs cat4, cat2 vs cat4, synthetic avg vs cat5

  Visualizations to generate (all with consistent colors):
  1. mAP by category (bar chart, colored bars)
  2. Precision-recall curves per category (colored lines)
  3. Confusion matrix heatmap
  4. Sample predictions grid (40 images, 4 rows × 10 columns, bounding boxes in category colors)
  5. IoU threshold sensitivity plot (category-colored lines)
  6. Category distribution analysis (pie/bar chart)
  7-10. Individual PR curves for cat1, cat2, cat4, cat5

  Save outputs to:
  - evaluation/metrics_comprehensive.json (all metrics)
  - evaluation/statistical_analysis.json (ANOVA, p-values, CIs)
  - evaluation/figures/*.png (8-10 figures)
  - evaluation/predictions/*.json (per-image predictions)
  - evaluation/report_evaluation.md (detailed analysis)

  Return summary with metrics and visualization paths.`,
  { schema: EVAL_SCHEMA }
)

log(`✓ Evaluation complete`)
log(`✓ Overall mAP@0.5: ${evaluation.overall_metrics.mAP50.toFixed(4)}`)
log(`✓ Visualizations generated: ${evaluation.visualization_paths.length}`)
log(`✓ Sample predictions: ${evaluation.sample_predictions} images`)

// ========================================
// Phase 6: Documentation
// ========================================
phase('Documentation')
const docs = await agent(
  `Generate comprehensive experiment documentation.

  Input data:
  - Experiment config: experiments/solid_grey_background/metadata/experiment_config.json
  - Dataset stats: ${dataset}
  - Training results: ${training}
  - Evaluation metrics: ${evaluation}

  Output directory: experiments/solid_grey_background/

  Tasks:
  1. Create EXPERIMENT_REPORT.md with sections:
     - Executive Summary (200-300 words)
     - Experimental Design and Rationale
       * Why exclude cat3 (reference previous 18.6% mAP)
       * Why 2-7 shapes per image
       * Why 1,000 training images
     - Dataset Analysis
       * Total shapes: ${dataset.total_shapes}
       * Per-category counts
       * Distribution statistics
       * Balance verification
     - Training Process
       * Hyperparameters table
       * Training curve analysis
       * Convergence discussion
       * Best epoch: ${training.best_epoch}
     - Results
       * Overall metrics
       * Per-category breakdown
       * Statistical significance (reference evaluation.statistical_tests)
       * Comparison to previous experiment (5 categories vs 4)
     - Discussion
       * Curvature complexity effects (cat1 vs cat2 vs cat4)
       * Synthetic vs natural performance
       * Improvement from excluding cat3
     - Failure Analysis
       * False positive patterns
       * Missed detection analysis
     - Conclusions and Recommendations

  2. Export all metrics to experiments/solid_grey_background/evaluation/metrics_all.json

  3. Create README.md with:
     - Experiment overview
     - Reproduction instructions (exact commands)
     - File structure explanation
     - Results summary

  4. Create bibliography.md with citations:
     - YOLOv8 (Ultralytics)
     - Shape data source (Elder, Oleskiw, & Fruend, 2018)

  Return paths to generated documents and total figure count.`,
  { schema: DOCS_SCHEMA }
)

log(`✓ Documentation complete`)
log(`✓ Main report: ${docs.report_path}`)
log(`✓ README: ${docs.readme_path}`)
log(`✓ Total figures: ${docs.total_figures}`)

// ========================================
// Return Summary
// ========================================
return {
  experiment: 'solid_grey_background',
  status: 'complete',
  categories: ['cat1', 'cat2', 'cat4', 'cat5'],
  excluded: ['cat3'],
  shapes_per_image: [2, 7],
  dataset: {
    total_shapes: dataset.total_shapes,
    train_images: images.train_images,
    test_images: images.test_images,
    avg_shapes_per_image: images.avg_shapes_per_image
  },
  training: {
    epochs: training.total_epochs,
    best_epoch: training.best_epoch,
    best_map50: training.final_map50,
    model: training.model_type,
    weights: training.weights_path
  },
  evaluation: {
    overall_map50: evaluation.overall_metrics.mAP50,
    overall_precision: evaluation.overall_metrics.precision,
    overall_recall: evaluation.overall_metrics.recall,
    per_category: evaluation.per_category_metrics,
    visualizations: evaluation.visualization_paths.length,
    statistical_tests: evaluation.statistical_tests
  },
  outputs: {
    experiment_dir: setup.experiment_dir,
    config: setup.config_paths,
    weights: training.weights_path,
    metrics: docs.metrics_json_path,
    report: docs.report_path,
    readme: docs.readme_path,
    figures: evaluation.visualization_paths
  },
  summary: {
    improvement_over_previous: evaluation.overall_metrics.mAP50 > 0.729,
    all_categories_above_threshold: Object.values(evaluation.per_category_metrics).every(m => m.mAP50 > 0.45),
    documentation_complete: docs.total_figures >= 8
  }
}
