# Solid Grey Background Experiment

## Quick Summary

- **Categories**: cat1, cat2, cat4, cat5
- **Excluded**: cat3
- **Performance**: mAP@0.5 = 96.0%

## Files

- `EXPERIMENT_REPORT.md` - Full experimental report
- `training/run_1/weights/best.pt` - Trained model
- `evaluation/metrics_comprehensive.json` - All metrics
- `evaluation/figures/` - Visualizations

## Reproduction

This experiment was run using:
```bash
sbatch slurm_solid_grey_experiment.sh
```

See parent `experiments/README.md` for details.
