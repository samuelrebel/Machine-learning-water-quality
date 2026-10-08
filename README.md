# Water Quality Prediction Thesis Package

This folder contains the code and selected outputs prepared for sharing and review. The project predicts monthly basin-level water-quality concentrations by linking WQMS monitoring data to HydroBASINS/HydroATLAS catchment predictors.

## Contents

```text

scripts/        Reproducible Python scripts for the full workflow
scores/         Model performance, feature-importance, and validation figures
visualization/  Monthly prediction maps and key-month comparison maps
```

More detailed notes are available inside each subfolder:

```text
scripts/README.md
scores/README.md
visualization/README.md
```

## Workflow

The main scripts should be run in this order:

```text
1_feature_transform.py
2_median_observations.py
3_modeltrain.py
4_visualize_scores.py
5_predictions.py
6_visualize_prediction_maps.py
7_validate_holdout_stations.py
```

Short description:

- `1_feature_transform.py`: prepares model-ready basin and station features.
- `2_median_observations.py`: converts raw observations to monthly station medians.
- `3_modeltrain.py`: trains and evaluates Random Forest models.
- `4_visualize_scores.py`: creates performance and feature-importance figures.
- `5_predictions.py`: creates monthly basin-level predictions.
- `6_visualize_prediction_maps.py`: creates map visualizations from prediction files.
- `7_validate_holdout_stations.py`: performs leave-station-out validation.

## Main Scenarios

The project uses two main modelling/prediction settings:

```text
WORLD_all / WORLD_train_WORLD_predict
EU_train_NL_test / EU_train_NL_predict
```

The WORLD scenario represents global basin-scale modelling. The EU/NL scenario focuses on transfer to the Dutch context.

## Outputs Included

The `scores/` folder contains:

- top-25 model performance figures
- CV heatmaps
- feature-importance plots per constituent
- station validation plots with and without leave-station-out training

The `visualization/` folder contains:

- 300 monthly world maps: 25 constituents x 12 months
- 300 monthly EU/NL maps: 25 constituents x 12 months
- 25 key-month world comparison figures
- 25 key-month EU/NL comparison figures

## Validation Note

For independent validation, use the figures in:

```text
scores/holdout_station_zonder_training/
```

These are based on models where the selected stations were removed from training before prediction. Figures in:

```text
scores/holdout_station_met_training/
```

should be interpreted only as consistency checks, because those stations may have contributed to model training.

## Reproducibility

The scripts are designed for Google Colab and assume the original project data are available on Google Drive under:

```text
/content/drive/MyDrive/SamuelRebel_thesis/T2/
```

Large raw input datasets are not included in this publication folder. The included figures and scripts document the analysis workflow and selected outputs.

