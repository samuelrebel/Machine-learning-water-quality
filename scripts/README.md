# Thesis Scripts Pipeline

This folder contains the scripts for the water-quality modelling pipeline: feature preparation, observation processing, model training, prediction, mapping, and independent station validation. The scripts are primarily designed for Google Colab with data stored on Google Drive. Several visualization scripts also include a local fallback for use in VSCode.

Recommended execution order:

```text
1_feature_transform.py
2_median_observations.py
3_modeltrain.py
4_visualize_scores.py
5_predictions.py
6_visualize_prediction_maps.py
7_validate_holdout_stations.py
```

## Overall Logic

The pipeline uses `hydrobasin_level12` as the main spatial key. Water-quality monitoring stations are linked to HydroBASINS level-12 catchments. The models then learn relationships between observed water-quality concentrations and basin-level predictors, including hydrological, climatic, land-cover, topographic, and socio-economic features.

The two main analysis settings are:

```text
WORLD_all
EU_train_NL_test
```

For final basin predictions, these become:

```text
WORLD_train_WORLD_predict
EU_train_NL_predict
```

The EU/NL route is used to evaluate and apply the model in the Dutch context. The WORLD route is used for global basin-scale predictions.

## Key Design Choices

### ID Fields Are Not Used as Model Predictors

Columns such as `PFAF_ID`, `HYBAS_ID`, `MAIN_BAS`, `NEXT_DOWN`, `ORDER`, `SORT`, `DIST_MAIN`, `DIST_SINK`, `country_name`, `wqms_id`, `wqms_lat`, `wqms_lon`, QC columns, and class-code fields are excluded from machine learning predictors.

These columns are useful for joining, mapping, or quality control, but they should not be used as model predictors because they may introduce artificial spatial identifiers or leakage.

### Human-Readable Feature Labels

HydroATLAS feature codes are converted to clearer labels in the visualization scripts. For example:

```text
pre_mm_syr -> Precipitation, annual value in sub-basin [mm]
tmp_dc_s01 -> Air temperature, month 01 in sub-basin [0.1 deg C]
urb_pc_sse -> Urban area, share in sub-basin [%]
```

The original feature code is usually kept in parentheses, so figures remain readable while still being traceable to the original predictor.

### Monthly Features

Some predictors are month-specific, such as air temperature, precipitation, discharge, and water temperature. During training, each observation is joined to the predictor values for the corresponding month. During prediction, this is repeated for months 1 to 12.

## Script 1: Feature Transform

File:

```text
1_feature_transform.py
```

Purpose:

Builds model-ready feature tables for station-level training and basin-level prediction.

Main steps:

- reads HydroATLAS/BasinATLAS predictor files
- links monitoring stations to canonical basin keys
- uses `hydrobasin_level12` as the primary basin key
- keeps `HYBAS_ID` only as a mapping/QC field
- creates basin support points
- extracts monthly discharge and water temperature at basin support points
- creates separate global and European feature tables
- exports QC/provenance information separately
- determines shared model-ready columns between station and basin feature tables

Important outputs:

```text
T3/output/features/features_modelready_basin_global.csv
T3/output/features/features_modelready_basin_europe.csv
T3/output/features/features_modelready_station_global.csv
T3/output/features/features_modelready_shared_columns_global.csv
T3/output/features/features_modelready_shared_columns_europe.csv
T3/output/features/features_qc_*.csv
```

## Script 2: Median Observations

File:

```text
2_median_observations.py
```

Purpose:

Converts raw WQMS observations into monthly station-level median values per constituent. These files are the target data used for model training.

Main steps:

- reads raw WQMS CSV files
- joins `wqms_site_info.csv`
- filters the period 2010-2023
- removes flag-based outliers where `flag == "*"`
- creates a monthly climatology: one median value per `wqms_id`, `month`, and constituent
- keeps relevant metadata such as `country_name`, coordinates, and basin key
- optionally applies balancing to reduce dominance by the top countries

Important output folder:

```text
T3/output/observations/mean_test/
```

This folder contains one CSV per constituent. These files are used by scripts 3, 5, and 7.

## Script 3: Model Training

File:

```text
3_modeltrain.py
```

Purpose:

Trains and evaluates Random Forest models per constituent using several cross-validation strategies.

Scenarios:

```text
EU_train_NL_test
WORLD_all
```

Main steps:

- reads monthly observations from script 2
- reads model-ready features from script 1
- joins observations to basin features using `hydrobasin_level12`
- selects only valid numeric model predictors
- runs cross-validation
- computes R2, RMSE, and normalized RMSE metrics
- computes feature importance
- writes results incrementally to reduce the risk of losing long runs after a crash

Cross-validation types:

```text
Random CV
Station CV
Basin CV
Country CV
```

Important outputs:

```text
T3/output/results_ML/all_pollutants/world/all_results.csv
T3/output/results_ML/all_pollutants/world/feature_importance.csv
T3/output/results_ML/all_pollutants/world/feature_importance_summary.csv
T3/output/results_ML/all_pollutants/europe_nederland/all_results.csv
T3/output/results_ML/all_pollutants/europe_nederland/feature_importance.csv
T3/output/results_ML/all_pollutants/europe_nederland/feature_importance_summary.csv
```

## Script 4: Visualize Scores

File:

```text
4_visualize_scores.py
```

Purpose:

Creates model performance and feature-importance visualizations.

Main figures:

- top-25 constituents by model performance
- heatmaps by CV type
- WORLD vs EU/NL comparison figures
- score distributions
- model stability figures
- feature importance per constituent
- recurring top features
- feature-importance heatmap

Ranking:

The default ranking metric is:

```text
RANK_METRIC = "R2"
```

For the score figures, `Country CV` and `Country` are excluded by default because they have a different interpretation and can dominate the comparison.

Important output folder:

```text
T3/output/scores/plots/
```

## Script 5: Predictions

File:

```text
5_predictions.py
```

Purpose:

Trains final models without cross-validation and creates monthly basin-level predictions.

Prediction scenarios:

```text
EU_train_NL_predict
WORLD_train_WORLD_predict
```

Main steps:

- trains a final Random Forest model per constituent
- uses the same feature logic as script 3
- predicts basin concentrations for months 1 to 12
- writes predictions in a compact wide format
- preserves existing predictions when the script is rerun
- adds or replaces only the 12 month columns for the current constituent

Important outputs:

```text
T3/output/predictions_basin_monthly/EU_train_NL_predict/predictions_EU_train_NL_predict_wide.csv
T3/output/predictions_basin_monthly/WORLD_train_WORLD_predict/predictions_WORLD_train_WORLD_predict_wide.csv
```

Wide prediction format:

```text
hydrobasin_level12, HYBAS_ID, DO_01, DO_02, ..., DO_12, DOC_01, ...
```

Note:

Script 5 does not create maps by default. Map creation is separated into script 6 to keep prediction generation and visualization independent.

## Script 6: Prediction Maps

File:

```text
6_visualize_prediction_maps.py
```

Purpose:

Creates map visualizations from existing wide prediction CSVs. This script does not train models and does not modify prediction files.

Main inputs:

```text
T3/output/predictions_basin_monthly/<scenario>/predictions_<scenario>_wide.csv
BasinATLAS_v10_lev12.shp
```

Outputs:

```text
T3/output/predictions_basin_monthly/<scenario>/maps/
T3/output/predictions_basin_monthly/<scenario>/maps_key_months/
T3/output/predictions_basin_monthly/map_visualization_manifest.csv
```

Map folders:

```text
maps/
```

Contains separate monthly maps per constituent, for example:

```text
map_WORLD_train_WORLD_predict_DO_01.png
map_WORLD_train_WORLD_predict_DO_02.png
...
map_WORLD_train_WORLD_predict_DO_12.png
```

```text
maps_key_months/
```

Contains compact comparison figures for selected key months. By default these are January and July:

```python
KEY_MONTHS = (1, 7)
```

Why January and July:

These months provide a simple winter/summer contrast and are useful for report figures. If other months are more relevant, change `KEY_MONTHS`.

### World Map Performance

World maps at HydroBASINS level 12 are computationally expensive because there are many small polygons. Therefore, `WORLD_train_WORLD_predict` uses:

```python
"map_style": "point"
```

This plots each basin as a representative point instead of drawing the full polygon. It is much faster and usually sufficient for world-scale figures.

For EU/NL, the script uses:

```python
"map_style": "polygon"
```

This keeps more spatial detail where the number of basins is smaller.

For the world scenario, all 12 monthly PNG maps are still created:

```python
"make_individual_month_maps": True
```

The world scenario also uses the top-25 constituents from the score files by default:

```python
"use_topn_from_scores": True
"top_n": 25
```

To run only the world maps:

```python
RUN_SCENARIOS = ("WORLD_train_WORLD_predict",)
```

To force world polygon maps:

```python
"map_style": "polygon"
```

This is possible, but usually much slower.

## Script 7: Leave-Station-Out Validation

File:

```text
7_validate_holdout_stations.py
```

Purpose:

Runs an independent station-level validation. This is more defensible than comparing predictions with stations that may have been included in model training.

Main logic:

- selects Dutch stations near the Dunea/Rhine/Meuse intake points
- removes these stations completely from the training data
- trains new models for a small set of constituents
- predicts basin-month values for the held-out stations
- compares predicted values with observed monthly medians
- creates publication-quality plots with residual subplots

Default top-5 constituents:

```python
CONSTITUENTS = ["TEMP", "TOC", "Cr-Dis", "DO", "Mn-Tot"]
```

These were selected based on EU/NL Station-CV performance and relevance for the validation.

Dunea intake points:

```text
Afgedamde Maas - Bergambacht: lat 51.93, lon 4.78
Maas upstream reference point near Heusden: lat 51.73, lon 5.14
De Vliet - Leidschendam: lat 52.08, lon 4.38
```

Important outputs:

```text
T3/output/holdout_station_validation/EU_train_NL_leave_station_out_top5/
```

Contains:

```text
selected_holdout_observations.csv
top5_constituent_selection.csv
leave_station_out_observed_vs_predicted_long.csv
leave_station_out_training_log.csv
leave_station_out_metrics.csv
plots/
```

The plots use:

- observed values as a dark blue line
- predicted values as a turquoise line
- month labels Jan-Dec
- seasonal background shading
- RMSE, MAE, Bias, and R2 in a metrics box
- residuals as a subplot below the main figure
- 300 dpi export

Important interpretation:

This validation is intended as an independent test because the selected stations are removed from training. If predictions from script 5 are compared with stations that were included in training, that comparison should be interpreted only as a consistency check, not as independent validation.

## Google Colab Usage

Most scripts mount Google Drive automatically:

```python
from google.colab import drive
drive.mount('/content/drive')
```

The expected base location is:

```text
/content/drive/MyDrive/SamuelRebel_thesis/T2/
```

Important folders:

```text
data/wq_data/
data/predictor_data/
T3/output/
```

When running locally in VSCode, some scripts use a fallback under:

```text
Scriptie/scripts/output/
Scriptie/scripts/Data/
```

## Important Files

Input data:

```text
data/wq_data/wqms-csv/
data/wq_data/wqms_site_info.csv
data/predictor_data/BasinAtlas/ShapeFiles/BasinATLAS_v10_lev12.shp
```

Intermediate outputs:

```text
T3/output/features/
T3/output/observations/mean_test/
T3/output/results_ML/
```

Final outputs:

```text
T3/output/scores/plots/
T3/output/predictions_basin_monthly/
T3/output/holdout_station_validation/
```

## Common Issues

### No Map Data After Filtering

This usually means one of the following:

- the wide prediction CSV does not contain columns such as `DO_01` to `DO_12`
- constituent names do not match between the score file and prediction columns
- the script found an old or empty wide CSV

Check the log lines:

```text
Prediction month columns found
Available constituents in wide CSV
Selected constituents for maps
Long map rows
```

### Prediction Wide CSV Is Overwritten

Script 5 was adjusted to preserve existing wide predictions. For each newly processed constituent, only the 12 month columns for that constituent are added or replaced.

### World Maps Are Slow

Use the point map style for world maps:

```python
"map_style": "point"
RUN_SCENARIOS = ("WORLD_train_WORLD_predict",)
```

This preserves the large-scale spatial patterns while avoiding repeated plotting of heavy level-12 polygons.

### Holdout Validation Finds No Constituents

Script 7 matches constituent names tolerantly, but the selected stations must still have observations for the requested top-5 constituents. If not, the script prints which constituents are available at the selected stations.

## Safe Configuration Changes

The following configuration values are generally safe to adjust:

```python
SELECTED_CONSTITUENTS
TOP_N
USE_TOPN_FROM_SCORES
KEY_MONTHS
RUN_SCENARIOS
CONSTITUENTS
HOLDOUT_STATION_IDS
NEAREST_STATIONS_PER_INTAKE
MAX_DISTANCE_KM
```

Be careful when changing:

```python
hydrobasin_level12
NON_PREDICTOR_EXACT
monthly feature suffixes
output path conventions
```

These parts keep training, prediction, and mapping consistent.

## Short Summary

```text
1_feature_transform.py          creates model-ready features
2_median_observations.py        creates monthly station medians
3_modeltrain.py                 trains and evaluates CV models
4_visualize_scores.py           creates score and feature-importance figures
5_predictions.py                creates monthly basin predictions
6_visualize_prediction_maps.py  creates maps from wide prediction CSVs
7_validate_holdout_stations.py  runs independent leave-station-out validation
```

