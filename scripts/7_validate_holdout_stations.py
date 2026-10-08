# -*- coding: utf-8 -*-
"""7_validate_holdout_stations

Independent leave-station-out validation for selected Dutch intake stations.

This script selects a small set of WQMS stations, removes those stations from
training, trains new models for the top constituents, predicts the basin-month
values for the held-out stations, and compares those predictions with observed
monthly medians.
"""

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IS_COLAB = True
except Exception:
    IS_COLAB = False

#!/usr/bin/env python3
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

try:
    import cudf
    from cuml.ensemble import RandomForestRegressor as cuRF
    try:
        import cupy as cp
    except Exception:
        cp = None
    HAS_CUML = True
except Exception:
    cudf = None
    cuRF = None
    cp = None
    HAS_CUML = False

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
if IS_COLAB:
    BASE = "/content/drive/MyDrive/SamuelRebel_thesis/T2/"
    OUTPUT_ROOT = f"{BASE}T3/output"
else:
    OUTPUT_ROOT = str(SCRIPT_DIR / "output")

OBS_DIR = f"{OUTPUT_ROOT}/observations/mean_test"
OUT_DIR = f"{OUTPUT_ROOT}/holdout_station_validation"

FEATURES_WORLD = f"{OUTPUT_ROOT}/features/features_modelready_basin_global.csv"
FEATURES_EUROPE = f"{OUTPUT_ROOT}/features/features_modelready_basin_europe.csv"
SHARED_WORLD = f"{OUTPUT_ROOT}/features/features_modelready_shared_columns_global.csv"
SHARED_EUROPE = f"{OUTPUT_ROOT}/features/features_modelready_shared_columns_europe.csv"

if IS_COLAB:
    SITE_INFO = f"{BASE}/data/wq_data/wqms_site_info.csv"
else:
    SITE_INFO = str(SCRIPT_DIR / "Data" / "wqms_site_info.csv")

SCENARIO = "EU_train_NL_leave_station_out_top5"
TRAIN_SCOPE = "EU"  # "EU" or "WORLD"; default is EU/NL validation
USE_GPU_IF_AVAILABLE = True
N_ESTIMATORS = 300
SEED = 42

# Optional: vul specifieke stations in; script 7 verwijdert ze zelf uit training.
HOLDOUT_STATION_IDS = []  # e.g. ["wqms_16500001", "wqms_16500002"]
HOLDOUT_STATIONS_CSV = None  # optional CSV with column wqms_id

# Als geen stationlijst is opgegeven: selecteer dichtstbijzijnde WQMS-stations
# bij de Dunea-innamepunten uit de screenshot.
USE_DUNEA_INTAKE_POINTS = True
NEAREST_STATIONS_PER_INTAKE = 3
MAX_DISTANCE_KM = 50  # zet None om geen afstandsgrens te gebruiken
DUNEA_INTAKE_POINTS = [
    {"name": "Afgedamde Maas - Bergambacht (hoofdinname)", "lat": 51.93, "lon": 4.78},
    {"name": "Maas upstream referentiepunt bij Heusden", "lat": 51.73, "lon": 5.14},
    {"name": "De Vliet (pilot-inname bij Leidschendam)", "lat": 52.08, "lon": 4.38},
]

# Fallback als USE_DUNEA_INTAKE_POINTS=False of geen matches gevonden worden.
AUTO_SELECT_IF_EMPTY = True
AUTO_SELECT_N_STATIONS = 6
AUTO_COUNTRY = "Netherlands"
RIVER_SEARCH_TERMS = ["Rijn", "Rhine", "Maas", "Meuse"]
HYDROBASIN_KEYS = []  # optional basin keys for Rhine/Meuse selection

# Standaard expliciete top-5 op basis van EU/NL Station-CV performance.
# Zet op None als je automatisch uit scorebestanden wilt ranken.
CONSTITUENTS = ["TEMP", "TOC", "Cr-Dis", "DO", "Mn-Tot"]
TOP_N_CONSTITUENTS = 5
RANK_METRIC = "R2"  # alternatives: RMSE, NRMSE_mean, NRMSE_range, NRMSE_std
TOPN_CV_TYPES = ("Station CV",)  # best passend bij leave-station-out validatie
EXCLUDE_CV_TYPES = {"Country CV", "Country"}
MIN_FOLDS_PER_CV = 2
MIN_SCORE_OBS = 110
ALLOW_TOPN_FALLBACK_BY_OBS_COUNT = False
SCORES_EU_NL_CANDIDATES = [
    f"{OUTPUT_ROOT}/results_ML/all_pollutants/europe_nederland/all_results.csv",
    f"{OUTPUT_ROOT}/results_ML/all_pollutants/europe_nederland/all_results_NL.csv",
    f"{OUTPUT_ROOT}/results_ML/europe_nederland/all_results.csv",
    f"{OUTPUT_ROOT}/results_ML/europe_nederland/all_results_NL.csv",
    str(SCRIPT_DIR.parent / "scores" / "all_results_NL.csv"),
]
MIN_MONTHS_PER_PLOT = 3
EXPORT_DPI = 300
LAST_TOPN_RANKING = pd.DataFrame()

sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "font.size": 13,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

EUROPE_COUNTRIES = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany", "Greece", "Hungary",
    "Iceland", "Ireland", "Italy", "Kazakhstan", "Kosovo", "Latvia", "Liechtenstein",
    "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco", "Montenegro", "Netherlands",
    "North Macedonia", "Republic of Macedonia", "Norway", "Poland", "Portugal", "Romania",
    "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden",
    "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City",
}

COUNTRY_FIELD = "country_name"
NL_NAME = "Netherlands"

NON_PREDICTOR_EXACT = {
    "HYBAS_ID", "hydrobasin_level12", "geometry",
    "NEXT_DOWN", "NEXT_SINK", "MAIN_BAS", "PFAF_ID", "SORT", "ORDER",
    "DIST_SINK", "DIST_MAIN", "SUB_AREA", "UP_AREA", "ENDO", "COAST",
    "supportpoint_method", "support_stream_order", "continent_tile",
    "country_name", "country_name_obs", "wqms_id", "wqms_lat", "wqms_lon",
    "hydrobasin_level12_obs", "hydrobasin_level12_spatial", "HYBAS_ID_spatial",
    "qc_has_obs_key", "qc_has_spatial_key", "qc_obs_vs_spatial_key_mismatch",
    "qc_primary_vs_hybasid_mismatch", "qc_primary_vs_hybasid_mismatch_basin",
    "qc_hybasid_from_basin", "gad_id_smj",
}

NON_PREDICTOR_CLASS_CODES = {
    "clz_cl_smj", "cls_cl_smj", "glc_cl_smj", "pnv_cl_smj", "wet_cl_smj",
    "tbi_cl_smj", "tec_cl_smj", "fmh_cl_smj", "fec_cl_smj", "lit_cl_smj",
}

NON_PREDICTOR_PREFIXES = ("qc_", "prov_")


def is_model_predictor_column(column: str) -> bool:
    c = str(column).strip()
    if c in NON_PREDICTOR_EXACT or c in NON_PREDICTOR_CLASS_CODES:
        return False
    if c.startswith(NON_PREDICTOR_PREFIXES):
        return False
    if c.endswith("_id") or "_id_" in c.lower():
        return False
    if re.search(r"_cl_", c):
        return False
    return True


def normalize_key(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:140] or "unknown"


def constituent_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value).upper())


def resolve_constituent_names(requested: list[str], available: set[str]) -> tuple[list[str], list[str]]:
    available_by_key = {constituent_key(v): v for v in sorted(available)}
    resolved = []
    missing = []
    for name in requested:
        match = available_by_key.get(constituent_key(name))
        if match is None:
            missing.append(str(name))
        elif match not in resolved:
            resolved.append(match)
    return resolved, missing


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * r * np.arcsin(np.sqrt(a))


def load_site_info(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"SITE_INFO niet gevonden: {path}")
    site = pd.read_csv(path)
    required = {"wqms_id", "wqms_lat", "wqms_lon", "hydrobasin_level12"}
    missing = required - set(site.columns)
    if missing:
        raise ValueError(f"SITE_INFO mist kolommen {missing}: {path}")
    site = site.copy()
    site["hydrobasin_level12"] = normalize_key(site["hydrobasin_level12"])
    site["wqms_lat"] = pd.to_numeric(site["wqms_lat"], errors="coerce")
    site["wqms_lon"] = pd.to_numeric(site["wqms_lon"], errors="coerce")
    site = site.dropna(subset=["wqms_id", "wqms_lat", "wqms_lon", "hydrobasin_level12"])
    return site


def nearest_stations_to_dunea_points(obs: pd.DataFrame, site_info_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select nearest WQMS stations with available observations for each Dunea intake point."""
    site = load_site_info(site_info_path)

    obs_station_ids = set(obs["wqms_id"].astype(str).unique())
    site = site[site["wqms_id"].astype(str).isin(obs_station_ids)].copy()
    if AUTO_COUNTRY and "country_name" in site.columns:
        site = site[site["country_name"] == AUTO_COUNTRY].copy()
    if site.empty:
        return obs.iloc[0:0].copy(), pd.DataFrame()

    selected_parts = []
    nearest_rows = []
    for point in DUNEA_INTAKE_POINTS:
        d = site.copy()
        d["intake_name"] = point["name"]
        d["intake_lat"] = point["lat"]
        d["intake_lon"] = point["lon"]
        d["distance_to_intake_km"] = haversine_km(
            d["wqms_lat"].to_numpy(float),
            d["wqms_lon"].to_numpy(float),
            float(point["lat"]),
            float(point["lon"]),
        )
        if MAX_DISTANCE_KM is not None:
            d = d[d["distance_to_intake_km"] <= MAX_DISTANCE_KM].copy()
        d = d.sort_values("distance_to_intake_km").head(NEAREST_STATIONS_PER_INTAKE)
        nearest_rows.append(d)
        selected_parts.append(obs[obs["wqms_id"].astype(str).isin(d["wqms_id"].astype(str))].copy())

    nearest = pd.concat(nearest_rows, ignore_index=True) if nearest_rows else pd.DataFrame()
    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else obs.iloc[0:0].copy()

    if not nearest.empty:
        nearest = nearest[[c for c in [
            "intake_name", "intake_lat", "intake_lon", "wqms_id", "wqms_lat", "wqms_lon",
            "country_name", "hydrobasin_level12", "distance_to_intake_km"
        ] if c in nearest.columns]].drop_duplicates()
        selected = selected.merge(
            nearest[["wqms_id", "intake_name", "distance_to_intake_km"]].drop_duplicates("wqms_id"),
            on="wqms_id",
            how="left",
        )
    return selected, nearest


def list_observation_files(obs_dir: str) -> list[str]:
    excluded = {"balancing_report_summary.csv", "balancing_report_by_country.csv"}
    files = sorted(Path(obs_dir).glob("*.csv"))
    return [str(p) for p in files if p.name not in excluded and not p.name.startswith(("features_", "all_results", "feature_importance", "table_"))]


def observation_file_map(obs_dir: str) -> dict[str, str]:
    return {Path(p).stem: p for p in list_observation_files(obs_dir)}


def load_observations(obs_dir: str, constituents: set[str] | None) -> pd.DataFrame:
    rows = []
    for path in list_observation_files(obs_dir):
        variable_from_file = Path(path).stem
        if constituents is not None and variable_from_file not in constituents:
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"⚠️ Kan observatiebestand niet lezen: {path} | {e}")
            continue
        required = {"wqms_id", "obs", "month", "hydrobasin_level12"}
        missing = required - set(df.columns)
        if missing:
            print(f"⚠️ Sla over, mist kolommen {missing}: {path}")
            continue
        if "variable" not in df.columns:
            df["variable"] = variable_from_file
        keep = [c for c in [
            "wqms_id", "obs", "month", "hydrobasin_level12", "variable", "country_name",
            "wqms_lat", "wqms_lon", "station_name", "site_name", "waterbody", "river", "water_body_name"
        ] if c in df.columns]
        rows.append(df[keep].copy())
    if not rows:
        return pd.DataFrame()
    obs = pd.concat(rows, ignore_index=True)
    obs["hydrobasin_level12"] = normalize_key(obs["hydrobasin_level12"])
    obs["month"] = pd.to_numeric(obs["month"], errors="coerce").astype("Int64")
    obs["obs"] = pd.to_numeric(obs["obs"], errors="coerce")
    return obs[obs["month"].between(1, 12, inclusive="both") & obs["obs"].notna()].copy()


def read_shared_features(path: str | None) -> list[str]:
    if not path or not os.path.exists(path):
        return []
    s = pd.read_csv(path)
    if "feature" in s.columns:
        return [str(v) for v in s["feature"].dropna().tolist()]
    return [str(v) for v in s.iloc[:, 0].dropna().tolist()]


def load_basin_features(features_path: str, shared_path: str | None):
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features-file niet gevonden: {features_path}")

    df = pd.read_csv(features_path)
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]

    if "hydrobasin_level12" not in df.columns:
        raise ValueError(f"Features-file mist hydrobasin_level12: {features_path}")

    shared = read_shared_features(shared_path)
    if shared:
        keep = [c for c in ["hydrobasin_level12", "HYBAS_ID"] if c in df.columns] + [c for c in shared if c in df.columns]
        df = df[keep].copy()

    df["hydrobasin_level12"] = normalize_key(df["hydrobasin_level12"])
    df = df.dropna(subset=["hydrobasin_level12"]).drop_duplicates(subset="hydrobasin_level12", keep="first").reset_index(drop=True)

    monthly_cols = [c for c in df.columns if re.search(r"_s(0[1-9]|1[0-2])$", str(c))]
    id_cols = {"hydrobasin_level12", "HYBAS_ID"}
    static_cols = [c for c in df.columns if c not in id_cols and c not in monthly_cols]
    return df, static_cols, monthly_cols


def build_training_table(obs_path: str,
                         feat_df: pd.DataFrame,
                         static_cols: list[str],
                         monthly_cols: list[str],
                         train_scope: str) -> pd.DataFrame:
    obs = pd.read_csv(obs_path)
    required = {"wqms_id", "obs", "month", "hydrobasin_level12"}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"Obs-file mist kolommen {missing}: {obs_path}")

    obs["hydrobasin_level12"] = normalize_key(obs["hydrobasin_level12"])
    obs["month"] = pd.to_numeric(obs["month"], errors="coerce")
    obs["obs"] = pd.to_numeric(obs["obs"], errors="coerce")
    obs = obs[obs["month"].between(1, 12, inclusive="both") & obs["obs"].notna()].copy()
    obs["month"] = obs["month"].astype(int)

    if train_scope == "EU" and COUNTRY_FIELD in obs.columns:
        obs = obs[obs[COUNTRY_FIELD].isin(EUROPE_COUNTRIES)].copy()

    obs = obs[obs["hydrobasin_level12"].isin(feat_df["hydrobasin_level12"])].copy()
    if obs.empty:
        return obs

    overlap = set(obs.columns) & set(static_cols)
    static_no_overlap = [c for c in static_cols if c not in overlap]
    base = obs.merge(
        feat_df[["hydrobasin_level12"] + static_no_overlap],
        on="hydrobasin_level12",
        how="left",
        validate="m:1",
    )

    if not monthly_cols:
        return base

    rows = []
    for m in range(1, 13):
        part = base.loc[base["month"] == m].copy()
        if part.empty:
            continue
        suffix = f"_s{m:02d}"
        mcols = [c for c in monthly_cols if str(c).endswith(suffix)]
        if not mcols:
            rows.append(part)
            continue
        feats_m = feat_df[["hydrobasin_level12"] + mcols].copy()
        feats_m = feats_m.rename(columns={c: c.rsplit("_", 1)[0] for c in mcols})
        rows.append(part.merge(feats_m, on="hydrobasin_level12", how="left", validate="m:1"))

    df_full = pd.concat(rows, ignore_index=True) if rows else base
    drop_monthly_raw = [c for c in df_full.columns if re.search(r"_s(0[1-9]|1[0-2])$", str(c))]
    return df_full.drop(columns=drop_monthly_raw, errors="ignore").loc[:, ~df_full.columns.duplicated()]


def build_predict_table_for_month(feat_df: pd.DataFrame,
                                  static_cols: list[str],
                                  monthly_cols: list[str],
                                  month: int) -> pd.DataFrame:
    suffix = f"_s{month:02d}"
    mcols = [c for c in monthly_cols if str(c).endswith(suffix)]
    base = feat_df[["hydrobasin_level12"] + [c for c in ["HYBAS_ID"] if c in feat_df.columns] + static_cols].copy()
    if mcols:
        mm = feat_df[["hydrobasin_level12"] + mcols].copy()
        mm = mm.rename(columns={c: c.rsplit("_", 1)[0] for c in mcols})
        base = base.merge(mm, on="hydrobasin_level12", how="left", validate="1:1")
    return base.loc[:, ~base.columns.duplicated()]


def prepare_feature_columns(df_train: pd.DataFrame) -> list[str]:
    exclude = {
        "obs", "variable", "unit", "dates", "year", "year_month", "month",
        "wqms_id", "hydrobasin_level12", "HYBAS_ID", COUNTRY_FIELD,
    }
    cols = [c for c in df_train.columns if c not in exclude and is_model_predictor_column(c)]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df_train[c])]


def fit_imputation_stats(df_train: pd.DataFrame, feature_cols: list[str]):
    x = df_train[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    mainbas_medians = None
    if "MAIN_BAS" in df_train.columns:
        mainbas_medians = df_train.groupby("MAIN_BAS")[feature_cols].median(numeric_only=True)
    return mainbas_medians, x.median(numeric_only=True)


def apply_imputation(df_any: pd.DataFrame,
                     feature_cols: list[str],
                     mainbas_medians,
                     global_medians) -> pd.DataFrame:
    x = df_any[feature_cols].copy().replace([np.inf, -np.inf], np.nan)
    if mainbas_medians is not None and "MAIN_BAS" in df_any.columns:
        main_keys = df_any["MAIN_BAS"]
        for c in feature_cols:
            if c in mainbas_medians.columns:
                x[c] = x[c].fillna(main_keys.map(mainbas_medians[c]))
    return x.fillna(global_medians).fillna(0).astype(np.float32)


def to_numpy_1d(pred_raw) -> np.ndarray:
    if hasattr(pred_raw, "to_numpy"):
        return np.asarray(pred_raw.to_numpy(), dtype=float).ravel()
    if hasattr(pred_raw, "values_host"):
        return np.asarray(pred_raw.values_host, dtype=float).ravel()
    if cp is not None and isinstance(pred_raw, cp.ndarray):
        return cp.asnumpy(pred_raw).astype(float).ravel()
    return np.asarray(pred_raw, dtype=float).ravel()


def fit_random_forest(x_train: pd.DataFrame, y_train: np.ndarray):
    if USE_GPU_IF_AVAILABLE and HAS_CUML:
        model = cuRF(n_estimators=N_ESTIMATORS, random_state=SEED, n_streams=1)
        model.fit(cudf.DataFrame(x_train), cudf.Series(y_train.astype(np.float32)))
        return model, "cuML RandomForest"

    model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=SEED,
        n_jobs=-1,
        min_samples_leaf=1,
    )
    model.fit(x_train, y_train.astype(np.float32))
    return model, "sklearn RandomForest"


def predict_random_forest(model, x_pred: pd.DataFrame) -> np.ndarray:
    if HAS_CUML and model.__class__.__module__.startswith("cuml"):
        return to_numpy_1d(model.predict(cudf.DataFrame(x_pred)))
    return to_numpy_1d(model.predict(x_pred))


def simplify_cv_type(cv):
    if pd.isna(cv):
        return cv
    return re.sub(r"\s*\(.*\)\s*$", "", str(cv)).strip()


def is_lower_better(metric: str) -> bool:
    m = metric.upper()
    return m.startswith("NRMSE") or m.endswith("RMSE")


def resolve_existing_path(candidates: list[str], label: str) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ {label}: {path}")
            return path
    print(f"⚠️ Geen scorebestand gevonden voor {label}; top-5 fallback gebruikt observatie-aantallen.")
    return None


def top_constituents_from_scores(available_vars: set[str], top_n: int) -> list[str]:
    global LAST_TOPN_RANKING
    LAST_TOPN_RANKING = pd.DataFrame()

    score_path = resolve_existing_path(SCORES_EU_NL_CANDIDATES, "scores EU_train_NL_test")
    if score_path is None:
        return []

    df = pd.read_csv(score_path)
    if "Obs." in df.columns and "Obs" not in df.columns:
        df = df.rename(columns={"Obs.": "Obs"})

    required = {"Variable", "CV_type", "Fold", "Obs", RANK_METRIC}
    missing = required - set(df.columns)
    if missing:
        print(f"⚠️ Scorebestand mist kolommen {missing}; top-5 fallback gebruikt observatie-aantallen.")
        return []

    for c in ["Fold", "Obs", "R2", "RMSE", "NRMSE_mean", "NRMSE_range", "NRMSE_std"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["CV_type"] = df["CV_type"].apply(simplify_cv_type)
    df = df[~df["CV_type"].isin(EXCLUDE_CV_TYPES)].copy()
    if TOPN_CV_TYPES:
        df = df[df["CV_type"].isin(TOPN_CV_TYPES)].copy()
    df = df[df["Variable"].astype(str).isin(available_vars)].copy()
    if df.empty:
        print("⚠️ Geen score-rows over na filter op beschikbare holdout-stoffen en TOPN_CV_TYPES.")
        return []

    cv_agg = (
        df.groupby(["Variable", "CV_type"], as_index=False)
          .agg(
              metric_med=(RANK_METRIC, "median"),
              rmse_med=("RMSE", "median"),
              n_med=("Obs", "median"),
              folds=("Fold", "nunique"),
          )
    )
    cv_agg = cv_agg[(cv_agg["n_med"] >= MIN_SCORE_OBS) & (cv_agg["folds"] >= MIN_FOLDS_PER_CV)].copy()
    if cv_agg.empty:
        print("⚠️ Geen score-rows over na MIN_SCORE_OBS/MIN_FOLDS_PER_CV filter.")
        return []

    var_agg = (
        cv_agg.groupby("Variable", as_index=False)
        .agg(
            metric_overall=("metric_med", "median"),
            rmse_median=("rmse_med", "median"),
            n_med=("n_med", "median"),
            cv_types=("CV_type", lambda s: ", ".join(sorted(set(map(str, s))))),
        )
    )
    ranked = var_agg.sort_values("metric_overall", ascending=is_lower_better(RANK_METRIC))
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selection_source"] = "score_file"
    ranked["rank_metric"] = RANK_METRIC
    LAST_TOPN_RANKING = ranked.copy()
    print("\nTop-N ranking gebruikt voor validatie:")
    print(ranked.head(max(top_n, 10)).to_string(index=False))
    return ranked.head(top_n)["Variable"].astype(str).tolist()


def choose_validation_constituents(holdout_obs: pd.DataFrame, top_n: int) -> list[str]:
    global LAST_TOPN_RANKING
    available = set(holdout_obs["variable"].dropna().astype(str).unique().tolist())
    if CONSTITUENTS is not None:
        chosen, missing = resolve_constituent_names(CONSTITUENTS, available)
        if not chosen:
            available_preview = sorted(available)[:80]
            raise ValueError(
                "Geen CONSTITUENTS beschikbaar bij de geselecteerde holdout-stations. "
                f"Gevraagd: {CONSTITUENTS}. "
                f"Beschikbaar bij deze stations, eerste {len(available_preview)}: {available_preview}"
            )
        if missing:
            print(f"⚠️ Niet alle handmatige CONSTITUENTS zijn beschikbaar bij deze stations: {missing}")
            print(f"Wel gevonden en gebruikt: {chosen[:top_n]}")
        LAST_TOPN_RANKING = pd.DataFrame({
            "Variable": chosen[:top_n],
            "rank": range(1, len(chosen[:top_n]) + 1),
            "selection_source": "manual_CONSTITUENTS",
        })
        return chosen[:top_n]

    chosen = top_constituents_from_scores(available, top_n)
    if chosen:
        return chosen

    if not ALLOW_TOPN_FALLBACK_BY_OBS_COUNT:
        raise ValueError(
            "Top-5 kon niet uit scorebestanden worden bepaald. Zet CONSTITUENTS handmatig "
            "of controleer SCORES_EU_NL_CANDIDATES. Fallback op observatie-aantallen staat uit."
        )

    fallback = (
        holdout_obs.groupby("variable", as_index=False)
        .agg(n_rows=("obs", "size"), n_stations=("wqms_id", "nunique"), n_months=("month", "nunique"))
        .sort_values(["n_stations", "n_months", "n_rows"], ascending=[False, False, False])
    )
    fallback = fallback.rename(columns={"variable": "Variable"})
    fallback["rank"] = np.arange(1, len(fallback) + 1)
    fallback["selection_source"] = "fallback_holdout_observation_count"
    LAST_TOPN_RANKING = fallback.copy()
    print("\nFallback top-N ranking gebruikt voor validatie:")
    print(fallback.head(max(top_n, 10)).to_string(index=False))
    return fallback.head(top_n)["Variable"].astype(str).tolist()


def read_holdout_station_ids() -> set[str]:
    ids = set(str(x) for x in HOLDOUT_STATION_IDS)
    if HOLDOUT_STATIONS_CSV and os.path.exists(HOLDOUT_STATIONS_CSV):
        t = pd.read_csv(HOLDOUT_STATIONS_CSV)
        if "wqms_id" not in t.columns:
            raise ValueError(f"HOLDOUT_STATIONS_CSV mist kolom wqms_id: {HOLDOUT_STATIONS_CSV}")
        ids.update(t["wqms_id"].dropna().astype(str).tolist())
    return ids


def select_holdout_observations(obs: pd.DataFrame) -> pd.DataFrame:
    ids = read_holdout_station_ids()
    if ids:
        selected = obs[obs["wqms_id"].astype(str).isin(ids)].copy()
        if selected.empty:
            raise ValueError(f"Geen observaties gevonden voor HOLDOUT_STATION_IDS: {sorted(ids)}")
        print(f"Holdout stations uit lijst: {selected['wqms_id'].nunique()}")
        return selected

    if USE_DUNEA_INTAKE_POINTS:
        selected, nearest = nearest_stations_to_dunea_points(obs, SITE_INFO)
        if not nearest.empty:
            Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
            nearest_path = os.path.join(OUT_DIR, "dunea_nearest_wqms_stations.csv")
            nearest.to_csv(nearest_path, index=False)
            print(f"Dichtstbijzijnde WQMS-stations bij Dunea-punten: {nearest['wqms_id'].nunique()}")
            print(f"Nearest-station tabel: {nearest_path}")
            print(nearest.head(12).to_string(index=False))
        if not selected.empty:
            return selected
        print("⚠️ Geen observaties gevonden voor Dunea-nearest stations; fallback naar automatische selectie.")

    d = obs.copy()
    if HYDROBASIN_KEYS:
        keys = set(str(x) for x in HYDROBASIN_KEYS)
        d = d[d["hydrobasin_level12"].isin(keys)].copy()
        print(f"Filter op HYDROBASIN_KEYS: stations={d['wqms_id'].nunique()}, rows={len(d):,}")
    if AUTO_COUNTRY and "country_name" in d.columns:
        d = d[d["country_name"] == AUTO_COUNTRY].copy()

    text_cols = [c for c in ["station_name", "site_name", "waterbody", "river", "water_body_name"] if c in d.columns]
    if text_cols and RIVER_SEARCH_TERMS:
        pattern = "|".join(re.escape(x) for x in RIVER_SEARCH_TERMS)
        mask = pd.Series(False, index=d.index)
        for col in text_cols:
            mask = mask | d[col].astype("string").str.contains(pattern, case=False, na=False)
        d_terms = d[mask].copy()
        if not d_terms.empty:
            d = d_terms
            print(f"Filter op rivierzoektermen {RIVER_SEARCH_TERMS}: stations={d['wqms_id'].nunique()}, rows={len(d):,}")
        else:
            print("⚠️ Geen matches op RIVER_SEARCH_TERMS; automatische selectie gebruikt Nederlandse stations met meeste data.")

    if not AUTO_SELECT_IF_EMPTY:
        return d
    station_rank = (
        d.groupby("wqms_id", as_index=False)
         .agg(n_rows=("obs", "size"), n_constituents=("variable", "nunique"), n_months=("month", "nunique"))
         .sort_values(["n_constituents", "n_rows"], ascending=[False, False])
    )
    selected_ids = station_rank.head(AUTO_SELECT_N_STATIONS)["wqms_id"].astype(str).tolist()
    print(f"Automatisch geselecteerde stations: {selected_ids}")
    return d[d["wqms_id"].astype(str).isin(selected_ids)].copy()


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    out = {
        "n_months": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)) if len(y_true) else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else np.nan,
        "bias": float(np.mean(y_pred - y_true)) if len(y_true) else np.nan,
        "r2": np.nan,
    }
    if len(y_true) >= 2 and np.nanstd(y_true) > 0:
        out["r2"] = float(r2_score(y_true, y_pred))
    return out


def add_seasonal_shading(ax):
    seasons = [
        (0.5, 2.5, "#d9ecff"),   # winter: Jan-Feb
        (2.5, 5.5, "#e4f4df"),   # spring: Mar-May
        (5.5, 8.5, "#fff2bf"),   # summer: Jun-Aug
        (8.5, 11.5, "#fde2c2"),  # autumn: Sep-Nov
        (11.5, 12.5, "#d9ecff"), # winter: Dec
    ]
    for start, end, color in seasons:
        ax.axvspan(start, end, color=color, alpha=0.32, zorder=0, linewidth=0)


def plot_station_constituent(comp: pd.DataFrame, station_id: str, variable: str, out_dir: str):
    d = comp[(comp["wqms_id"].astype(str) == str(station_id)) & (comp["variable"] == variable)].sort_values("month")
    d = d[d["obs"].notna() & d["prediction"].notna()].copy()
    if len(d) < MIN_MONTHS_PER_PLOT:
        return None

    metrics = compute_metrics(d["obs"].to_numpy(float), d["prediction"].to_numpy(float))
    d["residual"] = d["prediction"] - d["obs"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    observed_color = "#08306b"   # dark scientific blue
    predicted_color = "#009e9a"  # teal/turquoise
    residual_pos = "#2b8cbe"
    residual_neg = "#f28e2b"

    fig, (ax, ax_res) = plt.subplots(
        2,
        1,
        figsize=(11.8, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.15], "hspace": 0.08},
        constrained_layout=True,
    )

    for axis in (ax, ax_res):
        add_seasonal_shading(axis)
        axis.grid(True, color="#d0d0d0", linewidth=0.7, alpha=0.45)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    ax.plot(
        d["month"],
        d["obs"],
        marker="o",
        markersize=5.2,
        linewidth=2.8,
        label="Observed monthly median",
        color=observed_color,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )
    ax.plot(
        d["month"],
        d["prediction"],
        marker="s",
        markersize=4.8,
        linewidth=2.8,
        label="Leave-station-out prediction",
        color=predicted_color,
        markerfacecolor="white",
        markeredgewidth=1.5,
        zorder=3,
    )

    intake_label = None
    if "intake_name" in d.columns and d["intake_name"].notna().any():
        intake_label = str(d["intake_name"].dropna().iloc[0])
    title_location = intake_label if intake_label else f"Station {station_id}"
    ax.set_title(f"{variable} | {title_location}", loc="left", pad=10, fontweight="bold")
    ax.set_ylabel("Concentration")
    ax.legend(loc="upper left", frameon=False)

    metrics_text = (
        f"RMSE  {metrics['rmse']:.3g}\n"
        f"MAE   {metrics['mae']:.3g}\n"
        f"Bias  {metrics['bias']:.3g}\n"
        f"R$^2$   {metrics['r2']:.3g}"
    )
    ax.text(
        0.985,
        0.965,
        metrics_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        linespacing=1.25,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.82},
    )

    residual_colors = np.where(d["residual"] >= 0, residual_pos, residual_neg)
    ax_res.bar(d["month"], d["residual"], width=0.58, color=residual_colors, alpha=0.82, edgecolor="white", linewidth=0.7)
    ax_res.axhline(0, color="#333333", linewidth=1.0)
    ax_res.set_ylabel("Residual\nPred - Obs", fontsize=12)
    ax_res.set_xlabel("Month")
    ax_res.set_xticks(range(1, 13))
    ax_res.set_xticklabels(month_labels)
    ax_res.set_xlim(0.5, 12.5)

    note = f"Station {station_id}"
    if "distance_to_intake_km" in d.columns and d["distance_to_intake_km"].notna().any():
        note += f" | nearest WQMS distance {float(d['distance_to_intake_km'].dropna().iloc[0]):.1f} km"
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=10, color="#555555")

    out_png = os.path.join(out_dir, f"leave_station_out_{safe_name(station_id)}_{safe_name(variable)}.png")
    fig.savefig(out_png, dpi=EXPORT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"wqms_id": station_id, "variable": variable, "plot_path": out_png, **metrics}


def validate_constituent_leave_station_out(variable: str,
                                           obs_path: str,
                                           holdout_obs_var: pd.DataFrame,
                                           holdout_station_ids: set[str],
                                           feat_df: pd.DataFrame,
                                           static_cols: list[str],
                                           monthly_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    train_df = build_training_table(
        obs_path=obs_path,
        feat_df=feat_df,
        static_cols=static_cols,
        monthly_cols=monthly_cols,
        train_scope=TRAIN_SCOPE,
    )
    if train_df.empty:
        return pd.DataFrame(), {"variable": variable, "status": "skip_empty_training_table", "train_rows": 0}

    train_df["wqms_id"] = train_df["wqms_id"].astype(str)
    n_before = len(train_df)
    train_df = train_df[~train_df["wqms_id"].isin(holdout_station_ids)].copy()
    train_df = train_df[train_df["obs"].notna()].copy()

    if train_df.empty:
        return pd.DataFrame(), {
            "variable": variable,
            "status": "skip_no_training_rows_after_holdout_removal",
            "train_rows": 0,
            "removed_holdout_rows": n_before,
        }

    feature_cols = prepare_feature_columns(train_df)
    if not feature_cols:
        return pd.DataFrame(), {
            "variable": variable,
            "status": "skip_no_features",
            "train_rows": len(train_df),
            "removed_holdout_rows": n_before - len(train_df),
        }

    mainbas_medians, global_medians = fit_imputation_stats(train_df, feature_cols)
    x_train = apply_imputation(train_df, feature_cols, mainbas_medians, global_medians)
    y_train = train_df["obs"].astype(np.float32).to_numpy()
    model, model_name = fit_random_forest(x_train, y_train)

    target = holdout_obs_var[["hydrobasin_level12", "month"]].drop_duplicates().copy()
    target["hydrobasin_level12"] = normalize_key(target["hydrobasin_level12"])
    needed_basins = set(target["hydrobasin_level12"].dropna().astype(str).tolist())
    pred_feat_use = feat_df[feat_df["hydrobasin_level12"].isin(needed_basins)].copy()

    pred_parts = []
    for month, month_target in target.groupby("month"):
        pred_tbl = build_predict_table_for_month(
            feat_df=pred_feat_use,
            static_cols=static_cols,
            monthly_cols=monthly_cols,
            month=int(month),
        )
        pred_tbl = pred_tbl[pred_tbl["hydrobasin_level12"].isin(month_target["hydrobasin_level12"])].copy()
        if pred_tbl.empty:
            continue

        for col in feature_cols:
            if col not in pred_tbl.columns:
                pred_tbl[col] = np.nan

        x_pred = apply_imputation(pred_tbl, feature_cols, mainbas_medians, global_medians)
        y_pred = predict_random_forest(model, x_pred)
        pred_parts.append(pd.DataFrame({
            "hydrobasin_level12": pred_tbl["hydrobasin_level12"].values,
            "month": int(month),
            "variable": variable,
            "prediction": y_pred,
        }))

    pred_long = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    if pred_long.empty:
        return pd.DataFrame(), {
            "variable": variable,
            "status": "skip_no_prediction_rows",
            "train_rows": len(train_df),
            "removed_holdout_rows": n_before - len(train_df),
            "feature_count": len(feature_cols),
            "model": model_name,
        }

    comp = holdout_obs_var.merge(
        pred_long,
        on=["hydrobasin_level12", "variable", "month"],
        how="left",
        validate="m:1",
    )
    comp = comp[comp["prediction"].notna()].copy()
    status = "ok" if not comp.empty else "skip_no_overlap_after_prediction"
    return comp, {
        "variable": variable,
        "status": status,
        "train_rows": len(train_df),
        "removed_holdout_rows": n_before - len(train_df),
        "feature_count": len(feature_cols),
        "holdout_rows": len(holdout_obs_var),
        "comparison_rows": len(comp),
        "holdout_stations": holdout_obs_var["wqms_id"].nunique(),
        "holdout_basins": holdout_obs_var["hydrobasin_level12"].nunique(),
        "model": model_name,
    }


def main():
    print(f"IS_COLAB: {IS_COLAB}")
    print(f"OBS_DIR: {OBS_DIR}")
    print(f"SITE_INFO: {SITE_INFO}")
    print(f"SCENARIO: {SCENARIO}")

    if TRAIN_SCOPE == "EU":
        features_path, shared_path = FEATURES_EUROPE, SHARED_EUROPE
    elif TRAIN_SCOPE == "WORLD":
        features_path, shared_path = FEATURES_WORLD, SHARED_WORLD
    else:
        raise ValueError("TRAIN_SCOPE moet 'EU' of 'WORLD' zijn.")

    print(f"TRAIN_SCOPE: {TRAIN_SCOPE}")
    print(f"FEATURES: {features_path}")
    print(f"ML backend: {'cuML GPU' if USE_GPU_IF_AVAILABLE and HAS_CUML else 'sklearn CPU'}")

    feat_df, static_cols, monthly_cols = load_basin_features(features_path, shared_path)
    print(f"Features geladen: basins={len(feat_df):,}, static={len(static_cols)}, monthly={len(monthly_cols)}")

    obs = load_observations(OBS_DIR, None)
    if obs.empty:
        raise ValueError(f"Geen observaties gevonden in {OBS_DIR}")

    obs_for_station_selection = obs
    if CONSTITUENTS is not None:
        requested_keys = {constituent_key(v) for v in CONSTITUENTS}
        obs_for_station_selection = obs[
            obs["variable"].astype(str).map(constituent_key).isin(requested_keys)
        ].copy()
        if obs_for_station_selection.empty:
            available_preview = sorted(obs["variable"].dropna().astype(str).unique().tolist())[:80]
            raise ValueError(
                "Geen observaties gevonden voor de handmatige CONSTITUENTS. "
                f"Gevraagd: {CONSTITUENTS}. "
                f"Beschikbare constituents, eerste {len(available_preview)}: {available_preview}"
            )
        print(
            "Stationselectie gebruikt alleen observaties voor handmatige top-5: "
            f"{CONSTITUENTS} | rows={len(obs_for_station_selection):,}, "
            f"stations={obs_for_station_selection['wqms_id'].nunique():,}"
        )

    holdout_obs_all = select_holdout_observations(obs_for_station_selection)
    if holdout_obs_all.empty:
        raise ValueError("Geen holdout observaties geselecteerd. Vul HOLDOUT_STATION_IDS of HYDROBASIN_KEYS in.")

    obs_files = observation_file_map(OBS_DIR)
    obs_files_by_key = {constituent_key(k): v for k, v in obs_files.items()}
    holdout_obs_all = holdout_obs_all[
        holdout_obs_all["variable"].astype(str).map(constituent_key).isin(obs_files_by_key)
    ].copy()
    holdout_obs_all = holdout_obs_all[holdout_obs_all["hydrobasin_level12"].isin(feat_df["hydrobasin_level12"])].copy()
    if holdout_obs_all.empty:
        raise ValueError("Holdout observaties hebben geen match met feature-basins.")

    chosen_vars = choose_validation_constituents(holdout_obs_all, TOP_N_CONSTITUENTS)
    if not chosen_vars:
        raise ValueError("Geen constituents geselecteerd voor validatie.")

    holdout_obs = holdout_obs_all[holdout_obs_all["variable"].isin(chosen_vars)].copy()
    holdout_station_ids = set(holdout_obs_all["wqms_id"].astype(str).unique().tolist())
    print(f"Holdout stations uitgesloten uit training: {len(holdout_station_ids)}")
    print(f"Top-{len(chosen_vars)} constituents voor leave-station-out validatie: {chosen_vars}")

    out_dir = os.path.join(OUT_DIR, SCENARIO)
    plot_dir = os.path.join(out_dir, "plots")
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ranking_path = os.path.join(out_dir, "top5_constituent_selection.csv")
    if not LAST_TOPN_RANKING.empty:
        LAST_TOPN_RANKING.to_csv(ranking_path, index=False)
        print(f"Top-5 selectietabel: {ranking_path}")

    selected_obs_path = os.path.join(out_dir, "selected_holdout_observations.csv")
    holdout_obs.to_csv(selected_obs_path, index=False)

    comparison_path = os.path.join(out_dir, "leave_station_out_observed_vs_predicted_long.csv")
    log_path = os.path.join(out_dir, "leave_station_out_training_log.csv")
    for old_path in [comparison_path, log_path]:
        if os.path.exists(old_path):
            os.remove(old_path)

    comparison_parts = []
    training_logs = []
    for i, variable in enumerate(chosen_vars, start=1):
        print(f"\n[{i}/{len(chosen_vars)}] Leave-station-out model voor {variable}")
        holdout_obs_var = holdout_obs[holdout_obs["variable"] == variable].copy()
        comp_var, log_row = validate_constituent_leave_station_out(
            variable=variable,
            obs_path=obs_files_by_key[constituent_key(variable)],
            holdout_obs_var=holdout_obs_var,
            holdout_station_ids=holdout_station_ids,
            feat_df=feat_df,
            static_cols=static_cols,
            monthly_cols=monthly_cols,
        )
        training_logs.append(log_row)
        pd.DataFrame([log_row]).to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

        if not comp_var.empty:
            comparison_parts.append(comp_var)
            comp_so_far = pd.concat(comparison_parts, ignore_index=True)
            comp_so_far.to_csv(comparison_path, index=False)
            print(f"✅ {variable}: comparison rows={len(comp_var):,}, train rows={log_row.get('train_rows', 0):,}")
        else:
            print(f"⚠️ {variable}: overgeslagen ({log_row['status']})")

    if not comparison_parts:
        raise ValueError("Geen leave-station-out predictions gemaakt. Check logbestand voor oorzaak.")

    comp = pd.concat(comparison_parts, ignore_index=True)
    comp.to_csv(comparison_path, index=False)

    metrics_rows = []
    for (station_id, variable), _ in comp.groupby(["wqms_id", "variable"]):
        result = plot_station_constituent(comp, station_id, variable, out_dir=plot_dir)
        if result is not None:
            metrics_rows.append(result)
    metrics = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(out_dir, "leave_station_out_metrics.csv")
    metrics.to_csv(metrics_path, index=False)

    print("\n✅ Holdout station validatie klaar")
    print("Deze validatie is onafhankelijker: geselecteerde stations zijn uit de training verwijderd.")
    if not LAST_TOPN_RANKING.empty:
        print(f"Top-5 selectie: {ranking_path}")
    print(f"Geselecteerde holdout observaties: {selected_obs_path}")
    print(f"Vergelijkingstabel: {comparison_path}")
    print(f"Training log: {log_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Plots: {plot_dir}")
    print(f"Aantal plots: {len(metrics)}")


if __name__ == "__main__":
    main()
