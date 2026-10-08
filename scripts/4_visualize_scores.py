# -*- coding: utf-8 -*-
"""4_visualize_scores

Visualisaties voor de gecombineerde WORLD + EU_train_NL_test analyse.

Maakt:
- performance top-25 figuren per analysebron
- CV heatmaps per analysebron
- WORLD-vs-EU/NL transfervergelijkingen
- score-distributies en coverage/stability figuren
- feature-importance figuren voor de top-25 best presterende constituents per analysebron
"""

from google.colab import drive
drive.mount('/content/drive')

#!/usr/bin/env python3
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# =====================================================
# CONFIG
# =====================================================
BASE = "/content/drive/MyDrive/SamuelRebel_thesis/T2/T3/output"

RESULT_CANDIDATES = {
    "WORLD_all": [
        f"{BASE}/results_ML/all_pollutants/world/all_results.csv",
        f"{BASE}/results_ML/world/all_results.csv",
    ],
    "EU_train_NL_test": [
        f"{BASE}/results_ML/all_pollutants/europe_nederland/all_results.csv",
        f"{BASE}/results_ML/all_pollutants/europe_nederland/all_results_NL.csv",
        f"{BASE}/results_ML/europe_nederland/all_results.csv",
        f"{BASE}/results_ML/europe_nederland/all_results_NL.csv",
    ],
}

FI_CANDIDATES = {
    "WORLD_all": [
        f"{BASE}/results_ML/all_pollutants/world/feature_importance_summary.csv",
        f"{BASE}/results_ML/all_pollutants/world/feature_importance.csv",
        f"{BASE}/results_ML/world/feature_importance_summary.csv",
        f"{BASE}/results_ML/world/feature_importance.csv",
    ],
    "EU_train_NL_test": [
        f"{BASE}/results_ML/all_pollutants/europe_nederland/feature_importance_summary.csv",
        f"{BASE}/results_ML/all_pollutants/europe_nederland/feature_importance.csv",
        f"{BASE}/results_ML/europe_nederland/feature_importance_summary.csv",
        f"{BASE}/results_ML/europe_nederland/feature_importance.csv",
    ],
}

SOURCES = ["WORLD_all", "EU_train_NL_test"]
SOURCE_LABELS = {
    "WORLD_all": "WORLD",
    "EU_train_NL_test": "EU train -> NL test",
}
SOURCE_COLORS = {
    "WORLD_all": "#1f77b4",
    "EU_train_NL_test": "#2ca02c",
}

TOP_N = 25
TOP_N_FEATURES = 20
MIN_FOLDS_PER_CV = 2
N_MIN = {
    "WORLD_all": 500,
    "EU_train_NL_test": 110,
}
EXCLUDE_CV_TYPES = {"Country CV", "Country"}

RANK_METRIC = "R2"
SECONDARY_METRIC = "NRMSE_std"

OUTDIR = f"{BASE}/scores/plots"
EXPORT_CSV = True
EXPORT_DPI = 320

# =====================================================
# FEATURE FILTERING + READABLE LABELS
# =====================================================
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


BASE_LABELS = {
    "dis_m3": "Natural river discharge",
    "run_mm": "Runoff depth",
    "inu_pc": "Inundated area",
    "lka_pc": "Lake area",
    "lkv_mc": "Lake volume",
    "rev_mc": "Reservoir volume",
    "dor_pc": "Degree of river regulation",
    "ria_ha": "River area",
    "riv_tc": "River volume",
    "gwt_cm": "Groundwater table depth",
    "ele_mt": "Elevation",
    "slp_dg": "Terrain slope",
    "sgr_dk": "Stream gradient",
    "tmp_dc": "Air temperature",
    "pre_mm": "Precipitation",
    "pet_mm": "Potential evapotranspiration",
    "aet_mm": "Actual evapotranspiration",
    "ari_ix": "Aridity index",
    "cmi_ix": "Climate moisture index",
    "snw_pc": "Snow cover",
    "glc_pc": "Land-cover fraction",
    "pnv_pc": "Potential natural vegetation fraction",
    "wet_pc": "Wetland fraction",
    "for_pc": "Forest cover",
    "crp_pc": "Cropland cover",
    "pst_pc": "Pasture cover",
    "ire_pc": "Irrigated area",
    "gla_pc": "Glacier cover",
    "prm_pc": "Permafrost extent",
    "pac_pc": "Protected area coverage",
    "cly_pc": "Soil clay fraction",
    "slt_pc": "Soil silt fraction",
    "snd_pc": "Soil sand fraction",
    "soc_th": "Soil organic carbon",
    "swc_pc": "Soil water content",
    "kar_pc": "Karst area",
    "ero_kh": "Soil erosion",
    "pop_ct": "Population count",
    "ppd_pk": "Population density",
    "urb_pc": "Urban area",
    "nli_ix": "Night-time lights index",
    "rdd_mk": "Road density",
    "hft_ix": "Human footprint index",
    "gdp_ud": "Gross domestic product",
    "hdi_ix": "Human Development Index",
    "dist_to_wwtp_km": "Distance to nearest wastewater treatment plant",
    "discharge": "Monthly discharge at basin support point",
    "watertemp": "Monthly water temperature at basin support point",
}

UNITS = {
    "dis_m3": "m3/s", "run_mm": "mm/yr", "inu_pc": "%", "lka_pc": "%",
    "lkv_mc": "million m3", "rev_mc": "million m3", "dor_pc": "%",
    "ria_ha": "ha", "gwt_cm": "cm", "ele_mt": "m", "slp_dg": "degrees",
    "tmp_dc": "0.1 deg C", "pre_mm": "mm", "pet_mm": "mm", "aet_mm": "mm",
    "snw_pc": "%", "glc_pc": "%", "pnv_pc": "%", "wet_pc": "%",
    "for_pc": "%", "crp_pc": "%", "pst_pc": "%", "ire_pc": "%",
    "gla_pc": "%", "prm_pc": "%", "pac_pc": "%", "cly_pc": "%",
    "slt_pc": "%", "snd_pc": "%", "swc_pc": "%", "urb_pc": "%",
    "ppd_pk": "people/km2", "dist_to_wwtp_km": "km",
}

SUFFIX_LABELS = {
    "syr": "annual value in sub-basin",
    "uyr": "annual value for upstream catchment",
    "sav": "average in sub-basin",
    "uav": "average for upstream catchment",
    "smn": "minimum in sub-basin",
    "umn": "minimum for upstream catchment",
    "smx": "maximum in sub-basin",
    "umx": "maximum for upstream catchment",
    "slt": "long-term low value in sub-basin",
    "ult": "long-term low value for upstream catchment",
    "sse": "share in sub-basin",
    "use": "share for upstream catchment",
    "ssu": "sum in sub-basin",
    "usu": "sum for upstream catchment",
    "pyr": "annual average",
    "pmn": "minimum monthly average",
    "pmx": "maximum monthly average",
    "pva": "variability indicator",
    "sg1": "class 1 in sub-basin",
    "ug1": "class 1 for upstream catchment",
    "sg2": "class 2 in sub-basin",
    "ug2": "class 2 for upstream catchment",
}

LAND_COVER_CLASSES = {
    **{f"s{i:02d}": f"class {i:02d} in sub-basin" for i in range(1, 23)},
    **{f"u{i:02d}": f"class {i:02d} for upstream catchment" for i in range(1, 23)},
}
MONTH_SUFFIXES = {f"s{i:02d}": f"month {i:02d} in sub-basin" for i in range(1, 13)}
MONTH_SUFFIXES.update({f"u{i:02d}": f"month {i:02d} for upstream catchment" for i in range(1, 13)})


def split_hydroatlas_code(feature: str) -> tuple[str, str | None]:
    f = str(feature)
    for suffix in sorted(set(SUFFIX_LABELS) | set(LAND_COVER_CLASSES) | set(MONTH_SUFFIXES), key=len, reverse=True):
        token = "_" + suffix
        if f.endswith(token):
            return f[: -len(token)], suffix
    return f, None


def feature_label(feature: str, include_code: bool = True) -> str:
    raw = str(feature)
    base, suffix = split_hydroatlas_code(raw)
    base_label = BASE_LABELS.get(base, raw.replace("_", " "))

    if suffix is None:
        label = base_label
    elif base in {"pre_mm", "pet_mm", "aet_mm", "tmp_dc", "snw_pc", "cmi_ix", "discharge", "watertemp"} and suffix in MONTH_SUFFIXES:
        label = f"{base_label}, {MONTH_SUFFIXES[suffix]}"
    elif base in {"glc_pc", "pnv_pc", "wet_pc"} and suffix in LAND_COVER_CLASSES:
        label = f"{base_label}, {LAND_COVER_CLASSES[suffix]}"
    else:
        suffix_label = SUFFIX_LABELS.get(suffix) or LAND_COVER_CLASSES.get(suffix) or MONTH_SUFFIXES.get(suffix)
        label = f"{base_label}, {suffix_label}" if suffix_label else base_label

    unit = UNITS.get(base)
    if unit:
        label = f"{label} [{unit}]"
    if include_code:
        label = f"{label} ({raw})"
    return label



# =====================================================
# STYLE + HELPERS
# =====================================================
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams.update({
    "figure.facecolor": "#f7f7f7",
    "axes.facecolor": "#fcfcfc",
    "axes.edgecolor": "#444444",
    "axes.titleweight": "bold",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "grid.color": "#d9d9d9",
    "grid.alpha": 0.35,
})


def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def resolve_existing_path(candidates: list[str], label: str, required: bool = True) -> str | None:
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ {label}: {path}")
            return path
    msg = f"Geen bestand gevonden voor {label}. Geprobeerd: {candidates}"
    if required:
        raise FileNotFoundError(msg)
    print(f"⚠️ {msg}")
    return None


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")[:140] or "unknown"


def simplify_cv_type(cv):
    if pd.isna(cv):
        return cv
    return re.sub(r"\s*\(.*\)\s*$", "", str(cv)).strip()


def metric_label(metric: str) -> str:
    return "R²" if metric == "R2" else metric


def is_lower_better(metric: str) -> bool:
    m = metric.upper()
    return m.startswith("NRMSE") or m.endswith("RMSE")


def savefig(name: str):
    out = os.path.join(OUTDIR, name)
    plt.savefig(out, dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()


# =====================================================
# LOAD + AGGREGATE PERFORMANCE
# =====================================================
def load_results(path: str, source_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Obs." in df.columns and "Obs" not in df.columns:
        df = df.rename(columns={"Obs.": "Obs"})

    required = {"Variable", "CV_type", "Fold", "Obs", "R2", "RMSE", "NRMSE_std"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ontbrekende kolommen in {path}: {missing}")

    for col in ["Fold", "Obs", "R2", "RMSE", "NRMSE_mean", "NRMSE_range", "NRMSE_std"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Source"] = source_label
    df["Source_label"] = SOURCE_LABELS.get(source_label, source_label)
    df["CV_type"] = df["CV_type"].apply(simplify_cv_type)
    df = df[~df["CV_type"].isin(EXCLUDE_CV_TYPES)].copy()
    return df


def load_all_results() -> pd.DataFrame:
    frames = []
    for source in SOURCES:
        path = resolve_existing_path(RESULT_CANDIDATES[source], f"results {source}")
        frames.append(load_results(path, source))
    return pd.concat(frames, ignore_index=True)


def aggregate_cv_level(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    out = (
        df.groupby(["Source", "Source_label", "Variable", "CV_type"], as_index=False)
          .agg(
              metric_med=(metric, "median"),
              metric_q25=(metric, lambda s: np.nanquantile(s, 0.25)),
              metric_q75=(metric, lambda s: np.nanquantile(s, 0.75)),
              n_med=("Obs", "median"),
              folds=("Fold", "nunique"),
          )
    )
    out["n_min_required"] = out["Source"].map(N_MIN)
    return out[(out["n_med"] >= out["n_min_required"]) & (out["folds"] >= MIN_FOLDS_PER_CV)].copy()


def aggregate_variable_level(df_cv: pd.DataFrame) -> pd.DataFrame:
    return (
        df_cv.groupby(["Source", "Source_label", "Variable"], as_index=False)
             .agg(
                 metric_overall=("metric_med", "median"),
                 metric_spread=("metric_med", lambda s: np.nanquantile(s, 0.75) - np.nanquantile(s, 0.25)),
                 metric_min=("metric_med", "min"),
                 metric_max=("metric_med", "max"),
                 n_med=("n_med", "median"),
                 cv_types=("CV_type", "nunique"),
             )
    )


def top_n_by_source(df_var: pd.DataFrame, source: str, metric: str, top_n: int) -> pd.DataFrame:
    return (
        df_var[df_var["Source"] == source]
        .sort_values("metric_overall", ascending=is_lower_better(metric))
        .head(top_n)
        .copy()
    )


def top_variables_by_source(df_var: pd.DataFrame, metric: str, top_n: int) -> dict[str, list[str]]:
    return {source: top_n_by_source(df_var, source, metric, top_n)["Variable"].tolist() for source in SOURCES}


# =====================================================
# PERFORMANCE PLOTS
# =====================================================
def plot_top_bar(df_var: pd.DataFrame, source: str, metric: str, top_n: int):
    top = top_n_by_source(df_var, source, metric, top_n)
    if top.empty:
        return
    top = top.sort_values("metric_overall", ascending=is_lower_better(metric)).iloc[::-1]

    plt.figure(figsize=(13, max(8, 0.38 * len(top))))
    ax = sns.barplot(data=top, x="metric_overall", y="Variable", color=SOURCE_COLORS[source], alpha=0.90)
    xlim = ax.get_xlim()
    x_pad = (xlim[1] - xlim[0]) * 0.015 if xlim[1] != xlim[0] else 0.01
    for i, row in enumerate(top.itertuples(index=False)):
        ax.text(row.metric_overall + x_pad, i, f"n~{int(row.n_med)} | cv={int(row.cv_types)}", va="center", fontsize=8, color="#333333")

    if metric == "R2":
        plt.axvline(0, linestyle="--", linewidth=1, color="#222222", alpha=0.65)
    plt.title(f"Top {len(top)} constituents by {metric_label(metric)} ({SOURCE_LABELS[source]})")
    plt.xlabel(f"Robust median {metric_label(metric)} across CV-types")
    plt.ylabel("Constituent")
    plt.tight_layout()
    savefig(f"top_{top_n}_{metric}_{source}.png")


def plot_cv_heatmap(df_cv: pd.DataFrame, df_var: pd.DataFrame, source: str, metric: str, top_n: int):
    top_vars = top_n_by_source(df_var, source, metric, top_n)["Variable"].tolist()
    if not top_vars:
        return
    sub = df_cv[(df_cv["Source"] == source) & (df_cv["Variable"].isin(top_vars))].copy()
    piv = sub.pivot_table(index="Variable", columns="CV_type", values="metric_med", aggfunc="median").reindex(top_vars)

    plt.figure(figsize=(13, max(8, 0.38 * len(top_vars))))
    sns.heatmap(
        piv,
        cmap="mako_r" if is_lower_better(metric) else "crest",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="#efefef",
        cbar_kws={"label": f"Median {metric_label(metric)}"},
    )
    plt.title(f"Top-{len(top_vars)} constituents per CV-type ({SOURCE_LABELS[source]})")
    plt.xlabel("CV-type")
    plt.ylabel("Constituent")
    plt.tight_layout()
    savefig(f"heatmap_top_{top_n}_{metric}_{source}.png")


def plot_transfer_dumbbell(df_var: pd.DataFrame, metric: str, top_n: int):
    w = df_var[df_var["Source"] == "WORLD_all"][["Variable", "metric_overall", "n_med"]].rename(columns={"metric_overall": "metric_world", "n_med": "n_world"})
    n = df_var[df_var["Source"] == "EU_train_NL_test"][["Variable", "metric_overall", "n_med"]].rename(columns={"metric_overall": "metric_nl", "n_med": "n_nl"})
    m = w.merge(n, on="Variable", how="inner")
    if m.empty:
        return
    m["delta_nl_minus_world"] = m["metric_nl"] - m["metric_world"]
    m = m.sort_values("metric_world", ascending=is_lower_better(metric)).head(top_n).reset_index(drop=True)

    y = np.arange(len(m))
    plt.figure(figsize=(13, max(8, 0.42 * len(m))))
    for i, row in m.iterrows():
        plt.plot([row["metric_world"], row["metric_nl"]], [i, i], color="#8c8c8c", linewidth=1.5, alpha=0.8)
    plt.scatter(m["metric_world"], y, color=SOURCE_COLORS["WORLD_all"], s=60, label="WORLD", zorder=3)
    plt.scatter(m["metric_nl"], y, color=SOURCE_COLORS["EU_train_NL_test"], s=60, label="EU train -> NL test", zorder=3)
    plt.yticks(y, m["Variable"])
    if metric == "R2":
        plt.axvline(0, linestyle="--", linewidth=1, color="#222222", alpha=0.65)
    plt.xlabel(metric_label(metric))
    plt.title(f"WORLD vs EU/NL transfer comparison for top-{len(m)} WORLD constituents")
    plt.legend(loc="best", frameon=True)
    plt.tight_layout()
    savefig(f"dumbbell_transfer_top_{top_n}_{metric}.png")


def plot_source_scatter(df_var: pd.DataFrame, metric: str):
    w = df_var[df_var["Source"] == "WORLD_all"][["Variable", "metric_overall"]].rename(columns={"metric_overall": "WORLD"})
    n = df_var[df_var["Source"] == "EU_train_NL_test"][["Variable", "metric_overall"]].rename(columns={"metric_overall": "EU_NL"})
    m = w.merge(n, on="Variable", how="inner").dropna()
    if m.empty:
        return

    plt.figure(figsize=(9, 9))
    ax = sns.scatterplot(data=m, x="WORLD", y="EU_NL", s=70, alpha=0.75, color="#4c78a8")
    lo = float(np.nanmin([m["WORLD"].min(), m["EU_NL"].min()]))
    hi = float(np.nanmax([m["WORLD"].max(), m["EU_NL"].max()]))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#333333", linewidth=1)
    label_df = m.assign(abs_delta=(m["EU_NL"] - m["WORLD"]).abs()).sort_values("abs_delta", ascending=False).head(8)
    for row in label_df.itertuples(index=False):
        ax.text(row.WORLD, row.EU_NL, f" {row.Variable}", fontsize=8)
    plt.xlabel(f"WORLD median {metric_label(metric)}")
    plt.ylabel(f"EU train -> NL test median {metric_label(metric)}")
    plt.title(f"Transfer scatter: WORLD vs EU/NL ({metric_label(metric)})")
    plt.tight_layout()
    savefig(f"scatter_world_vs_eunl_{metric}.png")


def plot_skill_vs_coverage(df_var: pd.DataFrame, metric: str):
    if df_var.empty:
        return
    plt.figure(figsize=(11, 8))
    ax = sns.scatterplot(
        data=df_var,
        x="n_med",
        y="metric_overall",
        hue="Source_label",
        size="cv_types",
        sizes=(35, 170),
        alpha=0.78,
        palette={SOURCE_LABELS[k]: v for k, v in SOURCE_COLORS.items()},
    )
    if metric == "R2":
        plt.axhline(0, linestyle="--", linewidth=1, color="#222222", alpha=0.65)
    d_rank = df_var.sort_values("metric_overall", ascending=is_lower_better(metric)).groupby("Source").head(6)
    for row in d_rank.itertuples(index=False):
        ax.text(row.n_med, row.metric_overall, f" {row.Variable}", fontsize=8, color="#333333")
    plt.xscale("log")
    plt.xlabel("Median observations per fold (log-scale)")
    plt.ylabel(f"Robust median {metric_label(metric)}")
    plt.title("Skill vs data coverage per constituent")
    plt.tight_layout()
    savefig(f"skill_vs_coverage_{metric}.png")


def plot_metric_distribution(df: pd.DataFrame, metric: str):
    if metric not in df.columns:
        return
    plt.figure(figsize=(12, 7))
    sns.boxplot(data=df, x="CV_type", y=metric, hue="Source_label", palette={SOURCE_LABELS[k]: v for k, v in SOURCE_COLORS.items()})
    if metric == "R2":
        plt.axhline(0, linestyle="--", linewidth=1, color="#222222", alpha=0.65)
    plt.xlabel("CV-type")
    plt.ylabel(metric_label(metric))
    plt.title(f"Fold-level score distribution ({metric_label(metric)})")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    savefig(f"distribution_by_cv_{metric}.png")


def plot_stability(df_var: pd.DataFrame, metric: str, top_n: int):
    pieces = []
    for source in SOURCES:
        top = top_n_by_source(df_var, source, metric, top_n).copy()
        pieces.append(top)
    d = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    if d.empty:
        return
    d = d.sort_values("metric_spread", ascending=True).head(min(len(d), top_n * len(SOURCES)))
    plt.figure(figsize=(12, max(8, 0.32 * len(d))))
    sns.barplot(data=d, x="metric_spread", y="Variable", hue="Source_label", palette={SOURCE_LABELS[k]: v for k, v in SOURCE_COLORS.items()})
    plt.xlabel(f"IQR across CV-types ({metric_label(metric)})")
    plt.ylabel("Constituent")
    plt.title("Model stability across CV-types; lower is more stable")
    plt.tight_layout()
    savefig(f"stability_iqr_top_{top_n}_{metric}.png")


# =====================================================
# FEATURE IMPORTANCE
# =====================================================
def load_feature_importance_for_source(source: str) -> pd.DataFrame:
    path = resolve_existing_path(FI_CANDIDATES[source], f"feature importance {source}", required=False)
    if path is None:
        return pd.DataFrame()

    fi = pd.read_csv(path)
    if fi.empty:
        return pd.DataFrame()

    if "Obs." in fi.columns and "Obs" not in fi.columns:
        fi = fi.rename(columns={"Obs.": "Obs"})

    if "Feature_label" not in fi.columns and "Feature" in fi.columns:
        fi["Feature_label"] = fi["Feature"].map(feature_label)

    if "Importance_mean" not in fi.columns and "Importance" in fi.columns:
        group_cols = [c for c in ["Variable", "CV_type", "Feature", "Feature_label"] if c in fi.columns]
        agg_kwargs = {
            "Importance_mean": ("Importance", "mean"),
            "Importance_std": ("Importance", "std"),
        }
        if "Obs" in fi.columns:
            agg_kwargs["Obs_sum"] = ("Obs", "sum")
        fi = fi.groupby(group_cols, as_index=False).agg(**agg_kwargs)

    required = {"Variable", "CV_type", "Feature", "Feature_label", "Importance_mean"}
    missing = required - set(fi.columns)
    if missing:
        raise ValueError(f"Feature-importance mist kolommen voor {source}: {missing}")

    fi["Source"] = source
    fi["Source_label"] = SOURCE_LABELS.get(source, source)
    fi["CV_type"] = fi["CV_type"].apply(simplify_cv_type)
    fi["Importance_mean"] = pd.to_numeric(fi["Importance_mean"], errors="coerce")
    if "Importance_std" in fi.columns:
        fi["Importance_std"] = pd.to_numeric(fi["Importance_std"], errors="coerce")
    fi = fi[~fi["CV_type"].isin(EXCLUDE_CV_TYPES)].copy()
    return fi


def load_feature_importance() -> pd.DataFrame:
    frames = [load_feature_importance_for_source(source) for source in SOURCES]
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def aggregate_feature_importance(fi: pd.DataFrame) -> pd.DataFrame:
    if fi.empty:
        return fi
    return (
        fi.groupby(["Source", "Source_label", "Variable", "Feature", "Feature_label"], as_index=False)
          .agg(
              Importance_mean=("Importance_mean", "mean"),
              Importance_std=("Importance_mean", "std"),
              cv_types=("CV_type", "nunique"),
          )
          .sort_values(["Source", "Variable", "Importance_mean"], ascending=[True, True, False])
    )


def plot_feature_importance_per_constituent(fi_var: pd.DataFrame, source: str, variable: str, out_dir: str):
    top = fi_var.sort_values("Importance_mean", ascending=False).head(TOP_N_FEATURES).copy()
    if top.empty:
        return
    top = top.iloc[::-1]
    plt.figure(figsize=(12, max(7, 0.36 * len(top))))
    ax = sns.barplot(data=top, x="Importance_mean", y="Feature_label", color=SOURCE_COLORS[source], alpha=0.92)
    if "Importance_std" in top.columns and top["Importance_std"].notna().any():
        ax.errorbar(
            x=top["Importance_mean"],
            y=np.arange(len(top)),
            xerr=top["Importance_std"].fillna(0),
            fmt="none",
            ecolor="#333333",
            elinewidth=0.8,
            capsize=2,
        )
    plt.axvline(0, linestyle="--", linewidth=1, color="#222222", alpha=0.55)
    plt.title(f"Feature importance | {SOURCE_LABELS[source]} | {variable}")
    plt.xlabel("Mean permutation importance across CV folds/types")
    plt.ylabel("Feature")
    plt.tight_layout()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(out_dir, f"feature_importance_{safe_name(variable)}.png"), dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close()


def plot_recurring_top_features(fi_agg: pd.DataFrame, source: str):
    d = fi_agg[fi_agg["Source"] == source].copy()
    if d.empty:
        return
    recurring = (
        d.sort_values(["Variable", "Importance_mean"], ascending=[True, False])
         .groupby("Variable")
         .head(5)
         .groupby(["Feature", "Feature_label"], as_index=False)
         .agg(constituents_top5=("Variable", "nunique"), mean_importance=("Importance_mean", "mean"))
         .sort_values(["constituents_top5", "mean_importance"], ascending=[False, False])
         .head(25)
    )
    if recurring.empty:
        return
    plt.figure(figsize=(12, max(7, 0.36 * len(recurring))))
    sns.barplot(data=recurring.iloc[::-1], x="constituents_top5", y="Feature_label", color=SOURCE_COLORS[source], alpha=0.92)
    plt.title(f"Most recurring top-5 features ({SOURCE_LABELS[source]})")
    plt.xlabel("Number of top constituents where feature appears in top 5")
    plt.ylabel("Feature")
    plt.tight_layout()
    savefig(f"feature_importance_recurring_top_features_{source}.png")


def plot_feature_importance_heatmap(fi_agg: pd.DataFrame, source: str, variables: list[str]):
    d = fi_agg[(fi_agg["Source"] == source) & (fi_agg["Variable"].isin(variables))].copy()
    if d.empty:
        return
    top_features = (
        d.groupby(["Feature", "Feature_label"], as_index=False)["Importance_mean"]
         .mean()
         .sort_values("Importance_mean", ascending=False)
         .head(25)["Feature_label"]
         .tolist()
    )
    piv = d[d["Feature_label"].isin(top_features)].pivot_table(index="Feature_label", columns="Variable", values="Importance_mean", aggfunc="mean")
    piv = piv.reindex(top_features)
    if piv.empty:
        return
    plt.figure(figsize=(max(12, 0.45 * len(variables)), max(8, 0.34 * len(top_features))))
    sns.heatmap(piv, cmap="viridis", linewidths=0.25, linecolor="#eeeeee", cbar_kws={"label": "Mean permutation importance"})
    plt.title(f"Feature-importance heatmap for top constituents ({SOURCE_LABELS[source]})")
    plt.xlabel("Constituent")
    plt.ylabel("Feature")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    savefig(f"feature_importance_heatmap_top_{len(variables)}_{source}.png")


def plot_feature_importance_all(fi: pd.DataFrame, top_vars: dict[str, list[str]]):
    if fi.empty:
        print("⚠️ Geen feature-importance bestanden gevonden; FI-figuren overgeslagen.")
        return

    fi_agg = aggregate_feature_importance(fi)
    if EXPORT_CSV:
        fi.to_csv(os.path.join(OUTDIR, "table_feature_importance_cv.csv"), index=False)
        fi_agg.to_csv(os.path.join(OUTDIR, "table_feature_importance_by_constituent.csv"), index=False)
        (fi_agg[["Feature", "Feature_label"]]
         .drop_duplicates()
         .sort_values("Feature")
         .to_csv(os.path.join(OUTDIR, "table_feature_label_lookup.csv"), index=False))

    for source, variables in top_vars.items():
        source_dir = os.path.join(OUTDIR, "feature_importance_per_constituent", source)
        for variable in variables:
            d = fi_agg[(fi_agg["Source"] == source) & (fi_agg["Variable"] == variable)].copy()
            plot_feature_importance_per_constituent(d, source, variable, source_dir)
        plot_recurring_top_features(fi_agg[fi_agg["Variable"].isin(variables)], source)
        plot_feature_importance_heatmap(fi_agg, source, variables)


# =====================================================
# RUN
# =====================================================
def main():
    ensure_outdir(OUTDIR)

    df = load_all_results()
    for metric in [RANK_METRIC, SECONDARY_METRIC]:
        if metric not in df.columns:
            raise ValueError(f"Metric ontbreekt in resultaten: {metric}")

    df_cv = aggregate_cv_level(df, metric=RANK_METRIC)
    df_var = aggregate_variable_level(df_cv)
    top_vars = top_variables_by_source(df_var, RANK_METRIC, TOP_N)

    df_cv_sec = aggregate_cv_level(df, metric=SECONDARY_METRIC)
    df_var_sec = aggregate_variable_level(df_cv_sec)

    if EXPORT_CSV:
        df_cv.to_csv(os.path.join(OUTDIR, f"table_cv_{RANK_METRIC}.csv"), index=False)
        df_var.to_csv(os.path.join(OUTDIR, f"table_variable_{RANK_METRIC}.csv"), index=False)
        df_cv_sec.to_csv(os.path.join(OUTDIR, f"table_cv_{SECONDARY_METRIC}.csv"), index=False)
        df_var_sec.to_csv(os.path.join(OUTDIR, f"table_variable_{SECONDARY_METRIC}.csv"), index=False)

    for metric, cv_df, var_df in [(RANK_METRIC, df_cv, df_var), (SECONDARY_METRIC, df_cv_sec, df_var_sec)]:
        for source in SOURCES:
            plot_top_bar(var_df, source=source, metric=metric, top_n=TOP_N)
            plot_cv_heatmap(cv_df, var_df, source=source, metric=metric, top_n=TOP_N)
        plot_transfer_dumbbell(var_df, metric=metric, top_n=TOP_N)
        plot_source_scatter(var_df, metric=metric)
        plot_skill_vs_coverage(var_df, metric=metric)
        plot_metric_distribution(df, metric=metric)
        plot_stability(var_df, metric=metric, top_n=TOP_N)

    fi = load_feature_importance()
    plot_feature_importance_all(fi, top_vars=top_vars)

    print("\n✅ Visualization pipeline klaar")
    print(f"Plots opgeslagen in: {OUTDIR}")
    print("Feature-importance per constituent staat in: feature_importance_per_constituent/<source>/")


if __name__ == "__main__":
    main()
