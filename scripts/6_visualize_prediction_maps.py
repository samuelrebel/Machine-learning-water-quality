# -*- coding: utf-8 -*-
"""6_visualize_prediction_maps

Losstaand script voor kaartvisualisaties op basis van bestaande wide prediction CSV's.

Run dit NA 5_predictions.py. Dit script traint niets opnieuw en overschrijft geen
prediction wide files. Het leest alleen:
- predictions_<scenario>_wide.csv
- BasinATLAS geometrie
- optioneel all_results.csv voor top-25 selectie

en schrijft PNG-kaarten weg.
"""

#!/usr/bin/env python3
import os
import re
from pathlib import Path

try:
    from google.colab import drive
    drive.mount('/content/drive')
    IS_COLAB = True
except Exception:
    IS_COLAB = False

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# =====================================================
# CONFIG
# =====================================================
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

if IS_COLAB:
    BASE = "/content/drive/MyDrive/SamuelRebel_thesis/T2/"
    OUTPUT_ROOT = f"{BASE}T3/output"
    ATLAS_SHP = f"{BASE}/data/predictor_data/BasinAtlas/ShapeFiles/BasinATLAS_v10_lev12.shp"
else:
    BASE = str(SCRIPT_DIR)
    OUTPUT_ROOT = str(SCRIPT_DIR / "output")
    ATLAS_SHP = str(SCRIPT_DIR / "Data" / "BasinATLAS_v10_lev12.shp")

PREDICTION_ROOT = f"{OUTPUT_ROOT}/predictions_basin_monthly"
POINT_ATLAS_CACHE = f"{PREDICTION_ROOT}/_map_cache/basin_level12_points.csv"

SCENARIOS = {
    "EU_train_NL_predict": {
        "source": "EU_train_NL_test",
        "wide_file": "predictions_EU_train_NL_predict_wide.csv",
        "label": "EU train -> NL predict",
        "map_style": "polygon",
        "make_key_month_maps": True,
        "make_individual_month_maps": True,
        "make_12_panel_maps": False,
    },
    "WORLD_train_WORLD_predict": {
        "source": "WORLD_all",
        "wide_file": "predictions_WORLD_train_WORLD_predict_wide.csv",
        "label": "WORLD train -> WORLD predict",
        # Wereldkaarten op HydroBASINS level 12 zijn extreem zwaar.
        # Voor rapportfiguren op wereldschaal is een point overview veel sneller
        # en visueel meestal net zo informatief.
        "map_style": "point",
        "use_topn_from_scores": True,
        "top_n": 25,
        "make_key_month_maps": True,
        "make_individual_month_maps": True,
        "make_12_panel_maps": False,
    },
}

# Zet bijvoorbeeld op ("WORLD_train_WORLD_predict",) als je alleen world-kaarten wilt maken.
RUN_SCENARIOS = ("EU_train_NL_predict", "WORLD_train_WORLD_predict")

SCORE_CANDIDATES = {
    "WORLD_all": [
        f"{OUTPUT_ROOT}/results_ML/all_pollutants/world/all_results.csv",
        f"{OUTPUT_ROOT}/results_ML/world/all_results.csv",
    ],
    "EU_train_NL_test": [
        f"{OUTPUT_ROOT}/results_ML/all_pollutants/europe_nederland/all_results.csv",
        f"{OUTPUT_ROOT}/results_ML/all_pollutants/europe_nederland/all_results_NL.csv",
        f"{OUTPUT_ROOT}/results_ML/europe_nederland/all_results.csv",
        f"{OUTPUT_ROOT}/results_ML/europe_nederland/all_results_NL.csv",
    ],
}

# Standaard: maak kaarten voor alle constituents die in de wide prediction CSV staan.
# Zet USE_TOPN_FROM_SCORES=True als je bewust alleen de top-N uit all_results wilt plotten.
SELECTED_CONSTITUENTS = None  # bv. ["DO", "DOC"] of None voor alles uit wide CSV
TOP_N = 25
USE_TOPN_FROM_SCORES = False
RANK_METRIC = "R2"
EXCLUDE_CV_TYPES = {"Country CV", "Country"}
N_MIN = {
    "WORLD_all": 700,
    "EU_train_NL_test": 110,
}
MIN_FOLDS_PER_CV = 2

MAKE_KEY_MONTH_MAPS = True
KEY_MONTHS = (1, 7)
KEY_MONTH_NAMES = {1: "January", 7: "July"}
KEY_MONTH_CMAP = "plasma"

# Key-month maps zijn vergelijkingsfiguren: standaard januari en juli als winter/zomer contrast.
# Pas KEY_MONTHS aan als andere maanden relevanter zijn.
# De losse maandkaarten hieronder worden standaard voor alle 12 maanden opgeslagen.
MAKE_INDIVIDUAL_MONTH_MAPS = True
MAKE_12_PANEL_MAPS = False
MONTHLY_CMAP = "viridis"

# Plotmodus:
# - "polygon": exacte basin-polygonen, mooier voor NL/EU maar traag voor WORLD
# - "point": snelle globale overview met representatieve punten per basin
DEFAULT_MAP_STYLE = "polygon"
POINT_MARKER_SIZE_WORLD = 0.18
POINT_MARKER_SIZE_EU = 1.8
POINT_ALPHA = 0.86

# Resume voor maps: bestaande PNG's overslaan.
SKIP_EXISTING_MAPS = True

EXPORT_DPI_KEY_MONTHS = 240
EXPORT_DPI_MONTHLY = 200


# =====================================================
# HELPERS
# =====================================================
def normalize_basin_key(s: pd.Series) -> pd.Series:
    return s.astype("string").str.replace(r"\.0$", "", regex=True).str.strip()


def prediction_month_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if re.search(r"_(0[1-9]|1[0-2])$", str(c))]


def variable_from_month_col(col: str) -> str:
    return re.sub(r"_(0[1-9]|1[0-2])$", "", str(col))


def variables_in_wide(df: pd.DataFrame) -> list[str]:
    return sorted({variable_from_month_col(c) for c in prediction_month_columns(df)})


def wide_candidate_paths(scenario_name: str, wide_file: str) -> list[str]:
    return list(dict.fromkeys([
        os.path.join(PREDICTION_ROOT, scenario_name, wide_file),
        os.path.join(PREDICTION_ROOT, wide_file),
        os.path.join(OUTPUT_ROOT, wide_file),
    ]))


def read_wide_candidate(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Kan wide kandidaat niet lezen: {path} | {e}")
        return None
    if "hydrobasin_level12" not in df.columns:
        print(f"⚠️ Wide kandidaat mist hydrobasin_level12 en wordt genegeerd: {path}")
        return None
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df["hydrobasin_level12"] = normalize_basin_key(df["hydrobasin_level12"])
    df = df.drop_duplicates(subset=["hydrobasin_level12"], keep="first")
    return df


def load_best_wide_prediction(scenario_name: str, wide_file: str) -> tuple[pd.DataFrame, str]:
    candidates = []
    for path in wide_candidate_paths(scenario_name, wide_file):
        df = read_wide_candidate(path)
        if df is None:
            continue
        n_pred_cols = len(prediction_month_columns(df))
        candidates.append((n_pred_cols, path, df))
        print(f"Wide kandidaat: {path} | shape={df.shape} | prediction month columns={n_pred_cols}")

    if not candidates:
        raise FileNotFoundError(
            f"Geen bruikbare wide prediction CSV gevonden voor {scenario_name}. Geprobeerd: {wide_candidate_paths(scenario_name, wide_file)}"
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    n_pred_cols, path, df = candidates[0]
    if n_pred_cols == 0:
        raise ValueError(
            f"Beste wide CSV voor {scenario_name} heeft 0 prediction month columns: {path}. "
            "Verwacht kolommen zoals DO_01 ... DO_12."
        )
    print(f"✅ Gekozen wide CSV voor {scenario_name}: {path}")
    return df, path


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
    print(f"⚠️ Geen scorebestand gevonden voor {label}; top-N valt terug op wide-kolommen.")
    return None


def top_variables_from_scores(source: str, available_variables: set[str], top_n: int) -> list[str]:
    path = resolve_existing_path(SCORE_CANDIDATES[source], f"scores {source}")
    if path is None:
        return sorted(available_variables)[:top_n]

    df = pd.read_csv(path)
    if "Obs." in df.columns and "Obs" not in df.columns:
        df = df.rename(columns={"Obs.": "Obs"})

    required = {"Variable", "CV_type", "Fold", "Obs", RANK_METRIC}
    missing = required - set(df.columns)
    if missing:
        print(f"⚠️ Scorebestand mist kolommen {missing}; top-N valt terug op wide-kolommen.")
        return sorted(available_variables)[:top_n]

    for c in ["Fold", "Obs", "R2", "RMSE", "NRMSE_mean", "NRMSE_range", "NRMSE_std"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["CV_type"] = df["CV_type"].apply(simplify_cv_type)
    df = df[~df["CV_type"].isin(EXCLUDE_CV_TYPES)].copy()
    df = df[df["Variable"].isin(available_variables)].copy()
    if df.empty:
        print("⚠️ Geen overlap tussen score-constituents en wide prediction-kolommen; top-N valt terug op wide-kolommen.")
        return sorted(available_variables)[:top_n]

    cv_agg = (
        df.groupby(["Variable", "CV_type"], as_index=False)
          .agg(
              metric_med=(RANK_METRIC, "median"),
              n_med=("Obs", "median"),
              folds=("Fold", "nunique"),
          )
    )
    n_min = N_MIN.get(source, 0)
    cv_agg = cv_agg[(cv_agg["n_med"] >= n_min) & (cv_agg["folds"] >= MIN_FOLDS_PER_CV)].copy()
    if cv_agg.empty:
        print("⚠️ Geen constituents over na scorefilters; top-N valt terug op wide-kolommen.")
        return sorted(available_variables)[:top_n]

    var_agg = cv_agg.groupby("Variable", as_index=False).agg(metric_overall=("metric_med", "median"))
    top = (
        var_agg.sort_values("metric_overall", ascending=is_lower_better(RANK_METRIC))
               .head(top_n)["Variable"]
               .tolist()
    )
    return top


def wide_to_long_for_maps(wide_df: pd.DataFrame, variables: list[str], months: list[int] | tuple[int, ...] | None = None) -> pd.DataFrame:
    months = list(months) if months is not None else list(range(1, 13))
    value_cols = []
    for var in variables:
        value_cols.extend([f"{var}_{m:02d}" for m in months if f"{var}_{m:02d}" in wide_df.columns])
    value_cols = list(dict.fromkeys(value_cols))
    if not value_cols:
        return pd.DataFrame(columns=["hydrobasin_level12", "variable", "month", "prediction"])

    long = wide_df[["hydrobasin_level12"] + value_cols].melt(
        id_vars=["hydrobasin_level12"],
        var_name="var_month",
        value_name="prediction",
    )
    long = long[long["prediction"].notna()].copy()
    if long.empty:
        return pd.DataFrame(columns=["hydrobasin_level12", "variable", "month", "prediction"])

    long["variable"] = long["var_month"].str.replace(r"_(0[1-9]|1[0-2])$", "", regex=True)
    long["month"] = long["var_month"].str.extract(r"_(0[1-9]|1[0-2])$").astype(int)
    return long.drop(columns=["var_month"])[["hydrobasin_level12", "variable", "month", "prediction"]]


def load_atlas_geometry(path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Atlas shapefile niet gevonden: {path}")
    g = gpd.read_file(path)
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    else:
        g = g.to_crs("EPSG:4326")

    if "hydrobasin_level12" not in g.columns:
        if "HYBAS_ID" in g.columns:
            g["hydrobasin_level12"] = g["HYBAS_ID"]
        else:
            raise ValueError("Atlas geometrie mist hydrobasin sleutelkolommen")

    g["hydrobasin_level12"] = normalize_basin_key(g["hydrobasin_level12"])
    return g[["hydrobasin_level12", "geometry"]].drop_duplicates(subset=["hydrobasin_level12"]).copy()


def prepare_point_atlas(atlas_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Maak een lichte point-representatie voor snelle wereldkaarten."""
    points = atlas_gdf.geometry.representative_point()
    return pd.DataFrame({
        "hydrobasin_level12": atlas_gdf["hydrobasin_level12"].to_numpy(),
        "plot_x": points.x.to_numpy(),
        "plot_y": points.y.to_numpy(),
    })


def load_or_create_point_atlas(atlas_path: str, cache_path: str, atlas_gdf: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    if os.path.exists(cache_path):
        p = pd.read_csv(cache_path)
        required = {"hydrobasin_level12", "plot_x", "plot_y"}
        if required.issubset(p.columns):
            p["hydrobasin_level12"] = normalize_basin_key(p["hydrobasin_level12"])
            print(f"Point atlas cache geladen: {cache_path} | rows={len(p):,}")
            return p
        print(f"⚠️ Point atlas cache mist kolommen en wordt opnieuw gemaakt: {cache_path}")

    if atlas_gdf is None:
        atlas_gdf = load_atlas_geometry(atlas_path)
    p = prepare_point_atlas(atlas_gdf)
    Path(os.path.dirname(cache_path)).mkdir(parents=True, exist_ok=True)
    p.to_csv(cache_path, index=False)
    print(f"Point atlas cache opgeslagen: {cache_path}")
    return p


def plot_prediction_layer(ax,
                          sm: pd.DataFrame,
                          atlas_gdf: gpd.GeoDataFrame | None,
                          point_atlas: pd.DataFrame,
                          map_style: str,
                          cmap: str,
                          vmin: float,
                          vmax: float,
                          marker_size: float):
    """Plot predictions als polygonen of als snelle point overview."""
    if map_style == "point":
        p = point_atlas.merge(sm, on="hydrobasin_level12", how="inner")
        if p.empty:
            return None
        sc = ax.scatter(
            p["plot_x"],
            p["plot_y"],
            c=p["prediction"],
            s=marker_size,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            alpha=POINT_ALPHA,
            rasterized=True,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 85)
        return sc

    if atlas_gdf is None:
        raise ValueError("Polygon map_style vereist atlas_gdf, maar die is niet geladen.")
    g = atlas_gdf.merge(sm, on="hydrobasin_level12", how="inner")
    if g.empty:
        return None
    g.plot(column="prediction", cmap=cmap, linewidth=0.0, ax=ax, vmin=vmin, vmax=vmax)
    return True


def robust_limits(values: pd.Series) -> tuple[float, float] | None:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return None
    vmin = float(np.nanquantile(values, 0.02))
    vmax = float(np.nanquantile(values, 0.98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(values.min())
        vmax = float(values.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    return vmin, vmax


def plot_key_month_maps(df_long: pd.DataFrame,
                        atlas_gdf: gpd.GeoDataFrame,
                        point_atlas: pd.DataFrame,
                        scenario_name: str,
                        scenario_label: str,
                        out_dir: str,
                        map_style: str,
                        marker_size: float):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = 0
    for var in sorted(df_long["variable"].dropna().unique()):
        sub = df_long[(df_long["variable"] == var) & (df_long["month"].isin(KEY_MONTHS))].copy()
        if sub.empty:
            continue
        limits = robust_limits(sub["prediction"])
        if limits is None:
            continue
        vmin, vmax = limits

        fig, axes = plt.subplots(1, len(KEY_MONTHS), figsize=(7.2 * len(KEY_MONTHS), 4.8), constrained_layout=True)
        if len(KEY_MONTHS) == 1:
            axes = [axes]

        for ax, month in zip(axes, KEY_MONTHS):
            sm = sub[sub["month"] == month][["hydrobasin_level12", "prediction"]]
            month_name = KEY_MONTH_NAMES.get(month, f"Month {month:02d}")
            plotted = plot_prediction_layer(
                ax=ax,
                sm=sm,
                atlas_gdf=atlas_gdf,
                point_atlas=point_atlas,
                map_style=map_style,
                cmap=KEY_MONTH_CMAP,
                vmin=vmin,
                vmax=vmax,
                marker_size=marker_size,
            )
            if plotted is None:
                ax.set_title(f"{month_name}\nno data", fontsize=12)
                ax.axis("off")
                continue
            ax.set_title(month_name, fontsize=13, fontweight="bold")
            ax.set_axis_off()
            ax.set_aspect("equal")

        smap = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=KEY_MONTH_CMAP)
        smap.set_array([])
        cbar = fig.colorbar(smap, ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("Predicted concentration", fontsize=11)
        fig.suptitle(f"{var}: predicted basin concentration | {scenario_label}", fontsize=15, fontweight="bold")
        out_png = os.path.join(out_dir, f"key_months_{scenario_name}_{var}.png")
        if SKIP_EXISTING_MAPS and os.path.exists(out_png):
            plt.close(fig)
            continue
        fig.savefig(out_png, dpi=EXPORT_DPI_KEY_MONTHS, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        made += 1
        print(f"  saved: {out_png}")
    return made


def plot_individual_month_maps(df_long: pd.DataFrame,
                               atlas_gdf: gpd.GeoDataFrame,
                               point_atlas: pd.DataFrame,
                               scenario_name: str,
                               scenario_label: str,
                               out_dir: str,
                               map_style: str,
                               marker_size: float):
    """Sla iedere maand apart op als PNG in maps/."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = 0
    for var in sorted(df_long["variable"].dropna().unique()):
        sub = df_long[df_long["variable"] == var].copy()
        if sub.empty:
            continue
        limits = robust_limits(sub["prediction"])
        if limits is None:
            continue
        vmin, vmax = limits

        for month in range(1, 13):
            sm = sub[sub["month"] == month][["hydrobasin_level12", "prediction"]]
            if sm.empty:
                continue

            out_png = os.path.join(out_dir, f"map_{scenario_name}_{var}_{month:02d}.png")
            if SKIP_EXISTING_MAPS and os.path.exists(out_png):
                continue

            fig, ax = plt.subplots(1, 1, figsize=(9.5, 5.2), constrained_layout=True)
            plotted = plot_prediction_layer(
                ax=ax,
                sm=sm,
                atlas_gdf=atlas_gdf,
                point_atlas=point_atlas,
                map_style=map_style,
                cmap=MONTHLY_CMAP,
                vmin=vmin,
                vmax=vmax,
                marker_size=marker_size,
            )
            if plotted is None:
                plt.close(fig)
                continue
            ax.set_axis_off()
            ax.set_aspect("equal")
            ax.set_title(f"{var}: month {month:02d} | {scenario_label}", fontsize=14, fontweight="bold")

            smap = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=MONTHLY_CMAP)
            smap.set_array([])
            cbar = fig.colorbar(smap, ax=ax, fraction=0.025, pad=0.02)
            cbar.set_label("Predicted concentration", fontsize=11)

            fig.savefig(out_png, dpi=EXPORT_DPI_MONTHLY, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            made += 1
            print(f"  saved: {out_png}")
    return made


def plot_12_panel_maps(df_long: pd.DataFrame,
                       atlas_gdf: gpd.GeoDataFrame,
                       point_atlas: pd.DataFrame,
                       scenario_name: str,
                       scenario_label: str,
                       out_dir: str,
                       map_style: str,
                       marker_size: float):
    """Optionele overzichtsfiguur met alle 12 maanden in een paneel."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    made = 0
    for var in sorted(df_long["variable"].dropna().unique()):
        sub = df_long[df_long["variable"] == var].copy()
        if sub.empty:
            continue
        limits = robust_limits(sub["prediction"])
        if limits is None:
            continue
        vmin, vmax = limits

        out_png = os.path.join(out_dir, f"map_12panel_{scenario_name}_{var}.png")
        if SKIP_EXISTING_MAPS and os.path.exists(out_png):
            continue

        fig, axes = plt.subplots(3, 4, figsize=(20, 13))
        axes = axes.flatten()
        for month in range(1, 13):
            ax = axes[month - 1]
            sm = sub[sub["month"] == month][["hydrobasin_level12", "prediction"]]
            plotted = plot_prediction_layer(
                ax=ax,
                sm=sm,
                atlas_gdf=atlas_gdf,
                point_atlas=point_atlas,
                map_style=map_style,
                cmap=MONTHLY_CMAP,
                vmin=vmin,
                vmax=vmax,
                marker_size=marker_size,
            )
            if plotted is None:
                ax.set_title(f"Month {month:02d} (no data)")
                ax.axis("off")
                continue
            ax.set_title(f"Month {month:02d}")
            ax.axis("off")

        smap = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=MONTHLY_CMAP)
        smap.set_array([])
        cbar = fig.colorbar(smap, ax=axes.tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("Predicted concentration")
        fig.suptitle(f"{scenario_label} | {var} | monthly basin predictions", fontsize=16, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(out_png, dpi=EXPORT_DPI_MONTHLY, bbox_inches="tight")
        plt.close(fig)
        made += 1
        print(f"  saved: {out_png}")
    return made


# =====================================================
# RUN
# =====================================================
def main():
    print(f"IS_COLAB: {IS_COLAB}")
    print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
    print(f"PREDICTION_ROOT: {PREDICTION_ROOT}")
    print(f"ATLAS_SHP: {ATLAS_SHP}")
    print(f"RUN_SCENARIOS: {RUN_SCENARIOS}")
    print(f"USE_TOPN_FROM_SCORES: {USE_TOPN_FROM_SCORES}")
    print(f"MAKE_KEY_MONTH_MAPS: {MAKE_KEY_MONTH_MAPS}")
    print(f"MAKE_INDIVIDUAL_MONTH_MAPS: {MAKE_INDIVIDUAL_MONTH_MAPS}")
    print(f"MAKE_12_PANEL_MAPS: {MAKE_12_PANEL_MAPS}")
    print(f"SKIP_EXISTING_MAPS: {SKIP_EXISTING_MAPS}")

    scenario_items = [(name, cfg) for name, cfg in SCENARIOS.items() if not RUN_SCENARIOS or name in RUN_SCENARIOS]
    if not scenario_items:
        raise ValueError("RUN_SCENARIOS bevat geen geldige scenario-namen.")

    needs_polygon = any(cfg.get("map_style", DEFAULT_MAP_STYLE) == "polygon" for _, cfg in scenario_items)
    needs_point = any(cfg.get("map_style", DEFAULT_MAP_STYLE) == "point" for _, cfg in scenario_items)

    atlas = load_atlas_geometry(ATLAS_SHP) if needs_polygon else None
    if atlas is not None:
        print(f"Atlas geometries loaded: {len(atlas):,}")

    point_atlas = load_or_create_point_atlas(ATLAS_SHP, POINT_ATLAS_CACHE, atlas_gdf=atlas) if needs_point else pd.DataFrame()

    manifest_rows = []
    for scenario_name, cfg in scenario_items:
        print("\n" + "=" * 80)
        print(f"SCENARIO: {scenario_name}")
        print("=" * 80)

        wide_df, wide_path = load_best_wide_prediction(scenario_name, cfg["wide_file"])
        available = set(variables_in_wide(wide_df))
        print(f"Available constituents in wide CSV: {len(available)}")
        print(f"Examples: {sorted(available)[:10]}")

        if SELECTED_CONSTITUENTS is not None:
            selected = [v for v in SELECTED_CONSTITUENTS if v in available]
            missing_selected = [v for v in SELECTED_CONSTITUENTS if v not in available]
            if missing_selected:
                print(f"⚠️ SELECTED_CONSTITUENTS niet gevonden in wide CSV: {missing_selected}")
        elif cfg.get("use_topn_from_scores", USE_TOPN_FROM_SCORES):
            selected = top_variables_from_scores(cfg["source"], available, cfg.get("top_n", TOP_N))
        else:
            selected = sorted(available)
        print(f"Selected constituents for maps: {len(selected)}")
        print(f"Selected examples: {selected[:10]}")

        map_style = cfg.get("map_style", DEFAULT_MAP_STYLE)
        make_key = cfg.get("make_key_month_maps", MAKE_KEY_MONTH_MAPS)
        make_monthly = cfg.get("make_individual_month_maps", MAKE_INDIVIDUAL_MONTH_MAPS)
        make_12panel = cfg.get("make_12_panel_maps", MAKE_12_PANEL_MAPS)
        marker_size = POINT_MARKER_SIZE_WORLD if scenario_name.startswith("WORLD") else POINT_MARKER_SIZE_EU

        if make_monthly or make_12panel:
            months_needed = list(range(1, 13))
        elif make_key:
            months_needed = list(KEY_MONTHS)
        else:
            months_needed = []

        print(f"Map style: {map_style}")
        print(f"Scenario maps: key_month={make_key}, individual_months={make_monthly}, panel_12={make_12panel}")
        print(f"Months loaded for maps: {months_needed}")

        df_long = wide_to_long_for_maps(wide_df, selected, months=months_needed)
        print(f"Long map rows: {len(df_long):,}")
        if df_long.empty:
            print("⚠️ Geen long map rows; deze scenario wordt overgeslagen.")
            continue

        scenario_out = os.path.join(PREDICTION_ROOT, scenario_name)
        key_dir = os.path.join(scenario_out, "maps_key_months")
        monthly_dir = os.path.join(scenario_out, "maps")

        n_key = plot_key_month_maps(
            df_long, atlas, point_atlas, scenario_name, cfg["label"], key_dir, map_style, marker_size
        ) if make_key else 0
        n_monthly = plot_individual_month_maps(
            df_long, atlas, point_atlas, scenario_name, cfg["label"], monthly_dir, map_style, marker_size
        ) if make_monthly else 0
        n_12panel = plot_12_panel_maps(
            df_long, atlas, point_atlas, scenario_name, cfg["label"], monthly_dir, map_style, marker_size
        ) if make_12panel else 0

        print(f"Key-month comparison maps gemaakt: {n_key} -> {key_dir}")
        print(f"Individual monthly maps gemaakt: {n_monthly} -> {monthly_dir}")
        print(f"12-panel overview maps gemaakt: {n_12panel} -> {monthly_dir}")

        manifest_rows.append({
            "scenario": scenario_name,
            "wide_path_used": wide_path,
            "available_constituents": len(available),
            "selected_constituents": len(selected),
            "long_rows": len(df_long),
            "map_style": map_style,
            "months_loaded": ",".join(f"{m:02d}" for m in months_needed),
            "key_month_maps": n_key,
            "monthly_maps": n_monthly,
            "twelve_panel_maps": n_12panel,
            "key_month_dir": key_dir,
            "monthly_dir": monthly_dir,
        })

    if manifest_rows:
        manifest = pd.DataFrame(manifest_rows)
        out_manifest = os.path.join(PREDICTION_ROOT, "map_visualization_manifest.csv")
        Path(PREDICTION_ROOT).mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_manifest, index=False)
        print(f"\nManifest opgeslagen: {out_manifest}")


if __name__ == "__main__":
    main()
