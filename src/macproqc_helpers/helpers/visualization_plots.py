"""
Per-figure plotting functions for `visualization.py`, plus the small shared
helpers they use (empty-placeholder figures, saving figures/tables). Each
figure lazily pulls whatever data it needs via `visualization_io`/
`visualization_tables`. See the `visualization` module docstring for the
overall architecture.
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from macproqc_helpers.helpers.visualization_io import (
    list_group_columns,
    read_array_metric,
    read_dataframe_metric,
    read_dataframe_metric_single_file,
    read_group_columns,
)
from macproqc_helpers.helpers.visualization_tables import assemble_result_table

pio.renderers.default = "png"

logger = logging.getLogger(__name__)


def make_empty_figure(
    message: str, width: int = 1500, height: int = 1000, title: str = "Empty Plot"
) -> go.Figure:
    """Build a placeholder plotly figure with a text annotation, used
    whenever a metric is missing from all provided files so that a plot file
    is still produced (with a clear message) instead of the script erroring
    out."""
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(width=width, height=height, title=title)
    return fig


def save_figure(fig: go.Figure, output_path: str, basename: str, args: argparse.Namespace) -> None:
    """Show/save a plotly figure according to the -fig_show/-figure_format CLI args."""
    if args.fig_show:
        fig.show()
    if args.fig_plotly:
        with open(os.path.join(output_path, basename + ".plotly.json"), "w") as json_file:
            json_file.write(plotly.io.to_json(fig))
    if args.fig_html:
        fig.write_html(file=os.path.join(output_path, basename + ".html"), auto_open=False)


def _apply_barplot_layout(fig: go.Figure, args: argparse.Namespace) -> None:
    fig.update_layout(height=int(args.height_barplots))
    if args.width_barplots > 0:
        fig.update_layout(width=int(args.width_barplots))


def _write_table(df: pd.DataFrame, output_path: str, basename: str, table_type: str) -> None:
    if table_type == "csv":
        df.to_csv(os.path.join(output_path, basename + ".csv"), index=False)
    elif table_type == "tsv":
        df.to_csv(os.path.join(output_path, basename + ".tsv"), index=False, sep="\t")
    elif table_type == "xlsx":
        df.to_excel(os.path.join(output_path, basename + ".xlsx"), index=False)


def plot_single_value_barplot(
    single_values: pd.DataFrame,
    columns: List[str],
    title: str,
    basename: str,
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """Grouped barplot of one or more single-value metrics, one group of bars per file (fig01/02/03)."""
    present = [c for c in columns if c in single_values.columns]
    for c in columns:
        if c not in present:
            logger.warning("Metric '%s' missing in all files; omitted from '%s'.", c, basename)

    if not present:
        fig = make_empty_figure(f"None of the required metrics ({', '.join(columns)}) are available!", title=title)
    else:
        df_long = single_values[["filename"] + present].melt(id_vars=["filename"])
        fig = px.bar(df_long, x="filename", y="value", color="variable", barmode="group", title=title)
        fig.update_yaxes(exponentformat="none")
        fig.update_xaxes(tickangle=-90)
        _apply_barplot_layout(fig, args)

    save_figure(fig, output_path, basename, args)


def plot_tic_overlay(
    hdf5_paths: List[str], registry: Dict[str, dict], args: argparse.Namespace, output_path: str
) -> None:
    """fig04: TIC overlay line plot."""
    title = "TIC overlay"
    entry = registry.get("MS1_TIC")
    dataframes = read_dataframe_metric(hdf5_paths, entry["example_full_key"]) if entry else {}

    if not dataframes:
        fig = make_empty_figure("Metric 'MS1_TIC' not available!", title=title)
        save_figure(fig, output_path, "fig04_MS1_TIC_overlay", args)
        return

    frames = []
    for fname, df in dataframes.items():
        df = df.copy()
        df["filename"] = fname
        frames.append(df)
    tic_df = pd.concat(frames, ignore_index=True)

    if args.RT_unit == "min":
        tic_df["second"] = tic_df["second"] / 60

    fig = px.line(tic_df, x="second", y="total ion current", color="filename", title=title)
    fig.update_traces(line=dict(width=0.5))
    fig.update_yaxes(exponentformat="E")
    _apply_barplot_layout(fig, args)
    fig.update_layout(xaxis_title="Retention Time (sec)" if args.RT_unit == "sec" else "Retention Time (min)")

    save_figure(fig, output_path, "fig04_MS1_TIC_overlay", args)


def plot_quantile_barplot(
    hdf5_paths: List[str],
    hdf5_file_names: List[str],
    registry: Dict[str, dict],
    metric: str,
    variable_prefix: str,
    title: str,
    basename: str,
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """fig05/06/07: Barplot of a quantile array metric, per file."""
    entry = registry.get(metric)
    array_values = read_array_metric(hdf5_paths, entry["example_full_key"]) if entry else {}

    if not array_values:
        fig = make_empty_figure(f"Metric '{metric}' not available!", title=title)
        save_figure(fig, output_path, basename, args)
        return

    rows = []
    for fname in hdf5_file_names:
        if fname not in array_values:
            continue
        for i, value in enumerate(array_values[fname], start=1):
            rows.append({"filename": fname, "variable": f"{variable_prefix}{i}", "value": value})
    df_long = pd.DataFrame(rows)

    fig = px.bar(df_long, x="filename", y="value", color="variable", title=title)
    fig.update_xaxes(tickangle=-90)
    _apply_barplot_layout(fig, args)

    save_figure(fig, output_path, basename, args)


def plot_category_fraction_barplot(
    hdf5_paths: List[str],
    hdf5_file_names: List[str],
    registry: Dict[str, dict],
    metric: str,
    category_col: str,
    title: str,
    basename: str,
    args: argparse.Namespace,
    output_path: str,
    replace_zero_with: Optional[str] = None,
) -> None:
    """fig08/09/10: Barplot of category fractions (charge states, missed cleavages), per file."""
    entry = registry.get(metric)
    dataframes = read_dataframe_metric(hdf5_paths, entry["example_full_key"]) if entry else {}

    frames = []
    for fname in hdf5_file_names:
        if fname not in dataframes:
            continue
        df = dataframes[fname]
        if category_col not in df.columns:
            logger.warning("Column '%s' missing for metric '%s' in file '%s'.", category_col, metric, fname)
            continue
        df = df.copy()
        df[category_col] = df[category_col].replace(df[category_col].max(), "more")
        if replace_zero_with is not None:
            df[category_col] = df[category_col].replace(0, replace_zero_with)
        df["filename"] = fname
        frames.append(df)

    if not frames:
        fig = make_empty_figure(f"Metric '{metric}' not available!", title=title)
        save_figure(fig, output_path, basename, args)
        return

    df_long = pd.concat(frames, ignore_index=True)
    fig = px.bar(df_long, x="filename", y="fraction", color=category_col, title=title)
    fig.update_xaxes(tickangle=-90)
    _apply_barplot_layout(fig, args)

    save_figure(fig, output_path, basename, args)


def run_pca_and_plot(
    hdf5_paths: List[str],
    registry: Dict[str, dict],
    single_values: pd.DataFrame,
    t_scaled: List[float],
    metric_list: List[str],
    RT_unit: str,
    use_group: bool,
    group: Optional[np.ndarray],
    title: str,
    loadings_title: str,
    args: argparse.Namespace,
    output_path: str,
    scatter_basename: str,
    loadings_basename: str,
    loadings_table_basename: str,
) -> None:
    """fig11/12: PCA scatter plot + loadings scatter plot + loadings table."""
    hdf5_file_names = single_values["filename"].tolist()

    if len(hdf5_file_names) <= 1:
        fig = make_empty_figure(
            "PCA cannot be computed using only one sample!", width=int(args.width_pca), height=int(args.height_pca)
        )
        save_figure(fig, output_path, scatter_basename, args)
        save_figure(fig, output_path, loadings_basename, args)
        loadings = pd.DataFrame(columns=["variable", "length", "PC1", "PC2"])
        _write_table(loadings, output_path, loadings_table_basename, args.output_table_type)
        return

    df_features = assemble_result_table(metric_list, hdf5_paths, registry, single_values, RT_unit=RT_unit)
    df_features = df_features.fillna(value=0)
    df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_features))

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df_scaled)
    col = ["pca" + str(y) for y in range(1, principal_components.shape[1] + 1)]
    principal_df = pd.DataFrame(data=principal_components, columns=col)
    principal_df["raw_file"] = hdf5_file_names

    pca_var = pca.explained_variance_ratio_
    label_x = f"PC1 ({round(pca_var[0] * 100, 1)}%)"
    label_y = f"PC2 ({round(pca_var[1] * 100, 1)}%)"

    if use_group:
        principal_df["group"] = group
        color_col, color_label = "group", "Group"
    else:
        principal_df["t_scaled"] = t_scaled
        color_col, color_label = "t_scaled", "timestamp"

    fig = px.scatter(
        principal_df,
        x="pca1",
        y="pca2",
        color=color_col,
        color_continuous_scale="bluered",
        title=title,
        hover_name="raw_file",
        hover_data=["pca1", "pca2"],
        labels={"pca1": label_x, "pca2": label_y, color_col: color_label},
    )
    fig.update_layout(width=int(args.width_pca), height=int(args.height_pca))
    fig.update_traces(marker=dict(size=20))

    loadings = pd.DataFrame(pca.components_.T, columns=["PC1", "PC2"], index=df_features.columns)
    loadings.insert(0, "length", np.sqrt(loadings["PC1"] ** 2 + loadings["PC2"] ** 2))
    loadings.insert(0, "variable", loadings.index)
    loadings.sort_values("length", ascending=False, inplace=True)
    fig_loadings = px.scatter(
        loadings, x="PC1", y="PC2", title=loadings_title, hover_name="variable", hover_data=["PC1", "PC2"]
    )
    fig_loadings.update_layout(width=int(args.width_pca), height=int(args.height_pca))

    save_figure(fig, output_path, scatter_basename, args)
    save_figure(fig_loadings, output_path, loadings_basename, args)
    _write_table(loadings.reset_index(drop=True), output_path, loadings_table_basename, args.output_table_type)


def plot_ms1_maps(
    hdf5_paths: List[str], registry: Dict[str, dict], args: argparse.Namespace, output_path: str
) -> None:
    """fig13: One ion map (RT vs. m/z, colored by log intensity) per file."""
    subfolder = os.path.join(output_path, "fig13_MS1_map")
    os.makedirs(subfolder, exist_ok=True)

    entry = registry.get("MS1_map")
    if entry is None:
        logger.warning("Metric 'MS1_map' not available; no ion maps produced.")
        return

    for path in hdf5_paths:
        fname = Path(path).stem
        df_map = read_dataframe_metric_single_file(path, entry["example_full_key"])
        if df_map is None:
            continue

        if len(df_map) > 1_000_000:
            samples = int(len(df_map) / 1_000_000)
            df_map = df_map.loc[range(0, len(df_map), samples), :]
        df_map = df_map.copy()
        df_map["log_intensity"] = np.log10(df_map["intensity"])

        if args.RT_unit == "min":
            df_map["retention_time"] = df_map["retention_time"] / 60

        fig, ax = plt.subplots(figsize=(15, 6))
        points = ax.scatter(df_map["retention_time"], df_map["mz"], c=df_map["log_intensity"], s=1, cmap="Blues")
        fig.colorbar(points, label="log10_intensity")
        fig.set_figheight(int(args.height_ionmaps))
        fig.set_figwidth(int(args.width_ionmaps))

        ax.set_xlabel("retention time (sec)" if args.RT_unit == "sec" else "retention time (min)")
        ax.set_ylabel("m/z")
        ax.set_title(fname)
        if args.fig_show:
            fig.show()
        fig.savefig(os.path.join(subfolder, "fig13_MS1_map_" + fname + ".png"))
        plt.close(fig)


def plot_pump_pressure(
    hdf5_paths: List[str],
    hdf5_file_names: List[str],
    registry: Dict[str, dict],
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """fig14: Pump pressure over time, overlaid across files."""
    title = "Pump Pressure"
    entry = registry.get("pump_pressure")
    dataframes = read_dataframe_metric(hdf5_paths, entry["example_full_key"]) if entry else {}

    frames = []
    for fname in hdf5_file_names:
        if fname not in dataframes:
            continue
        df = dataframes[fname]
        if df.shape[0] > 10000:
            samples = int(df.shape[0] / 10000)
            df = df.iloc[range(0, df.shape[0], samples)]
        frames.append(df.assign(filename=fname))

    if not frames:
        fig = make_empty_figure("No Pump Pressure data available!", title=title)
        save_figure(fig, output_path, "fig14_Pump_pressure", args)
        return

    df_long = pd.concat(frames, ignore_index=True)
    # x-axis data for pump pressure are in minutes, convert to seconds if necessary
    if args.RT_unit == "sec":
        df_long["retention time"] = df_long["retention time"] * 60

    fig = px.line(df_long, x="retention time", y="pressure unit", color="filename", title=title)
    fig.update_traces(line=dict(width=0.5))
    fig.update_yaxes(exponentformat="E")
    _apply_barplot_layout(fig, args)
    fig.update_layout(yaxis_title="Pump pressure")
    fig.update_layout(xaxis_title="Time (sec)" if args.RT_unit == "sec" else "Time (min)")

    save_figure(fig, output_path, "fig14_Pump_pressure", args)


def plot_psm_ppm_error_boxplot(
    hdf5_paths: List[str],
    hdf5_file_names: List[str],
    registry: Dict[str, dict],
    single_values: pd.DataFrame,
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """fig15: Boxplot of PSM ppm error quartiles, per file."""
    title = "Boxplot of PSM ppm error quartiles"
    entry = registry.get("filtered_psms_ppm_error_quartiles")
    array_values = read_array_metric(hdf5_paths, entry["example_full_key"]) if entry else {}

    if not array_values:
        fig = make_empty_figure("Metric 'filtered_psms_ppm_error_quartiles' not available!", title=title)
        save_figure(fig, output_path, "fig15_PSM_error_boxplots", args)
        return

    rows = []
    for fname in hdf5_file_names:
        if fname not in array_values:
            continue
        q = array_values[fname]
        rows.append({"filename": fname, "Q1": q[0], "Q2": q[1], "Q3": q[2]})
    df_pl15 = pd.DataFrame(rows)

    mean_vals = single_values["filtered_psms_ppm_error_mean"].values if "filtered_psms_ppm_error_mean" in single_values.columns else None
    sd_vals = single_values["filtered_psms_ppm_error_sigma"].values if "filtered_psms_ppm_error_sigma" in single_values.columns else None

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            q1=df_pl15["Q1"],
            median=df_pl15["Q2"],
            q3=df_pl15["Q3"],
            mean=mean_vals,
            sd=sd_vals,
            name="PSM ppm error",
            x=df_pl15["filename"],
        )
    )
    fig.update_layout(title=title, yaxis_title="PSM ppm error", xaxis_title="sample")
    _apply_barplot_layout(fig, args)

    save_figure(fig, output_path, "fig15_PSM_error_boxplots", args)


def plot_additional_headers(
    hdf5_paths: List[str],
    hdf5_file_names: List[str],
    registry: Dict[str, dict],
    args: argparse.Namespace,
    output_path: str,
) -> None:
    """fig16: One line plot per vendor-specific extra header column (Thermo/Bruker), overlaid across files."""
    subfolder = os.path.join(output_path, "fig16_additional_headers")
    os.makedirs(subfolder, exist_ok=True)

    entry = registry.get("Extracted_Headers")
    if entry is None:
        logger.warning("Metric 'Extracted_Headers' not available; no additional-header plots produced.")
        return
    full_key = entry["example_full_key"]

    columns_by_file: Dict[str, List[str]] = {}
    all_headers: set = set()
    for path in hdf5_paths:
        fname = Path(path).stem
        cols = list_group_columns(path, full_key)
        if cols:
            columns_by_file[fname] = cols
            all_headers.update(cols)

    if not all_headers:
        logger.warning("No 'Extracted_Headers' columns found in any file; no additional-header plots produced.")
        return

    time_header = None
    if "Time" in all_headers:
        time_header = "Time"  # Bruker
    if "Scan_StartTime" in all_headers:
        time_header = "Scan_StartTime"  # Thermo

    if time_header is None:
        logger.warning("No time header found in the extracted headers, cannot plot additional headers!")
        return

    skip_headers = {time_header, "MsMsType", "Scan_msLevel"}
    plot_headers = sorted(h for h in all_headers if h not in skip_headers)

    for header in plot_headers:
        # Ion injection time and lock mass correction should be filtered to only contain values for MS1 spectra
        ms1_filtered = header in ("EXTRA_Ion Injection Time (ms)", "EXTRA_LM mz-Correction (ppm),LM Correction")
        display_header = header + " (MS1 filtered)" if ms1_filtered else header

        rows_x, rows_y, rows_fn = [], [], []
        for path in hdf5_paths:
            fname = Path(path).stem
            if header not in columns_by_file.get(fname, []):
                continue

            wanted = [header, time_header] + (["Scan_msLevel"] if ms1_filtered else [])
            values = read_group_columns(path, full_key, wanted)
            y_tmp, x_tmp = values[header], values[time_header]
            if y_tmp is None or x_tmp is None:
                continue

            if ms1_filtered:
                ms_level = values.get("Scan_msLevel")
                if ms_level is not None:
                    y_tmp = y_tmp[ms_level == 1]
                    x_tmp = x_tmp[ms_level == 1]

            rows_x.extend(float(v) for v in x_tmp)
            rows_y.extend(float(v) for v in y_tmp)
            rows_fn.extend([fname] * len(x_tmp))

        df_tmp = pd.DataFrame({"filename": rows_fn, "x": rows_x, "y": rows_y})
        if args.RT_unit == "sec":
            df_tmp["x"] = df_tmp["x"] * 60

        basename = os.path.join("fig16_additional_headers", re.sub(r"\W+", "", display_header))

        if not df_tmp.empty:
            fig = px.line(df_tmp, x="x", y="y", color="filename", title=display_header)
            fig.update_traces(line=dict(width=0.5))
            fig.update_yaxes(exponentformat="E")
            _apply_barplot_layout(fig, args)
            fig.update_layout(yaxis_title=display_header)
            fig.update_layout(xaxis_title="Time (sec)" if args.RT_unit == "sec" else "Time (min)")
        else:
            fig = make_empty_figure(f"No '{display_header}' available!", title="Empty Plot")

        save_figure(fig, output_path, basename, args)


def plot_bruker_calibrants(
    hdf5_paths: List[str], registry: Dict[str, dict], args: argparse.Namespace, output_path: str
) -> None:
    """fig17: Bruker-only calibrant m/z and ion-mobility traces over time, one pair of plots per calibrant."""
    subfolder = os.path.join(output_path, "fig17_BRUKER_calibrants")
    os.makedirs(subfolder, exist_ok=True)

    entry = registry.get("Calibrants")
    if entry is None:
        # Calibrants are a Bruker-only metric; silently absent for Thermo data (no plots produced).
        return
    dataframes = read_dataframe_metric(hdf5_paths, entry["example_full_key"])
    if not dataframes:
        return

    frames = []
    for fname, df in dataframes.items():
        df = df.copy()
        df["filename"] = fname
        frames.append(df)
    df_calibrants = pd.concat(frames, ignore_index=True)
    calibrants = df_calibrants[["calibrant_mz", "calibrant_mobility"]].drop_duplicates()

    for i, (_, row) in enumerate(calibrants.iterrows(), start=1):
        df_tmp = df_calibrants[
            (df_calibrants["calibrant_mz"] == row["calibrant_mz"])
            & (df_calibrants["calibrant_mobility"] == row["calibrant_mobility"])
        ].copy()

        mz_tmp = row["calibrant_mz"]
        mobility_tmp = row["calibrant_mobility"]

        # rt is given in milliseconds here, convert to seconds or minutes
        if args.RT_unit == "sec":
            df_tmp["observed_calibrant_rt"] = df_tmp["observed_calibrant_rt"] / 1000
        elif args.RT_unit == "min":
            df_tmp["observed_calibrant_rt"] = df_tmp["observed_calibrant_rt"] / 60000

        title_tmp = f"Calibrant {i} m/z: {mz_tmp}, ion mobility: {mobility_tmp}"

        fig_mz = px.line(df_tmp, x="observed_calibrant_rt", y="observed_calibrant_mz", color="filename", title=title_tmp)
        fig_mz.update_traces(line=dict(width=0.5))
        fig_mz.add_hline(y=mz_tmp)
        _apply_barplot_layout(fig_mz, args)
        fig_mz.update_layout(xaxis_title="Time (sec)" if args.RT_unit == "sec" else "Time (min)")
        save_figure(fig_mz, output_path, os.path.join("fig17_BRUKER_calibrants", f"fig17a_Calibrant_mz_{i}"), args)

        fig_mobility = px.line(
            df_tmp, x="observed_calibrant_rt", y="observed_calibrant_mobility", color="filename", title=title_tmp
        )
        fig_mobility.update_traces(line=dict(width=0.5))
        fig_mobility.add_hline(y=mobility_tmp)
        _apply_barplot_layout(fig_mobility, args)
        fig_mobility.update_layout(xaxis_title="Time (sec)" if args.RT_unit == "sec" else "Time (min)")
        save_figure(
            fig_mobility, output_path, os.path.join("fig17_BRUKER_calibrants", f"fig17b_Calibrant_ionmobility{i}"), args
        )
