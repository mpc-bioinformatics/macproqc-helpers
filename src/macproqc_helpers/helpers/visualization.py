"""
Visualization of QC metrics stored in HDF5 files.

This module is the CLI entry point (`argparse_setup`/`visualize`) only; the
implementation is split across a few sibling modules by concern:

- `visualization_registry.py`: the metric registry (type detection - single
  vs. array vs. dataframe - plus the editable, persisted `pca_raw` /
  `pca_all` / `in_summary_table` flags per metric).
- `visualization_io.py`: lazy per-metric HDF5 readers (open/read/close one
  metric at a time).
- `visualization_tables.py`: result-table assembly (00_table_summary,
  99_hdf5_feature_table, and the per-metric feature tables used as PCA
  input).
- `visualization_plots.py`: one function per figure (or per closely-related
  group of figures), plus the small shared figure/table-saving helpers.

Architecture
------------
- `build_metric_registry` scans all provided HDF5 files (metadata only, no
  data is read) to determine, for every metric key, whether it is a single
  value, an array, or a dataframe (HDF5 group). A metric is classified as
  "array" if it has length != 1 (including 0) in ANY of the provided files
  (this avoids misclassifying a metric as "single" just because one
  specific file happens to only have one entry, or because it happens to be
  empty in every file).
- The registry is persisted to a small JSON file (`-metric_registry_file`)
  which doubles as an editable config: besides the auto-detected `type`, it
  stores `pca_raw` / `pca_all` / `in_summary_table` flags per metric that
  control which metrics feed the PCA plots and the summary table. These
  flags are preserved across runs (edit the file to turn a metric on/off
  without touching this script); only genuinely new metrics get seeded
  with defaults.
- Data is read lazily and each "group" (array/dataframe) metric is read
  from disk exactly ONCE per run, no matter how many consumers need it: for
  each such metric, `_process_group_metrics` reads it, builds every
  consumer's output from that one read - a `00_table_summary` column, a
  PCA-raw column, a PCA-all column, and/or a standalone figure - and then
  lets it be freed before moving to the next metric. So peak memory holds
  at most one group metric's worth of data at a time, and metrics shared
  across the summary table/PCA tables/a figure (e.g. `MS2_prec_charge_fraction`)
  are not re-read for each consumer. `MS1_map` and the `Extracted_Headers`/
  `Extracted_Log_Headers` metrics (which can be very large) are excluded
  from this generic pass and instead stream one file at a time directly in
  their own figure functions (fig13, fig16).
- Each figure has its own dedicated plotting function (closely related
  figures share one generic function). `visualize()` is the single
  orchestrator that reads/derives the small pieces of data it needs (single
  values + the registry), drives `_process_group_metrics`, and calls out to
  the per-figure functions with whatever data they need already fetched.
- Missing metrics never abort the run: if a metric is missing from a
  specific file, that file is skipped (with a warning) for that particular
  plot/table row; if a metric is missing from *all* files, an empty
  placeholder plot (or an all-NaN table column) is produced instead (with a
  warning), and the script continues.
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Union

import h5py
import numpy as np
import pandas as pd

from macproqc_helpers.helpers.visualization_io import (
    read_array_metric,
    read_dataframe_metric,
    read_single_values,
)
from macproqc_helpers.helpers.visualization_plots import (
    _write_table,
    plot_additional_headers,
    plot_bruker_calibrants,
    plot_category_fraction_barplot,
    plot_ms1_maps,
    plot_psm_ppm_error_boxplot,
    plot_pump_pressure,
    plot_quantile_barplot,
    plot_single_value_barplot,
    plot_tic_overlay,
    run_pca_and_plot,
)
from macproqc_helpers.helpers.visualization_registry import (
    DEFAULT_PCA_ALL_EXTRA_METRICS,
    DEFAULT_PCA_RAW_METRICS,
    DEFAULT_SUMMARY_TABLE_METRICS,
    TYPE_ARRAY,
    TYPE_DATAFRAME,
    TYPE_SINGLE,
    get_or_update_metric_registry,
    metrics_with_flag,
)
from macproqc_helpers.helpers.visualization_tables import (
    _build_hdf5_feature_table,
    _build_metric_columns,
    build_filename_column,
    build_single_value_column,
    relabel_category_extremes,
)

logger = logging.getLogger(__name__)


def check_if_file_exists(s: str):
    """checks if a file exists. If not: raise Exception"""
    if os.path.isfile(s):
        return s
    else:
        raise Exception(f"File '{s}' does not exists")


def argparse_setup(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "visualize",
        description="Visualize the QC results in the HDF5 files. This will create a table with all metrics and also create some plots. The plots will be saved as json files in the output folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-hdf5_files",
        type=check_if_file_exists,
        nargs="+",
        help="hdf5 files which are used for visualization as string separated by whitespace",
        default=None,
    )
    parser.add_argument(
        "-output", help="Output folder for the plots as json files.", default="graphics"
    )
    parser.add_argument(
        "-output_table_type", help="Type of output table (one of csv, tsv or xlsx)", default="csv"
    )
    parser.add_argument(
        "-figure_format",
        help="Type of output figures (one or more of html and plotly)",
        default="plotly",
    )  # "html,ploty" for both
    parser.add_argument(
        "-spikeins", help="Whether to analyse spike-ins", default=False, action="store_true"
    )
    parser.add_argument(
        "-group", help="List of the experimental group (comma-separated).", default=None
    )  ### TODO: input table with group information
    parser.add_argument(
        "-RT_unit",
        help="Unit of the retention time, either sec for seconds or min for minutes.",
        default="sec",
    )
    parser.add_argument(
        "-fig_show", help="Show figures, e.g. for debugging?", default=False, action="store_true"
    )
    parser.add_argument(
        "-output_column_order", help="Order of columns in the output table", default=None, type=str
    )
    parser.add_argument(
        "-spikein_columns",
        help="Columns of the spike-in dataframes that should end up in the result table",
        default="MS1 feature maximum intensity,retention time,count of identified spectra,Delta_to_expected_RT",
        type=str,
    )
    parser.add_argument(
        "-height_barplots", help="Height of the barplots in pixels", default=700, type=int
    )  # in pixels
    parser.add_argument(
        "-width_barplots", help="Width of the barplots in pixels", default=0, type=int
    )  # default 0: flexible width, in pixels
    parser.add_argument(
        "-height_pca", help="Height of the PCA plots in pixels", default=1000, type=int
    )  # in pixels
    parser.add_argument(
        "-width_pca", help="Width of the PCA plots in pixels", default=1000, type=int
    )  # in pixels
    parser.add_argument(
        "-height_ionmaps", help="Height of the ionmaps in inches", default=10, type=int
    )
    parser.add_argument(
        "-width_ionmaps", help="Width of the ionmaps in inches", default=10, type=int
    )
    parser.add_argument(
        "-spike_ins_table", help="Path to the spike-ins table file", default=None, type=str
    )
    parser.add_argument(
        "-metric_registry_file",
        help=(
            "Path to a JSON file storing the metric-type registry (single/array/dataframe per "
            "metric) plus editable 'pca_raw'/'pca_all'/'in_summary_table' flags controlling which "
            "metrics are used for the PCA plots and the summary table. Created automatically with "
            "sensible defaults if it does not exist yet; edits to these flags are preserved across runs."
        ),
        default="metric_registry.json",
        type=str,
    )
    parser.add_argument(
        "-log_file",
        help=(
            "Additionally write a persisted log file ('visualization.log') inside -output, so logs "
            "are published alongside the plots/tables. By default, warnings/errors only go to stderr "
            "(so they show up in Nextflow's per-task .command.err)."
        ),
        default=False,
        action="store_true",
    )

    parser.set_defaults(func=visualize)


def _run_figure_step(step_name: str, func, *args, **kwargs) -> None:
    """Run a figure-building function; on ANY unexpected failure, log a
    warning and move on instead of aborting the whole run."""
    try:
        func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
        logger.warning("Figure step '%s' failed unexpectedly (%s); skipping.", step_name, exc)


def _assemble_table(
    metric_order: List[str], columns: Dict[str, pd.DataFrame], n_rows: int
) -> pd.DataFrame:
    """Concatenate already-built per-metric column-frames into one table, in
    `metric_order` order. A metric with no entry in `columns` (unknown to
    the registry, or its data could not be fetched/assembled) falls back to
    an all-NaN column, so the table is always fully built."""
    frames = []
    for metric in metric_order:
        frame = columns.get(metric)
        if frame is None:
            frame = pd.DataFrame({metric: [np.nan] * n_rows})
        frames.append(frame.reset_index(drop=True))
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=range(n_rows))


def _process_group_metrics(
    hdf5_files: List[str],
    hdf5_file_names: List[str],
    registry: Dict[str, dict],
    single_values: pd.DataFrame,
    metric_list: List[str],
    pca_raw_metrics: List[str],
    pca_all_metrics: List[str],
    args: argparse.Namespace,
    output_path: str,
):
    """Read every "group" (array/dataframe) metric needed by the summary
    table, the two PCA feature tables, and/or a standalone figure exactly
    ONCE from disk, build every one of those consumers' output from that
    single read, and let the metric's data be freed before moving on to the
    next metric - see the module docstring. `MS1_map` and any
    `dynamic_columns` metric (`Extracted_Headers`/`Extracted_Log_Headers`)
    are excluded; fig13/fig16 stream those directly, one file at a time,
    since they can be very large.

    Returns `(table0_columns, pca_raw_columns, pca_all_columns)` - dicts of
    metric name -> column-DataFrame, not yet concatenated/ordered; the
    caller assembles the final tables via `_assemble_table`.
    """
    # For these metrics, the PCA feature columns use `relabel_category_extremes`
    # (category_col, replace_zero_with) on top of the raw per-file data, while
    # `00_table_summary` uses the raw values unchanged.
    pca_category_relabel = {
        "MS2_prec_charge_fraction": ("charge state", "Unknown"),
        "PSM_charge_fractions": ("charge state", None),
        "PSM_missed_cleavages_fractions": ("number of missed cleavages", None),
    }

    standalone_figures = [
        (
            "MS1_TIC",
            lambda data: _run_figure_step("fig04", plot_tic_overlay, data, args, output_path),
        ),
        (
            "RT_TIC_quantiles",
            lambda data: _run_figure_step(
                "fig05",
                plot_quantile_barplot,
                data,
                hdf5_file_names,
                "RT_TIC_quantiles",
                "RT_TIC_Q_",
                "Quartiles of TIC over retention time",
                "fig05_barplot_TIC_quartiles",
                args,
                output_path,
            ),
        ),
        (
            "RT_MS1_quantiles",
            lambda data: _run_figure_step(
                "fig06",
                plot_quantile_barplot,
                data,
                hdf5_file_names,
                "RT_MS1_quantiles",
                "RT_MS1_Q_",
                "Quartiles of MS1 over retention time",
                "fig06_barplot_MS1_TIC_quartiles",
                args,
                output_path,
            ),
        ),
        (
            "RT_MS2_quantiles",
            lambda data: _run_figure_step(
                "fig07",
                plot_quantile_barplot,
                data,
                hdf5_file_names,
                "RT_MS2_quantiles",
                "RT_MS2_Q_",
                "Quartiles of MS2 over retention time",
                "fig07_barplot_MS2_TIC_quartiles",
                args,
                output_path,
            ),
        ),
        (
            "MS2_prec_charge_fraction",
            lambda data: _run_figure_step(
                "fig08",
                plot_category_fraction_barplot,
                data,
                hdf5_file_names,
                "MS2_prec_charge_fraction",
                pca_category_relabel["MS2_prec_charge_fraction"][0],
                "Charge states of precursors",
                "fig08_barplot_precursor_charge",
                args,
                output_path,
                replace_zero_with=pca_category_relabel["MS2_prec_charge_fraction"][1],
            ),
        ),
        (
            "PSM_charge_fractions",
            lambda data: _run_figure_step(
                "fig09",
                plot_category_fraction_barplot,
                data,
                hdf5_file_names,
                "PSM_charge_fractions",
                pca_category_relabel["PSM_charge_fractions"][0],
                "Charge states of PSMs",
                "fig09_barplot_PSM_charge",
                args,
                output_path,
            ),
        ),
        (
            "PSM_missed_cleavages_fractions",
            lambda data: _run_figure_step(
                "fig10",
                plot_category_fraction_barplot,
                data,
                hdf5_file_names,
                "PSM_missed_cleavages_fractions",
                pca_category_relabel["PSM_missed_cleavages_fractions"][0],
                "Fraction of missed cleavages for PSMs",
                "fig10_barplot_PSM_missedcleavages",
                args,
                output_path,
            ),
        ),
        (
            "pump_pressure",
            lambda data: _run_figure_step(
                "fig14", plot_pump_pressure, data, hdf5_file_names, args, output_path
            ),
        ),
        (
            "filtered_psms_ppm_error_quartiles",
            lambda data: _run_figure_step(
                "fig15",
                plot_psm_ppm_error_boxplot,
                data,
                hdf5_file_names,
                single_values,
                args,
                output_path,
            ),
        ),
        (
            "Calibrants",
            lambda data: _run_figure_step("fig17", plot_bruker_calibrants, data, args, output_path),
        ),
    ]
    figures_by_metric = dict(standalone_figures)

    def _is_group_metric(entry: Optional[dict]) -> bool:
        return (
            entry is not None
            and entry["type"] in (TYPE_ARRAY, TYPE_DATAFRAME)
            and not entry.get("dynamic_columns")
        )

    needed: Dict[str, dict] = {}
    for metric in list(metric_list) + pca_raw_metrics + pca_all_metrics:
        entry = registry.get(metric)
        if entry is not None and _is_group_metric(entry) and metric != "MS1_map":
            needed.setdefault(metric, entry)
    for metric in figures_by_metric:
        entry = registry.get(metric)
        if entry is not None:
            needed.setdefault(metric, entry)

    table0_columns: Dict[str, pd.DataFrame] = {}
    pca_raw_columns: Dict[str, pd.DataFrame] = {}
    pca_all_columns: Dict[str, pd.DataFrame] = {}

    for metric, entry in needed.items():
        metric_type = entry["type"]
        try:
            if metric_type == TYPE_ARRAY:
                data = read_array_metric(hdf5_files, entry["example_full_key"])
            else:
                data = read_dataframe_metric(hdf5_files, entry["example_full_key"])
        except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
            logger.warning("Could not read metric '%s' (%s); treating as missing.", metric, exc)
            data = {}

        if metric in metric_list or metric in pca_raw_metrics or metric in pca_all_metrics:
            try:
                col_frame = _build_metric_columns(
                    metric, metric_type, data, hdf5_file_names, args.RT_unit, args.spike_ins_table
                )
            except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
                logger.warning(
                    "Could not assemble metric '%s' (%s); filling with NaN.", metric, exc
                )
                col_frame = pd.DataFrame({metric: [np.nan] * len(hdf5_file_names)})
            if metric in metric_list:
                table0_columns[metric] = col_frame

            pca_col_frame = col_frame
            if metric in pca_category_relabel and (
                metric in pca_raw_metrics or metric in pca_all_metrics
            ):
                category_col, replace_zero_with = pca_category_relabel[metric]
                # `data` may hold `np.ndarray` values when `metric_type == TYPE_ARRAY`;
                # those are passed through unchanged below, only DataFrames get relabeled.
                relabeled_data: Dict[str, Union[pd.DataFrame, np.ndarray]] = {}
                for fname, df in data.items():
                    try:
                        if (
                            isinstance(df, pd.DataFrame)
                            and not df.empty
                            and category_col in df.columns
                        ):
                            relabeled_data[fname] = relabel_category_extremes(
                                df, category_col, replace_zero_with
                            )
                        else:
                            relabeled_data[fname] = df
                    except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
                        logger.warning(
                            "Could not relabel metric '%s' data for PCA use in file '%s' (%s); using raw values.",
                            metric,
                            fname,
                            exc,
                        )
                        relabeled_data[fname] = df
                try:
                    pca_col_frame = _build_metric_columns(
                        metric,
                        metric_type,
                        relabeled_data,
                        hdf5_file_names,
                        args.RT_unit,
                        args.spike_ins_table,
                    )
                except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
                    logger.warning(
                        "Could not assemble metric '%s' (%s); filling with NaN.", metric, exc
                    )
                    pca_col_frame = pd.DataFrame({metric: [np.nan] * len(hdf5_file_names)})

            if metric in pca_raw_metrics:
                pca_raw_columns[metric] = pca_col_frame
            if metric in pca_all_metrics:
                pca_all_columns[metric] = pca_col_frame

        if metric in figures_by_metric:
            figures_by_metric[metric](data)

    return table0_columns, pca_raw_columns, pca_all_columns


##########################################################################################################
# Main orchestrator
##########################################################################################################


def visualize(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.hdf5_files = sorted(
        args.hdf5_files
    )  # sorts the file names alphabetically (assumes same folder)
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

    if args.log_file:
        file_handler = logging.FileHandler(os.path.join(output_path, "visualization.log"))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(file_handler)

    args.fig_html = args.figure_format.find("html") >= 0
    args.fig_plotly = args.figure_format.find("plotly") >= 0

    ### give warning if files from both Bruker and Thermo machines are present; also detect + drop empty files
    thermo_files: List[str] = []
    bruker_files: List[str] = []
    empty_files: List[str] = []
    for path in args.hdf5_files:
        with h5py.File(path, "r") as hdf:
            keys = set(hdf.keys())
            if len(keys) == 0:
                logger.warning("HDF5 file '%s' is empty and will be excluded from plotting.", path)
                empty_files.append(path)
                continue
            if "THERMO ! Extracted_Headers" in keys:
                thermo_files.append(path)
            elif "BRUKER ! Extracted_Headers" in keys:
                bruker_files.append(path)

    if thermo_files and bruker_files:
        logger.warning(
            "You have files from both Thermo and Bruker machines. Direct comparison is not possible, no plots will be created."
        )
        sys.exit(0)

    hdf5_files = [f for f in args.hdf5_files if f not in empty_files]
    if not hdf5_files:
        logger.warning("All provided HDF5 files are empty. No plots will be created.")
        sys.exit(0)
    args.hdf5_files = hdf5_files

    registry = get_or_update_metric_registry(hdf5_files, args.metric_registry_file)
    single_values = read_single_values(hdf5_files, registry)
    hdf5_file_names = single_values["filename"].tolist()

    if args.group is None:
        use_group = False
        group = None
    else:
        use_group = True
        group = np.array(args.group.split(","))

    if "startTime" in single_values.columns:
        timestamps = np.asarray(single_values["startTime"].values, dtype=float)
        n_missing = int(np.isnan(timestamps).sum())
        if n_missing == len(timestamps):
            logger.warning(
                "Metric 'startTime' missing/invalid in all files; PCA scatter coloring disabled."
            )
            t_scaled = [0] * len(hdf5_file_names)
        else:
            if 0 < n_missing < len(timestamps):
                missing_files = [f for f, t in zip(hdf5_file_names, timestamps) if np.isnan(t)]
                logger.warning(
                    "Metric 'startTime' missing/invalid in %d of %d files (%s); those points will be colored neutrally.",
                    n_missing,
                    len(timestamps),
                    ", ".join(missing_files),
                )
            mintime, maxtime = np.nanmin(timestamps), np.nanmax(timestamps)
            if mintime == maxtime:
                t_scaled = [np.nan if np.isnan(t) else 1 for t in timestamps]
            else:
                t_scaled = [
                    np.nan if np.isnan(t) else (t - mintime) / (maxtime - mintime) * 100
                    for t in timestamps
                ]
    else:
        t_scaled = [0] * len(hdf5_file_names)

    ##########################################################################################
    ### Metric lists (summary table columns, PCA feature sets)
    if args.output_column_order is None:
        metric_list = metrics_with_flag(registry, "in_summary_table", DEFAULT_SUMMARY_TABLE_METRICS)
    else:
        metric_list = args.output_column_order.split(",")

    pca_raw_metrics = metrics_with_flag(registry, "pca_raw", DEFAULT_PCA_RAW_METRICS)
    pca_all_metrics = metrics_with_flag(
        registry, "pca_all", DEFAULT_PCA_RAW_METRICS + DEFAULT_PCA_ALL_EXTRA_METRICS
    )

    ##########################################################################################
    ### Figures that only need `single_values` - no group-metric I/O yet
    _run_figure_step(
        "fig01",
        plot_single_value_barplot,
        single_values,
        ["nr_MS1", "nr_MS2"],
        "Number of MS1 and MS2 spectra",
        "fig01_barplot_MS1_MS2",
        args,
        output_path,
    )
    _run_figure_step(
        "fig02",
        plot_single_value_barplot,
        single_values,
        ["nr_PSMs", "nr_peptides", "nr_protein_groups", "nr_accessions"],
        "Number of filtered PSMs, filtered peptides, filtered protein groups and accessions",
        "fig02_barplot_PSMs_peptides_proteins",
        args,
        output_path,
    )
    _run_figure_step(
        "fig03",
        plot_single_value_barplot,
        single_values,
        ["nr_features", "nr_ident_features"],
        "Number of features and identified features",
        "fig03_barplot_features",
        args,
        output_path,
    )

    ##########################################################################################
    ### `filename`/single-value columns for the summary table and PCA feature tables, built
    ### directly from `single_values` (already in memory, no I/O needed).
    single_columns: Dict[str, pd.DataFrame] = {}
    for metric in set(metric_list) | set(pca_raw_metrics) | set(pca_all_metrics):
        if metric == "filename":
            single_columns[metric] = build_filename_column(hdf5_file_names)
            continue
        entry = registry.get(metric)
        if entry is not None and entry["type"] == TYPE_SINGLE:
            single_columns[metric] = build_single_value_column(metric, single_values)

    ##########################################################################################
    ### Group (array/dataframe) metrics: read each one exactly once, build every consumer's
    ### output (summary-table column, PCA-raw column, PCA-all column, standalone figure) from it
    table0_columns, pca_raw_columns, pca_all_columns = _process_group_metrics(
        hdf5_files,
        hdf5_file_names,
        registry,
        single_values,
        metric_list,
        pca_raw_metrics,
        pca_all_metrics,
        args,
        output_path,
    )

    ##########################################################################################
    ### 00_table_summary
    df_table0 = _assemble_table(
        metric_list, {**single_columns, **table0_columns}, len(hdf5_file_names)
    )
    _write_table(df_table0, output_path, "00_table_summary", args.output_table_type)

    ##########################################################################################
    ### 99_hdf5_feature_table
    hdf5_feature_table = _build_hdf5_feature_table(hdf5_files)
    _write_table(hdf5_feature_table, output_path, "99_hdf5_feature_table", args.output_table_type)

    ##########################################################################################
    ### fig11/fig12: PCA (feature tables assembled from the metrics already fetched above)
    df_pl11 = _assemble_table(
        pca_raw_metrics, {**single_columns, **pca_raw_columns}, len(hdf5_file_names)
    )
    _run_figure_step(
        "fig11",
        run_pca_and_plot,
        df_pl11,
        hdf5_file_names,
        t_scaled,
        use_group,
        group,
        "PCA on raw data",
        "PCA loadings (raw data)",
        args,
        output_path,
        "fig11a_PCA_raw",
        "fig11b_Loadings_raw",
        "fig11c_table_loadings_raw",
    )

    df_pl12 = _assemble_table(
        pca_all_metrics, {**single_columns, **pca_all_columns}, len(hdf5_file_names)
    )
    _run_figure_step(
        "fig12",
        run_pca_and_plot,
        df_pl12,
        hdf5_file_names,
        t_scaled,
        use_group,
        group,
        "PCA on all data",
        "PCA loadings (all data)",
        args,
        output_path,
        "fig12a_PCA_all",
        "fig12b_Loadings_all",
        "fig12c_table_loadings_raw",
    )

    ##########################################################################################
    ### fig13/fig16: large, per-file-streamed figures - not part of the group-metric pass above
    _run_figure_step("fig13", plot_ms1_maps, hdf5_files, registry, args, output_path)
    _run_figure_step(
        "fig16", plot_additional_headers, hdf5_files, hdf5_file_names, registry, args, output_path
    )
