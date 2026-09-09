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
  "array" if it has length > 1 in ANY of the provided files (this avoids
  misclassifying a metric as "single" just because one specific file
  happens to only have one entry).
- The registry is persisted to a small JSON file (`-metric_registry_file`)
  which doubles as an editable config: besides the auto-detected `type`, it
  stores `pca_raw` / `pca_all` / `in_summary_table` flags per metric that
  control which metrics feed the PCA plots and the summary table. These
  flags are preserved across runs (edit the file to turn a metric on/off
  without touching this script); only genuinely new metrics get seeded
  with defaults.
- Data is read lazily: HDF5 files are opened only for the specific metric
  currently needed and closed again immediately afterwards, so peak memory
  usage does not scale with (number of input files) x (number of metrics).
- Each figure has its own dedicated plotting function (closely related
  figures share one generic function). `visualize()` is the single
  orchestrator that reads/derives the small pieces of data it needs (single
  values + the registry) and calls out to the per-figure functions, which
  lazily pull whatever additional data they require.
- Missing metrics never abort the run: if a metric is missing from a
  specific file, that file is skipped (with a warning) for that particular
  plot; if a metric is missing from *all* files, an empty placeholder plot
  is produced instead (with a warning), and the script continues.
"""

import argparse
import logging
import os
import sys
from typing import List

import h5py
import numpy as np

from macproqc_helpers.helpers.visualization_io import read_single_values
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
    get_or_update_metric_registry,
    metrics_with_flag,
)
from macproqc_helpers.helpers.visualization_tables import _build_hdf5_feature_table, assemble_result_table

logger = logging.getLogger(__name__)


def check_if_file_exists(s: str):
    """ checks if a file exists. If not: raise Exception """
    if os.path.isfile(s):
        return s
    else:
        raise Exception("File '{}' does not exists".format(s))


def argparse_setup(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "visualize",
        description="Visualize the QC results in the HDF5 files. This will create a table with all metrics and also create some plots. The plots will be saved as json files in the output folder.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-hdf5_files", type=check_if_file_exists, nargs="+", help = "hdf5 files which are used for visualization as string separated by whitespace", default = None)
    parser.add_argument("-output", help="Output folder for the plots as json files.", default = "graphics")
    parser.add_argument("-output_table_type", help="Type of output table (one of csv, tsv or xlsx)", default = "csv")
    parser.add_argument("-figure_format", help="Type of output figures (one or more of html and plotly)", default = "plotly") # "html,ploty" for both
    parser.add_argument("-spikeins", help = "Whether to analyse spike-ins", default = False, action = "store_true")
    parser.add_argument("-group", help="List of the experimental group (comma-separated).", default=None)  ### TODO: input table with group information
    parser.add_argument("-RT_unit", help="Unit of the retention time, either sec for seconds or min for minutes.", default = "sec")
    parser.add_argument("-fig_show", help = "Show figures, e.g. for debugging?", default = False, action = "store_true")
    parser.add_argument("-output_column_order", help = "Order of columns in the output table", default = None, type = str)
    parser.add_argument("-spikein_columns", help = "Columns of the spike-in dataframes that should end up in the result table", default = "MS1 feature maximum intensity,retention time,count of identified spectra,Delta_to_expected_RT", type = str)
    parser.add_argument("-height_barplots", help = "Height of the barplots in pixels", default = 700, type = int) # in pixels
    parser.add_argument("-width_barplots", help = "Width of the barplots in pixels", default = 0, type = int) # default 0: flexible width, in pixels
    parser.add_argument("-height_pca", help = "Height of the PCA plots in pixels", default = 1000, type = int) # in pixels
    parser.add_argument("-width_pca", help = "Width of the PCA plots in pixels", default = 1000, type = int) # in pixels
    parser.add_argument("-height_ionmaps", help = "Height of the ionmaps in inches", default = 10, type = int)
    parser.add_argument("-width_ionmaps", help = "Width of the ionmaps in inches", default = 10, type = int)
    parser.add_argument("-spike_ins_table", help = "Path to the spike-ins table file", default = None, type = str)
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
    
    parser.set_defaults(func=visualize)


def _run_figure_step(step_name: str, func, *args, **kwargs) -> None:
    """Run a figure-building function; on ANY unexpected failure, log a
    warning and move on instead of aborting the whole run."""
    try:
        func(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, see module docstring
        logger.warning("Figure step '%s' failed unexpectedly (%s); skipping.", step_name, exc)


##########################################################################################################
# Main orchestrator
##########################################################################################################


def visualize(args: argparse.Namespace) -> None:
    logging.basicConfig(filename="to_log_with_nf_later.log", level=logging.DEBUG)

    args.hdf5_files = sorted(args.hdf5_files)  # sorts the file names alphabetically (assumes same folder)
    output_path = args.output
    os.makedirs(output_path, exist_ok=True)

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
        timestamps = single_values["startTime"].values.flatten().tolist()
        mintime, maxtime = min(timestamps), max(timestamps)
        t_scaled = [
            1 if mintime == maxtime else (t - mintime) / (maxtime - mintime) * 100 for t in timestamps
        ]
    else:
        t_scaled = [0] * len(hdf5_file_names)

    ##########################################################################################
    ### 00_table_summary
    if args.output_column_order is None:
        metric_list = metrics_with_flag(registry, "in_summary_table", DEFAULT_SUMMARY_TABLE_METRICS)
    else:
        metric_list = args.output_column_order.split(",")

    df_table0 = assemble_result_table(
        metric_list, hdf5_files, registry, single_values, RT_unit=args.RT_unit, spike_ins_table=args.spike_ins_table
    )
    _write_table(df_table0, output_path, "00_table_summary", args.output_table_type)

    ##########################################################################################
    ### 99_hdf5_feature_table
    hdf5_feature_table = _build_hdf5_feature_table(hdf5_files)
    _write_table(hdf5_feature_table, output_path, "99_hdf5_feature_table", args.output_table_type)

    ##########################################################################################
    ### Figures
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
    _run_figure_step("fig04", plot_tic_overlay, hdf5_files, registry, args, output_path)
    _run_figure_step(
        "fig05",
        plot_quantile_barplot,
        hdf5_files,
        hdf5_file_names,
        registry,
        "RT_TIC_quantiles",
        "RT_TIC_Q_",
        "Quartiles of TIC over retention time",
        "fig05_barplot_TIC_quartiles",
        args,
        output_path,
    )
    _run_figure_step(
        "fig06",
        plot_quantile_barplot,
        hdf5_files,
        hdf5_file_names,
        registry,
        "RT_MS1_quantiles",
        "RT_MS1_Q_",
        "Quartiles of MS1 over retention time",
        "fig06_barplot_MS1_TIC_quartiles",
        args,
        output_path,
    )
    _run_figure_step(
        "fig07",
        plot_quantile_barplot,
        hdf5_files,
        hdf5_file_names,
        registry,
        "RT_MS2_quantiles",
        "RT_MS2_Q_",
        "Quartiles of MS2 over retention time",
        "fig07_barplot_MS2_TIC_quartiles",
        args,
        output_path,
    )
    _run_figure_step(
        "fig08",
        plot_category_fraction_barplot,
        hdf5_files,
        hdf5_file_names,
        registry,
        "MS2_prec_charge_fraction",
        "charge state",
        "Charge states of precursors",
        "fig08_barplot_precursor_charge",
        args,
        output_path,
        replace_zero_with="Unknown",
    )
    _run_figure_step(
        "fig09",
        plot_category_fraction_barplot,
        hdf5_files,
        hdf5_file_names,
        registry,
        "PSM_charge_fractions",
        "charge state",
        "Charge states of PSMs",
        "fig09_barplot_PSM_charge",
        args,
        output_path,
    )
    _run_figure_step(
        "fig10",
        plot_category_fraction_barplot,
        hdf5_files,
        hdf5_file_names,
        registry,
        "PSM_missed_cleavages_fractions",
        "number of missed cleavages",
        "Fraction of missed cleavages for PSMs",
        "fig10_barplot_PSM_missedcleavages",
        args,
        output_path,
    )

    pca_raw_metrics = metrics_with_flag(registry, "pca_raw", DEFAULT_PCA_RAW_METRICS)
    _run_figure_step(
        "fig11",
        run_pca_and_plot,
        hdf5_files,
        registry,
        single_values,
        t_scaled,
        pca_raw_metrics,
        args.RT_unit,
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

    pca_all_metrics = metrics_with_flag(
        registry, "pca_all", DEFAULT_PCA_RAW_METRICS + DEFAULT_PCA_ALL_EXTRA_METRICS
    )
    _run_figure_step(
        "fig12",
        run_pca_and_plot,
        hdf5_files,
        registry,
        single_values,
        t_scaled,
        pca_all_metrics,
        args.RT_unit,
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

    _run_figure_step("fig13", plot_ms1_maps, hdf5_files, registry, args, output_path)
    _run_figure_step("fig14", plot_pump_pressure, hdf5_files, hdf5_file_names, registry, args, output_path)
    _run_figure_step(
        "fig15", plot_psm_ppm_error_boxplot, hdf5_files, hdf5_file_names, registry, single_values, args, output_path
    )
    _run_figure_step("fig16", plot_additional_headers, hdf5_files, hdf5_file_names, registry, args, output_path)
    _run_figure_step("fig17", plot_bruker_calibrants, hdf5_files, registry, args, output_path)


