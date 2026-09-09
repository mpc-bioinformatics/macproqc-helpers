"""
Result-table assembly for `visualization.py`: builds the 00_table_summary
table and the per-metric feature tables used as PCA input, plus the
99_hdf5_feature_table describing every distinct metric key seen across the
input files. See the `visualization` module docstring for the overall
architecture.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from macproqc_helpers.helpers.visualization_io import read_array_metric, read_dataframe_metric
from macproqc_helpers.helpers.visualization_registry import TYPE_ARRAY, TYPE_DATAFRAME, TYPE_SINGLE

logger = logging.getLogger(__name__)


def _sort_key(value: Any):
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _build_wide_from_arrays(
    metric: str, hdf5_file_names: List[str], array_values: Dict[str, np.ndarray], RT_unit: str
) -> pd.DataFrame:
    """Turn one array-type metric (e.g. RT quantiles) into one row per file,
    one column per array position, named "{metric}_1", "{metric}_2", etc.
    (or "{metric}_min"/"{metric}_max" for the special-cased "RT_range").
    Files missing the metric get a row of NaN; the column count is set by
    the longest array seen across all files."""
    ncols = max((len(v) for v in array_values.values()), default=0)
    rows = [
        list(array_values[file]) if file in array_values else [np.nan] * ncols
        for file in hdf5_file_names
    ]
    if metric == "RT_range":
        columns = [metric + "_" + suffix for suffix in ["min", "max"]]
    else:
        columns = [metric + "_" + str(i) for i in range(1, ncols + 1)]
    df_tmp = pd.DataFrame(rows, columns=columns)
    if metric == "RT_range" and RT_unit == "min":
        df_tmp["RT_range_min"] = df_tmp["RT_range_min"] / 60
        df_tmp["RT_range_max"] = df_tmp["RT_range_max"] / 60
    return df_tmp


def _build_wide_from_dataframes(
    metric: str, hdf5_file_names: List[str], dataframes: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Pivot a 2-column (category, value) dataframe metric (e.g. charge state
    fractions) into one row per file, one column per category."""
    pivoted_by_file: Dict[str, pd.Series] = {}
    for file, df in dataframes.items():
        if df is None or df.empty:
            continue
        pivoted = df.pivot_table(columns=df.columns[0], values=df.columns[1], aggfunc="first")
        pivoted_by_file[file] = pivoted.iloc[0]

    all_columns = sorted(
        {col for series in pivoted_by_file.values() for col in series.index}, key=_sort_key
    )
    rows = []
    for file in hdf5_file_names:
        series = pivoted_by_file.get(file)
        if series is None:
            rows.append({col: np.nan for col in all_columns})
        else:
            rows.append({col: series.get(col, np.nan) for col in all_columns})

    df_tmp = pd.DataFrame(rows, columns=all_columns)
    df_tmp.columns = [metric + "_" + str(col) for col in df_tmp.columns]
    return df_tmp


def _build_spike_in_table(
    hdf5_file_names: List[str],
    dataframes: Dict[str, pd.DataFrame],
    RT_unit: str,
    spike_ins_table: Optional[str],
) -> pd.DataFrame:
    """Turn the "spike_in_metrics" dataframe metric (one row per spike-in
    peptide, per file) into one row per file with columns named
    "{spike_in_name}_{original_column}" for every original column (plus the
    derived "Delta_to_expected_RT"). The spike-in name used as the column
    prefix is either taken from `spike_ins_table` (a CSV with a "name"
    column, in the same row order as the per-file data) if provided, or
    falls back to the peptide's own "proforma peptidoform sequence". Files
    missing the metric get a row of NaN for every column seen across all
    files."""
    per_file_rows: Dict[str, pd.DataFrame] = {}

    for file, spike_data in dataframes.items():
        if spike_data is None or spike_data.empty:
            continue
        spike_data = spike_data.copy()

        if RT_unit == "min":
            spike_data["retention time"] = spike_data["retention time"] / 60
            spike_data["predicted retention time"] = spike_data["predicted retention time"] / 60

        spike_data["Delta_to_expected_RT"] = (
            spike_data["retention time"] - spike_data["predicted retention time"]
        )
        # utf8 decoding of the proforma peptidoform sequence
        spike_data["proforma peptidoform sequence"] = spike_data["proforma peptidoform sequence"].apply(
            lambda x: x.decode("utf8") if isinstance(x, bytes) else x
        )

        if spike_ins_table is not None:
            spike_ins_info = pd.read_csv(spike_ins_table, sep=",")
            spike_in_list = spike_ins_info["name"].astype(str).tolist()
        else:
            spike_in_list = spike_data["proforma peptidoform sequence"].astype(str).tolist()

        value_columns = spike_data.columns.to_list()

        row_frame = pd.DataFrame()
        for index, _ in spike_data.iterrows():
            column_names = [spike_in_list[index] + "_" + col for col in value_columns]
            single_spikein = pd.DataFrame(columns=column_names)
            single_spikein.loc[0] = spike_data.loc[index, value_columns].values
            row_frame = pd.concat([row_frame, single_spikein], axis=1)

        per_file_rows[file] = row_frame

    if not per_file_rows:
        raise ValueError("metric 'spike_in_metrics' not present in any file")

    all_columns: List[str] = []
    for row_frame in per_file_rows.values():
        for col in row_frame.columns:
            if col not in all_columns:
                all_columns.append(col)

    rows = []
    for file in hdf5_file_names:
        row_frame = per_file_rows.get(file)
        if row_frame is None:
            logger.warning("Metric 'spike_in_metrics' missing in file '%s'.", file)
            rows.append({col: np.nan for col in all_columns})
        else:
            rows.append({col: row_frame.iloc[0].get(col, np.nan) for col in all_columns})

    return pd.DataFrame(rows, columns=all_columns)


def _assemble_metric_columns(
    metric: str,
    hdf5_paths: List[str],
    registry: Dict[str, dict],
    single_values: pd.DataFrame,
    hdf5_file_names: List[str],
    RT_unit: str,
    spike_ins_table: Optional[str],
) -> pd.DataFrame:
    """Build the column(s) for one requested metric, one row per file (in
    `hdf5_file_names` order), dispatching on the metric's registry type:
    - "filename" is special-cased (just echoes `hdf5_file_names` back).
    - `TYPE_SINGLE` metrics are pulled directly from the already-read
      `single_values` DataFrame (with `startTime` converted to a proper
      timestamp column).
    - `TYPE_ARRAY` metrics are read lazily via `read_array_metric` and
      pivoted wide via `_build_wide_from_arrays`.
    - `TYPE_DATAFRAME` metrics are read lazily via `read_dataframe_metric`,
      then handled by one of three special cases (the two
      "*_maxima_per_time_range(s)" metrics, which just take the first row
      of one column; "spike_in_metrics", via `_build_spike_in_table`) or,
      for any other dataframe metric, the generic 2-column pivot in
      `_build_wide_from_dataframes`.
    Raises if the metric is missing from the registry, not present in any
    file, or has an unrecognized registry type - the caller
    (`assemble_result_table`) catches these and fills NaN instead."""
    if metric == "filename":
        return pd.DataFrame({"filename": hdf5_file_names})

    entry = registry.get(metric)
    if entry is None:
        raise KeyError(f"metric '{metric}' not found in the metric registry")

    metric_type = entry["type"]

    if metric_type == TYPE_SINGLE:
        if metric == "startTime":
            converted = [
                datetime.fromtimestamp(t, timezone.utc) for t in single_values["startTime"]
            ]
            s = pd.to_datetime(pd.Series(converted)).dt.tz_localize(None)
            return pd.DataFrame({"startTime": s.values})
        return pd.DataFrame({metric: single_values[metric].values})

    if metric_type == TYPE_ARRAY:
        array_values = read_array_metric(hdf5_paths, entry["example_full_key"])
        if not array_values:
            raise ValueError(f"metric '{metric}' not present in any file")
        return _build_wide_from_arrays(metric, hdf5_file_names, array_values, RT_unit)

    if metric_type == TYPE_DATAFRAME:
        dataframes = read_dataframe_metric(hdf5_paths, entry["example_full_key"])
        if not dataframes:
            raise ValueError(f"metric '{metric}' not present in any file")

        if metric in ("base_peak_intensity_maxima_per_time_range", "total_ion_current_maxima_per_time_ranges"):
            col = "base peak intensity" if metric == "base_peak_intensity_maxima_per_time_range" else "total ion current"
            values = [
                dataframes[file][col].iloc[0]
                if file in dataframes and col in dataframes[file].columns
                else np.nan
                for file in hdf5_file_names
            ]
            return pd.DataFrame({metric: values})

        if metric == "spike_in_metrics":
            return _build_spike_in_table(hdf5_file_names, dataframes, RT_unit, spike_ins_table)

        return _build_wide_from_dataframes(metric, hdf5_file_names, dataframes)

    raise ValueError(f"unknown metric type '{metric_type}' for metric '{metric}'")


def assemble_result_table(
    metric_list: List[str],
    hdf5_paths: List[str],
    registry: Dict[str, dict],
    single_values: pd.DataFrame,
    RT_unit: str = "sec",
    spike_ins_table: Optional[str] = None,
) -> pd.DataFrame:
    """
    Assemble a result table with one row per hdf5 file and one (or more)
    column(s) per requested metric.

    Data for each metric is fetched lazily (only the metrics in
    `metric_list` are read from disk, one at a time) via the metric
    registry. If a metric is missing from the registry entirely, or fails to
    be assembled for any reason, the resulting column(s) are filled with NaN
    and a warning is logged - this table is always fully built, even if some
    metrics are missing or malformed in the provided hdf5 files.
    """
    hdf5_file_names = single_values["filename"].tolist()
    df_table = pd.DataFrame(index=range(len(hdf5_file_names)))

    for metric in metric_list:
        try:
            df_metric = _assemble_metric_columns(
                metric, hdf5_paths, registry, single_values, hdf5_file_names, RT_unit, spike_ins_table
            )
        except Exception as exc:
            logger.warning("Could not assemble metric '%s' (%s); filling with NaN.", metric, exc)
            df_metric = pd.DataFrame({metric: [np.nan] * len(hdf5_file_names)})
        df_table = pd.concat([df_table.reset_index(drop=True), df_metric.reset_index(drop=True)], axis=1)

    return df_table


def _build_hdf5_feature_table(hdf5_paths: List[str]) -> pd.DataFrame:
    """Build a table describing every distinct metric key seen across all
    hdf5 files (its qc_description/qc_name/... attributes). Only attributes
    are read - no metric data - so this is cheap even for large files."""
    known_keys: set = set()
    rows = []
    for path in hdf5_paths:
        with h5py.File(path, "r") as hdf:
            current_keys = set(hdf.keys())
            new_keys = current_keys - known_keys
            known_keys |= current_keys
            for key in new_keys:
                attrs = hdf[key].attrs
                rows.append(
                    {
                        "key": key,
                        "qc_description": attrs.get("qc_description", ""),
                        "qc_name": attrs.get("qc_name", ""),
                        "qc_short_name": attrs.get("qc_short_name", ""),
                        "unit_accession": attrs.get("unit_accession", ""),
                        "unit_name": attrs.get("unit_name", ""),
                    }
                )
    df = pd.DataFrame(
        rows, columns=["key", "qc_description", "qc_name", "qc_short_name", "unit_accession", "unit_name"]
    )
    return df.sort_values(by="key", ascending=True).reset_index(drop=True)
