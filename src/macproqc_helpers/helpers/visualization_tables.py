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

from macproqc_helpers.helpers.visualization_registry import TYPE_ARRAY, TYPE_DATAFRAME

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


def relabel_category_extremes(
    df: pd.DataFrame, category_col: str, replace_zero_with: Optional[str] = None
) -> pd.DataFrame:
    """Return a copy of `df` with `category_col`'s maximum value relabeled
    to 'more' (and, if given, 0 relabeled to `replace_zero_with`)."""
    df = df.copy()
    df[category_col] = df[category_col].replace(df[category_col].max(), "more")
    if replace_zero_with is not None:
        df[category_col] = df[category_col].replace(0, replace_zero_with)
    return df


def _build_wide_from_dataframes(
    metric: str, hdf5_file_names: List[str], dataframes: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Pivot a 2-column (category, value) dataframe metric (e.g. charge state
    fractions) into one row per file, one column per category."""
    pivoted_by_file: Dict[str, pd.Series] = {}
    for file, df in dataframes.items():
        if df is None or df.empty:
            continue
        try:
            pivoted = df.pivot_table(columns=df.columns[0], values=df.columns[1], aggfunc="first")
            pivoted_by_file[file] = pivoted.iloc[0]
        except Exception as exc:
            logger.warning(
                "Could not pivot metric '%s' data for file '%s' (%s); treating as missing.", metric, file, exc
            )

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
    per_file_rows: Dict[str, Dict[str, Any]] = {}

    for file, spike_data in dataframes.items():
        if spike_data is None or spike_data.empty:
            continue
        try:
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

            row_values: Dict[str, Any] = {}
            for idx in range(len(spike_data)):
                row = spike_data.iloc[idx]
                for col in value_columns:
                    row_values[f"{spike_in_list[idx]}_{col}"] = row[col]

            per_file_rows[file] = row_values
        except Exception as exc:
            logger.warning(
                "Could not assemble metric 'spike_in_metrics' data for file '%s' (%s); treating as missing.",
                file, exc,
            )

    if not per_file_rows:
        raise ValueError("metric 'spike_in_metrics' not present in any file")

    all_columns: List[str] = []
    for row_values in per_file_rows.values():
        for col in row_values.keys():
            if col not in all_columns:
                all_columns.append(col)

    rows = []
    for file in hdf5_file_names:
        row_values = per_file_rows.get(file)
        if row_values is None:
            logger.warning("Metric 'spike_in_metrics' missing in file '%s'.", file)
            rows.append({col: np.nan for col in all_columns})
        else:
            rows.append({col: row_values.get(col, np.nan) for col in all_columns})

    return pd.DataFrame(rows, columns=all_columns)


def build_filename_column(hdf5_file_names: List[str]) -> pd.DataFrame:
    """Build the 'filename' column - no I/O, just echoes `hdf5_file_names` back."""
    return pd.DataFrame({"filename": hdf5_file_names})


def build_single_value_column(metric: str, single_values: pd.DataFrame) -> pd.DataFrame:
    """Build the column for one `TYPE_SINGLE` metric directly from the
    already-in-memory `single_values` DataFrame - no I/O needed, since single
    values are all read once up front (see `visualization_io.read_single_values`).

    `startTime` gets special handling: each file's value is converted to a
    proper timestamp independently, and a missing/invalid value blanks only
    that file's row."""
    if metric == "startTime":
        converted = []
        for t in single_values["startTime"]:
            try:
                converted.append(datetime.fromtimestamp(t, timezone.utc))
            except Exception:
                converted.append(pd.NaT)
        s = pd.to_datetime(pd.Series(converted)).dt.tz_localize(None)
        return pd.DataFrame({"startTime": s.values})
    return pd.DataFrame({metric: single_values[metric].values})


def _build_metric_columns(
    metric: str,
    metric_type: str,
    data: Any,
    hdf5_file_names: List[str],
    RT_unit: str,
    spike_ins_table: Optional[str],
) -> pd.DataFrame:
    """Build the column(s) for one already-fetched group ('array' or
    'dataframe' type) metric, one row per file (in `hdf5_file_names` order).

    `data` is the corresponding array_values/dataframes dict, already read
    from disk by the caller (see `visualization.py`'s `_process_group_metrics`,
    which reads each metric from disk once and reuses that one `data` for
    every consumer - the summary table, the PCA feature tables, and/or a
    standalone figure - that needs it).

    `filename` and `TYPE_SINGLE` metrics are NOT handled here - see
    `build_filename_column`/`build_single_value_column`, which need no data
    fetch at all.

    Raises if `data` is empty or the metric type is unrecognized - the
    caller catches these and fills NaN instead."""
    if metric_type == TYPE_ARRAY:
        if not data:
            raise ValueError(f"metric '{metric}' not present in any file")
        return _build_wide_from_arrays(metric, hdf5_file_names, data, RT_unit)

    if metric_type == TYPE_DATAFRAME:
        if not data:
            raise ValueError(f"metric '{metric}' not present in any file")

        if metric in ("base_peak_intensity_maxima_per_time_range", "total_ion_current_maxima_per_time_ranges"):
            col = "base peak intensity" if metric == "base_peak_intensity_maxima_per_time_range" else "total ion current"
            values = []
            for file in hdf5_file_names:
                if file not in data or col not in data[file].columns:
                    values.append(np.nan)
                    continue
                try:
                    values.append(data[file][col].iloc[0])
                except Exception as exc:
                    logger.warning(
                        "Could not read '%s' from metric '%s' for file '%s' (%s).", col, metric, file, exc
                    )
                    values.append(np.nan)
            return pd.DataFrame({metric: values})

        if metric == "spike_in_metrics":
            return _build_spike_in_table(hdf5_file_names, data, RT_unit, spike_ins_table)

        return _build_wide_from_dataframes(metric, hdf5_file_names, data)

    raise ValueError(f"unknown metric type '{metric_type}' for metric '{metric}'")


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
