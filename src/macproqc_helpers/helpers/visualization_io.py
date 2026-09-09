"""
Lazy per-metric HDF5 readers for `visualization.py` - each opens/closes files
one at a time and only materializes the one metric requested, so peak memory
does not scale with (number of files) x (number of metrics). See the
`visualization` module docstring for the overall architecture.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from macproqc_helpers.helpers.visualization_registry import TYPE_SINGLE

logger = logging.getLogger(__name__)


def read_single_values(hdf5_paths: List[str], registry: Dict[str, dict]) -> pd.DataFrame:
    """Read all "single" metrics for every file into one small DataFrame
    (one row per file, in the same order as `hdf5_paths`). Single values are
    cheap (scalars), so unlike arrays/dataframes they are all read once up
    front rather than lazily per plot."""
    single_keys = {
        short: entry["example_full_key"]
        for short, entry in registry.items()
        if short != "filename" and entry.get("type") == TYPE_SINGLE and entry.get("example_full_key")
    }

    data: Dict[str, List[Any]] = {"filename": []}
    for short in single_keys:
        data[short] = []

    for path in hdf5_paths:
        fname = Path(path).stem
        data["filename"].append(fname)
        with h5py.File(path, "r") as hdf:
            for short, full_key in single_keys.items():
                if full_key in hdf:
                    data[short].append(hdf[full_key][0])
                else:
                    logger.warning("Metric '%s' missing in file '%s'.", short, fname)
                    data[short].append(np.nan)

    return pd.DataFrame(data)


def read_array_metric(hdf5_paths: List[str], full_key: str) -> Dict[str, np.ndarray]:
    """Read one array-type metric for every file, opening/closing each file
    one at a time. Files missing the metric are omitted (with a warning)."""
    result: Dict[str, np.ndarray] = {}
    for path in hdf5_paths:
        fname = Path(path).stem
        with h5py.File(path, "r") as hdf:
            if full_key not in hdf:
                logger.warning("Metric '%s' missing in file '%s'.", full_key, fname)
                continue
            result[fname] = hdf[full_key][:]
    return result


def _dataframe_from_group(group: h5py.Group) -> pd.DataFrame:
    """Materialize an HDF5 group (table) into a pandas DataFrame with short
    column names, honoring the `column_order` attribute if present. Values
    are explicitly sliced (`[:]`) so the resulting DataFrame is safe to use
    after the owning HDF5 file has been closed."""
    column_order = group.attrs.get("column_order")
    df = pd.DataFrame({col: group[col][:] for col in group.keys()})
    if column_order:
        columns = [c for c in column_order.split("|") if c in df.columns]
        if columns:
            df = df[columns]
    df.columns = df.columns.str.split(" ! ").str[-1]
    return df


def read_dataframe_metric(hdf5_paths: List[str], full_key: str) -> Dict[str, pd.DataFrame]:
    """Read one dataframe-type metric (an HDF5 group) for every file. Files
    missing the metric are omitted (with a warning)."""
    result: Dict[str, pd.DataFrame] = {}
    for path in hdf5_paths:
        fname = Path(path).stem
        with h5py.File(path, "r") as hdf:
            if full_key not in hdf:
                logger.warning("Metric '%s' missing in file '%s'.", full_key, fname)
                continue
            result[fname] = _dataframe_from_group(hdf[full_key])
    return result


def read_dataframe_metric_single_file(hdf5_path: str, full_key: str) -> Optional[pd.DataFrame]:
    """Same as `read_dataframe_metric` but scoped to a single file - used by
    figures that process one file at a time (e.g. MS1 ion maps) to avoid
    holding more than one file's data in memory at once."""
    fname = Path(hdf5_path).stem
    with h5py.File(hdf5_path, "r") as hdf:
        if full_key not in hdf:
            logger.warning("Metric '%s' missing in file '%s'.", full_key, fname)
            return None
        return _dataframe_from_group(hdf[full_key])


def list_group_columns(hdf5_path: str, full_key: str) -> List[str]:
    """Return the short column names available in a dataframe/group metric
    for a single file, without reading any actual data."""
    with h5py.File(hdf5_path, "r") as hdf:
        if full_key not in hdf:
            return []
        group = hdf[full_key]
        column_order = group.attrs.get("column_order")
        full_cols = column_order.split("|") if column_order else list(group.keys())
        return [c.split(" ! ")[-1].strip() for c in full_cols]


def read_group_columns(
    hdf5_path: str, full_key: str, short_col_names: List[str]
) -> Dict[str, Optional[np.ndarray]]:
    """Read only the requested columns of a dataframe/group metric for a
    single file (opening it only once), leaving the rest untouched. Used by
    `plot_additional_headers`, where a group may have many columns but only
    one or two are needed for a given plot."""
    result: Dict[str, Optional[np.ndarray]] = {name: None for name in short_col_names}
    wanted = set(short_col_names)
    with h5py.File(hdf5_path, "r") as hdf:
        if full_key not in hdf:
            return result
        group = hdf[full_key]
        for col in group.keys():
            short = col.split(" ! ")[-1].strip()
            if short in wanted:
                result[short] = group[col][:]
    return result
