"""
Metric registry for `visualization.py`: type detection (single/array/dataframe)
plus an editable, persisted config (`pca_raw` / `pca_all` / `in_summary_table`
flags per metric) controlling which metrics feed the PCA plots and the
summary table. See the `visualization` module docstring for the overall
architecture.
"""

import json
import logging
import os
from typing import Dict, List

import h5py

logger = logging.getLogger(__name__)

HDF5_ACCESSION_SEPARATOR = "!"

# Metric "type" values used in the metric registry
TYPE_SINGLE = "single"
TYPE_ARRAY = "array"
TYPE_DATAFRAME = "dataframe"

# Metrics whose per-file column set is not fixed/enumerable (it differs by
# instrument vendor, e.g. Thermo vs. Bruker headers). Tracked as a single
# opaque registry entry - the actual columns are discovered dynamically from
# the data every run (see `plot_additional_headers`), not enumerated here.
DYNAMIC_COLUMN_METRICS = {"Extracted_Headers", "Extracted_Log_Headers"}

# Default metric order/columns for 00_table_summary.csv. Only used to seed
# the `in_summary_table` flag for metrics the very first time they are seen
# in a brand-new (or not-yet-edited) metric registry file.
DEFAULT_SUMMARY_TABLE_METRICS = [
    "filename",
    "startTime",
    "RT_range",
    "nr_MS1",
    "nr_MS2",
    "accumulated_MS1_TIC",
    "accumulated_MS2_TIC",
    "base_peak_intensity_max",
    "total_ion_current_max",
    "base_peak_intensity_maxima_per_time_range",
    "total_ion_current_maxima_per_time_ranges",
    "MS2_prec_charge_fraction",
    "RT_MS1_quantiles",
    "RT_MS2_quantiles",
    "RT_TIC_quantiles",
    "MS1_freq_max",
    "MS2_freq_max",
    "MS1_density_quantiles",
    "MS2_density_quantiles",
    "MS1_TIC_change_quantiles",
    "MS1_TIC_quantiles",
    "nr_PSMs",
    "nr_peptides",
    "nr_protein_groups",
    "nr_accessions",
    "PSM_charge_fractions",
    "PSM_missed_cleavages_fractions",
    "nr_features",
    "nr_ident_features",
    "features_charges",
    "ident_features_charge",
    "spike_in_metrics",
]

# Default metrics used for the PCA on raw (pre-identification) data. Only
# used to seed the `pca_raw` flag for newly-discovered metrics.
DEFAULT_PCA_RAW_METRICS = [
    "RT_range",
    "nr_MS1",
    "nr_MS2",
    "accumulated_MS1_TIC",
    "accumulated_MS2_TIC",
    "base_peak_intensity_max",
    "total_ion_current_max",
    "MS2_prec_charge_fraction",
    "RT_MS1_quantiles",
    "RT_MS2_quantiles",
    "RT_TIC_quantiles",
    "MS1_freq_max",
    "MS2_freq_max",
    "MS1_density_quantiles",
    "MS2_density_quantiles",
    "MS1_TIC_change_quantiles",
    "MS1_TIC_quantiles",
]

# Additional metrics used for the PCA on all (post-identification) data (on
# top of DEFAULT_PCA_RAW_METRICS). Only used to seed the `pca_all` flag for
# newly-discovered metrics.
DEFAULT_PCA_ALL_EXTRA_METRICS = [
    "nr_PSMs",
    "nr_peptides",
    "nr_protein_groups",
    "nr_accessions",
    "PSM_charge_fractions",
    "PSM_missed_cleavages_fractions",
    "nr_features",
    "nr_ident_features",
    "features_charges",
    "ident_features_charge",
]


def _short_name(full_key: str) -> str:
    """Extract the short metric name from a full HDF5 key like 'ACC ! short_name'."""
    return full_key.split(HDF5_ACCESSION_SEPARATOR)[-1].strip()


def build_metric_registry(hdf5_paths: List[str]) -> Dict[str, dict]:
    """
    Scan all hdf5 files (metadata only - no data is read) and determine, for
    every metric key, whether it is a single value, an array or a dataframe
    (HDF5 group).

    A metric is classified as "array" if it has length != 1 (including 0) in
    ANY of the provided files, and "single" only if it consistently has
    length 1 in every file that contains it. Groups are always "dataframe".
    This avoids misclassifying a metric as "single" just because one
    specific file happens to only have a single entry (e.g. a raw file with
    only one PSM), or because it happens to be empty in every file (e.g. an
    error-quartile array for a file with 0 relevant PSMs).

    Returns
    -------
    Dict[str, dict]
        Mapping of short metric name -> {"type", "example_full_key",
        "dynamic_columns"}. Only metrics actually present in at least one of
        `hdf5_paths` are included. PCA/summary-table flags are not set here,
        see `seed_default_flags`.
    """
    non_single: Dict[str, bool] = {}
    example_full_key: Dict[str, str] = {}
    is_group: Dict[str, bool] = {}

    for path in hdf5_paths:
        with h5py.File(path, "r") as hdf:
            for key in hdf.keys():
                short = _short_name(key)
                example_full_key.setdefault(short, key)
                item = hdf[key]
                if isinstance(item, h5py.Group):
                    is_group[short] = True
                elif isinstance(item, h5py.Dataset):
                    is_group.setdefault(short, False)
                    length = item.shape[0] if len(item.shape) > 0 else 1
                    if length != 1:
                        non_single[short] = True

    registry: Dict[str, dict] = {}
    for short in sorted(example_full_key.keys()):
        if is_group.get(short, False):
            metric_type = TYPE_DATAFRAME
        elif non_single.get(short, False):
            metric_type = TYPE_ARRAY
        else:
            metric_type = TYPE_SINGLE
        registry[short] = {
            "type": metric_type,
            "example_full_key": example_full_key[short],
            "dynamic_columns": short in DYNAMIC_COLUMN_METRICS,
        }
    return registry


def merge_metric_registry(fresh: Dict[str, dict], existing: Dict[str, dict]) -> Dict[str, dict]:
    """
    Combine freshly-scanned type info (`fresh`, current batch of hdf5 files)
    with previously saved user flags (`existing`, loaded from the registry
    file). Only contains keys present in `fresh` (i.e. relevant to the
    current batch of files) - `type`/`example_full_key`/`dynamic_columns`
    are always taken from `fresh` (data-derived facts), while `pca_raw` /
    `pca_all` / `in_summary_table` are carried over from `existing` when
    present there, so user edits stick across runs.
    """
    merged: Dict[str, dict] = {}
    for short, entry in fresh.items():
        merged_entry = dict(entry)
        if short in existing:
            for flag in ("pca_raw", "pca_all", "in_summary_table"):
                if flag in existing[short]:
                    merged_entry[flag] = existing[short][flag]
        merged[short] = merged_entry
    return merged


def seed_default_flags(registry: Dict[str, dict]) -> Dict[str, dict]:
    """Fill in `pca_raw` / `pca_all` / `in_summary_table` for any entry that
    doesn't already have them (i.e. metrics seen for the very first time),
    mirroring today's hard-coded default metric lists so behavior is
    unchanged out of the box. Modifies `registry` in place and returns it."""
    pca_all_metrics = set(DEFAULT_PCA_RAW_METRICS) | set(DEFAULT_PCA_ALL_EXTRA_METRICS)
    for short, entry in registry.items():
        entry.setdefault("pca_raw", short in DEFAULT_PCA_RAW_METRICS)
        entry.setdefault("pca_all", short in pca_all_metrics)
        entry.setdefault("in_summary_table", short in DEFAULT_SUMMARY_TABLE_METRICS)
    return registry


def load_metric_registry(path: str) -> Dict[str, dict]:
    with open(path) as f:
        return json.load(f)


def save_metric_registry(registry: Dict[str, dict], path: str) -> None:
    with open(path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)


def get_or_update_metric_registry(hdf5_paths: List[str], registry_path: str) -> Dict[str, dict]:
    """
    Build the metric registry for the current run, merging in any existing
    registry file (preserving user-edited pca_raw/pca_all/in_summary_table
    flags for metrics still relevant to this batch of files) and persisting
    the (possibly extended) result back to disk. Metrics only known from a
    previous, differently-instrumented run are kept in the persisted file
    for future reference, but are not part of the registry used during this
    run (they are irrelevant to the current hdf5 files anyway).
    """
    fresh = build_metric_registry(hdf5_paths)
    fresh.setdefault(
        "filename", {"type": TYPE_SINGLE, "example_full_key": "", "dynamic_columns": False}
    )

    existing: Dict[str, dict] = {}
    if os.path.isfile(registry_path):
        try:
            existing = load_metric_registry(registry_path)
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not read metric registry file '%s' (%s); regenerating it.",
                registry_path,
                exc,
            )

    active = merge_metric_registry(fresh, existing)
    seed_default_flags(active)

    persisted = dict(existing)
    persisted.update(active)

    try:
        save_metric_registry(persisted, registry_path)
    except OSError as exc:
        logger.warning("Could not write metric registry file '%s' (%s).", registry_path, exc)

    return active


def metrics_with_flag(
    registry: Dict[str, dict], flag: str, preferred_order: List[str]
) -> List[str]:
    """Return metric short names for which `registry[name][flag]` is truthy,
    ordered by `preferred_order` first, then any remaining flagged metrics
    (e.g. user-added ones) alphabetically."""
    selected = {short for short, entry in registry.items() if entry.get(flag)}
    ordered = [short for short in preferred_order if short in selected]
    remaining = sorted(selected - set(ordered))
    return ordered + remaining
