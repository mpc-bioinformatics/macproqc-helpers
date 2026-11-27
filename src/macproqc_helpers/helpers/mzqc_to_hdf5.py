"""
Convert an mzQC file (with a single run) into the per-run HDF5 format of macproqc.

This is the reverse of `hdf5_to_mzqc`: every quality metric of the run becomes one
HDF5 entry with the key "ACCESSION ! short_name" (see `utils/hdf5.py`). Tables
(mzQC values given as dict) become HDF5 groups with one dataset per column.
The run metadata (label, input file, software, CV) is stored as root attributes.

Metric and column names are taken from the PSI-MS ontology: a given local OBO file, or the
current one downloaded from GitHub (if offline: the copy bundled with this package). The short name of a metric (the
HDF5 key suffix, also used by the visualization) is its sanitised ontology name. Names that
are used by several terms in the ontology (e.g. MS:4000194 and MS:4000196) get the accession
number appended, as the visualization identifies metrics by their short name only.
"""

import argparse
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import h5py
from mzqc import MZQCFile as qc

import macproqc_helpers.utils.hdf5 as mzhdf5
from macproqc_helpers.utils.obo import PSI_MS_OBO_URL, Ontology, load_psi_ms_ontology

logger = logging.getLogger(__name__)


def argparse_setup(subparsers: argparse._SubParsersAction):
    parser = subparsers.add_parser(
        "mzqc-to-hdf5",
        description="Convert an mzQC file (containing a single run) into the HDF5 format used by macproqc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-mzqc", help="Path to the mzQC file (with exactly one run)", required=True)
    parser.add_argument("-out_hdf5", help="Output path for the generated HDF5 file", required=True)
    parser.add_argument(
        "-obo",
        help=(
            f"Local PSI-MS OBO file to use. If not given, the current ontology is downloaded from "
            f"{PSI_MS_OBO_URL} (if offline: the copy bundled with macproqc-helpers is used)"
        ),
        default=None,
    )
    parser.set_defaults(func=convert)


def _sanitise(name: str) -> str:
    """Replace runs of non-alphanumeric characters by '_' (usable as short name / attribute)."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name.strip()).strip("_")


def _short_name(accession: str, name: str, ambiguous_names: set, used: set) -> str:
    """
    The short name for a metric: its sanitised name, with the accession number appended,
    if the name is used by several ontology terms or already used in this file.
    """
    short = _sanitise(name) or _sanitise(accession)
    if name in ambiguous_names or short in used:
        short = f"{short}_{_sanitise(accession.split(':')[-1])}"
    return short


def _get(obj: Any, attr: str) -> Any:
    """Get an attribute from a pymzqc object or a key from a dict."""
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _unit(metric: qc.QualityMetric) -> Tuple[str, str]:
    """Accession and name of the metric's unit (empty strings, if missing or not a single unit)."""
    unit = metric.unit
    if unit is None or isinstance(unit, list):
        return "", ""
    return (_get(unit, "accession") or "", _get(unit, "name") or "")


def _array_type(values: List[Any]) -> Optional[str]:
    """The HDF5 dtype for a flat list of values, or None if not storable."""
    if all(isinstance(v, bool) for v in values):
        return "bool" if values else "float64"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int64"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return "float64"
    if all(isinstance(v, str) for v in values):
        return "str"
    return None


def _column_name(key: str, ontology: Ontology) -> str:
    """The HDF5 column name "ACCESSION ! name" for an mzQC table column."""
    return f"{key} ! {ontology.name(key) or key}"


def add_metric_to_hdf5(
    f: h5py.File, metric: qc.QualityMetric, name: str, short_name: str, ontology: Ontology
) -> bool:
    """Write a single mzQC metric into the HDF5 file. Returns False, if it was skipped."""
    accession = metric.accession
    description = metric.description or ""
    value = metric.value

    if isinstance(value, dict):
        columns = list(value.keys())
        column_data = [list(value[c]) for c in columns]
        column_types = [_array_type(d) for d in column_data]
        if any(t is None for t in column_types):
            logger.warning("Skipping table '%s': unsupported column values.", accession)
            return False
        mzhdf5.add_table_to_hdf5(
            f=f,
            qc_acc=accession,
            qc_short_name=short_name,
            qc_name=name,
            qc_description=description,
            column_names=[_column_name(c, ontology) for c in columns],
            column_data=column_data,
            column_types=[t for t in column_types if t is not None],
        )
        return True

    values = value if isinstance(value, list) else [value]
    value_type = _array_type(values) if value is not None else None
    if value_type is None:
        logger.warning("Skipping metric '%s': unsupported value '%s'.", accession, value)
        return False

    unit_accession, unit_name = _unit(metric)
    mzhdf5.add_entry_to_hdf5(
        f=f,
        qc_acc=accession,
        qc_short_name=short_name,
        qc_name=name,
        qc_description=description,
        value=values,
        value_shape=(len(values),),
        value_type=value_type,
        unit_accession=unit_accession,
        unit_name=unit_name,
    )
    return True


def add_metadata_to_hdf5(
    f: h5py.File, mzqc: qc.MzQcFile, run: qc.RunQuality, ontology: Ontology
) -> None:
    """Store the run metadata of the mzQC file as root attributes of the HDF5 file."""
    attrs: Dict[str, str] = {
        "mzqc_creation_date": str(mzqc.creationDate or ""),
        "obo_version": ontology.version,
        "obo_source": ontology.source,
    }

    meta = run.metadata
    attrs["mzqc_label"] = meta.label or ""

    if meta.inputFiles:
        input_file = meta.inputFiles[0]
        attrs["mzqc_input_file"] = input_file.name or ""
        attrs["mzqc_input_file_format"] = _get(input_file.fileFormat, "name") or ""
        for prop in input_file.fileProperties or []:
            prop_name = _sanitise(prop.name or prop.accession or "")
            if prop_name and prop.value is not None:
                attrs[f"mzqc_{prop_name}"] = str(prop.value)

    if meta.analysisSoftware:
        software = meta.analysisSoftware[0]
        attrs["mzqc_software_name"] = str(software.value or software.name or "")
        attrs["mzqc_software_version"] = software.version or ""

    if mzqc.controlledVocabularies:
        attrs["mzqc_cv_version"] = mzqc.controlledVocabularies[0].version or ""

    for k, v in attrs.items():
        f.attrs[k] = v


def convert(args: argparse.Namespace) -> None:
    with open(args.mzqc) as mzqc_file:
        mzqc = qc.JsonSerialisable.from_json(mzqc_file.read())

    runs = mzqc.runQualities or []
    if len(runs) != 1:
        raise ValueError(
            f"The mzQC file '{args.mzqc}' must contain exactly one run, but contains {len(runs)}."
        )
    if mzqc.setQualities:
        logger.warning("Ignoring the setQualities of '%s'.", args.mzqc)
    run = runs[0]

    accessions = [m.accession for m in run.qualityMetrics]
    duplicates = sorted({a for a in accessions if accessions.count(a) > 1})
    if duplicates:
        raise ValueError(f"The mzQC file '{args.mzqc}' contains duplicate metrics: {duplicates}")

    ontology = load_psi_ms_ontology(getattr(args, "obo", None))
    ambiguous_names = ontology.ambiguous_names()

    with h5py.File(args.out_hdf5, "w") as out_h5:
        used_short_names: set = set()
        for metric in run.qualityMetrics:
            # prefer the ontology name, fall back to the name given in the mzQC file
            name = (ontology.name(metric.accession) or metric.name or "").strip()
            short_name = _short_name(metric.accession, name, ambiguous_names, used_short_names)
            if add_metric_to_hdf5(out_h5, metric, name, short_name, ontology):
                used_short_names.add(short_name)

        add_metadata_to_hdf5(out_h5, mzqc, run, ontology)
