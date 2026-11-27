"""
Tests for `mzqc-to-hdf5`, using the real DIAuditor (OpenMS) output of a small Thermo DIA run
(tests/data/sample_small_raw.mzQC).

The ontology download is disabled for all tests (the bundled PSI-MS OBO is used instead),
so the tests do not depend on network access. The DIAuditor output contains neither table
values nor unknown accessions, so the table conversion and the short-name fallback for
unknown accessions are not covered here.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from macproqc_helpers.helpers import hdf5_to_mzqc, mzqc_to_hdf5
from macproqc_helpers.utils import obo

FIXTURE = Path(__file__).parent / "data" / "sample_small_raw.mzQC"

EXPECTED_KEYS = [
    "MS:4000059 ! number_of_MS1_spectra",
    "MS:4000060 ! number_of_MS2_spectra",
    "MS:4000061 ! MS1_density_quantiles",
    "MS:4000062 ! MS2_density_quantiles",
    "MS:4000067 ! MS_run_duration",
    "MS:4000069 ! m_z_acquisition_range",
    "MS:4000070 ! retention_time_acquisition_range",
    "MS:4000190 ! MS1_TIC_quantile_RT",
    "MS:4000191 ! MS2_TIC_quantile_RT",
    "MS:4000192 ! MS1_median_cycle_time",
    "MS:4000193 ! DIA_isolation_window_median_cycle_time",
    # the same name is used by two terms in the ontology: the accession number is appended
    "MS:4000194 ! DIA_isolation_window_count_4000194",
    "MS:4000195 ! DIA_isolation_window_m_z_widths",
    "MS:4000196 ! DIA_isolation_window_count_4000196",
    "MS:4000197 ! DIA_isolation_window_half_TIC_RT",
    "MS:4000198 ! DIA_isolation_window_TIC",
    "MS:4000199 ! DIA_isolation_window_peak_count",
]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Use the bundled ontology instead of downloading the current one."""

    def no_download(url):
        raise OSError("offline (test)")

    monkeypatch.setattr(obo, "_download", no_download)


def _convert(mzqc_path, tmp_path) -> Path:
    out_hdf5 = tmp_path / "out.hdf5"
    mzqc_to_hdf5.convert(argparse.Namespace(mzqc=str(mzqc_path), out_hdf5=str(out_hdf5)))
    return out_hdf5


@pytest.fixture
def hdf5(tmp_path):
    with h5py.File(_convert(FIXTURE, tmp_path), "r") as f:
        yield f


def test_all_metrics_converted(hdf5):
    assert sorted(hdf5.keys()) == sorted(EXPECTED_KEYS)


@pytest.mark.parametrize(
    "key, expected",
    [
        ("MS:4000059 ! number_of_MS1_spectra", [611]),
        ("MS:4000060 ! number_of_MS2_spectra", [4888]),
        ("MS:4000061 ! MS1_density_quantiles", [280, 636, 1018]),
        ("MS:4000194 ! DIA_isolation_window_count_4000194", [8]),
        ("MS:4000195 ! DIA_isolation_window_m_z_widths", [50.0, 50.0]),
        ("MS:4000196 ! DIA_isolation_window_count_4000196", [611, 611]),
        ("MS:4000199 ! DIA_isolation_window_peak_count", [141, 298]),
    ],
)
def test_exact_values(hdf5, key, expected):
    assert hdf5[key][:].tolist() == expected


@pytest.mark.parametrize(
    "key, expected",
    [
        ("MS:4000067 ! MS_run_duration", [1862.191978224]),
        ("MS:4000070 ! retention_time_acquisition_range", [0.0772728, 1862.269251024]),
        ("MS:4000198 ! DIA_isolation_window_TIC", [131026257.65625, 914358514.703125]),
    ],
)
def test_float_values(hdf5, key, expected):
    np.testing.assert_allclose(hdf5[key][:], expected)


def test_dtypes(hdf5):
    for key in [
        "MS:4000059 ! number_of_MS1_spectra",
        "MS:4000194 ! DIA_isolation_window_count_4000194",
    ]:
        assert np.issubdtype(hdf5[key].dtype, np.integer)
    for key in [
        "MS:4000070 ! retention_time_acquisition_range",
        "MS:4000198 ! DIA_isolation_window_TIC",
    ]:
        assert np.issubdtype(hdf5[key].dtype, np.floating)


def test_attributes(hdf5):
    # both carry the same (duplicated) name in the PSI-MS ontology, but get distinct keys
    assert (
        hdf5["MS:4000194 ! DIA_isolation_window_count_4000194"].attrs["qc_name"]
        == "DIA isolation window count"
    )
    assert (
        hdf5["MS:4000196 ! DIA_isolation_window_count_4000196"].attrs["qc_name"]
        == "DIA isolation window count"
    )

    # leading whitespace in the name is stripped
    assert hdf5["MS:4000190 ! MS1_TIC_quantile_RT"].attrs["qc_name"] == "MS1 TIC quantile RT"

    units = {
        "MS:4000067 ! MS_run_duration": ("UO:0000010", "second"),
        "MS:4000190 ! MS1_TIC_quantile_RT": ("UO:0000031", "minute"),
        "MS:4000059 ! number_of_MS1_spectra": ("UO:0000189", "count unit"),
        "MS:4000195 ! DIA_isolation_window_m_z_widths": ("MS:1000040", "m/z"),
    }
    for key, (unit_accession, unit_name) in units.items():
        assert hdf5[key].attrs["unit_accession"] == unit_accession
        assert hdf5[key].attrs["unit_name"] == unit_name
        assert hdf5[key].attrs["qc_description"]


def test_metadata(hdf5):
    assert hdf5.attrs["mzqc_label"] == "sample_small_raw"
    assert hdf5.attrs["mzqc_input_file"] == "sample_small_raw.mzML"
    assert hdf5.attrs["mzqc_instrument_model"] == "Orbitrap Eclipse"
    assert hdf5.attrs["mzqc_instrument_serial_number"] == "FSN40480"
    assert hdf5.attrs["mzqc_software_name"] == "DIAuditor"
    assert hdf5.attrs["mzqc_software_version"]
    assert hdf5.attrs["mzqc_cv_version"] == "4.2.2"
    # the ontology used for the names (here: the bundled one, as the download is disabled)
    assert hdf5.attrs["obo_source"] == "bundled:psi-ms.obo"
    assert hdf5.attrs["obo_version"]


def test_round_trip(tmp_path):
    out_hdf5 = _convert(FIXTURE, tmp_path)
    out_mzqc = tmp_path / "round_trip.mzQC"
    hdf5_to_mzqc.convert(argparse.Namespace(hdf5=str(out_hdf5), mzqc_out=str(out_mzqc)))

    def metrics(path):
        run = json.loads(Path(path).read_text())["mzQC"]["runQualities"][0]
        return {m["accession"]: m["value"] for m in run["qualityMetrics"]}

    original = metrics(FIXTURE)
    round_trip = metrics(out_mzqc)

    assert round_trip.keys() == original.keys()
    for accession, value in original.items():
        np.testing.assert_allclose(round_trip[accession], value, err_msg=accession)


def _modified_fixture(tmp_path, modify) -> Path:
    content = json.loads(FIXTURE.read_text())
    modify(content["mzQC"]["runQualities"])
    path = tmp_path / "modified.mzQC"
    path.write_text(json.dumps(content))
    return path


def test_multiple_runs_raise(tmp_path):
    path = _modified_fixture(tmp_path, lambda runs: runs.append(runs[0]))
    with pytest.raises(ValueError, match="exactly one run"):
        _convert(path, tmp_path)


def test_duplicate_metrics_raise(tmp_path):
    def duplicate_metric(runs):
        runs[0]["qualityMetrics"].append(runs[0]["qualityMetrics"][0])

    path = _modified_fixture(tmp_path, duplicate_metric)
    with pytest.raises(ValueError, match="duplicate metrics"):
        _convert(path, tmp_path)
