import argparse
import math

import h5py
import numpy as np
import pyopenms
import pytest

from macproqc_helpers.helpers import collect_metrics_from_mzml


FILTER_THRESHOLD = 0.1
REPORT_UP_TO_CHARGE = 3
BASE_PEAK_TIC_UP_TO = 10  # minutes


def _make_spectrum(rt, ms_level, mzs, intensities, precursor_mz=None, precursor_charge=None):
    spectrum = pyopenms.MSSpectrum()
    spectrum.setRT(rt)
    spectrum.setMSLevel(ms_level)
    spectrum.set_peaks((np.asarray(mzs, dtype=np.float64), np.asarray(intensities, dtype=np.float64)))
    if ms_level == 2:
        precursor = pyopenms.Precursor()
        precursor.setMZ(precursor_mz)
        precursor.setCharge(precursor_charge)
        spectrum.setPrecursors([precursor])
    return spectrum


@pytest.fixture
def small_mzml(tmp_path):
    """A tiny, fully deterministic run: 6 MS1 spectra (RT 0..50s) each
    followed by 2 MS2 spectra, with hand-picked peaks so every metric can be
    checked against an independently computed reference value."""
    exp = pyopenms.MSExperiment()

    rng = np.random.default_rng(0)
    ms1_rts = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0]
    charges = [1, 2, 2, 3, 1, 2, 3, 4, 2, 1, 2, 3]
    charge_iter = iter(charges)

    for rt in ms1_rts:
        mzs = np.sort(rng.uniform(300, 1500, size=20))
        intens = rng.uniform(1, 1000, size=20)
        # guarantee a clear, reproducible base peak per spectrum
        intens[0] = 5000.0 + rt
        exp.addSpectrum(_make_spectrum(rt, 1, mzs, intens))

        for j in range(2):
            ms2_mzs = np.sort(rng.uniform(100, 1000, size=10))
            ms2_intens = rng.uniform(1, 500, size=10)
            exp.addSpectrum(
                _make_spectrum(
                    rt + 1 + j,
                    2,
                    ms2_mzs,
                    ms2_intens,
                    precursor_mz=500.0 + j,
                    precursor_charge=next(charge_iter),
                )
            )

    path = tmp_path / "small.mzML"
    pyopenms.MzMLFile().store(str(path), exp)
    return str(path), exp


def _run_collect(mzml_path, tmp_path, **overrides):
    out_hdf5 = tmp_path / "out.hdf5"
    args = argparse.Namespace(
        mzml=mzml_path,
        out_hdf5=str(out_hdf5),
        base_peak_tic_up_to=BASE_PEAK_TIC_UP_TO,
        filter_threshold=FILTER_THRESHOLD,
        report_up_to_charge=REPORT_UP_TO_CHARGE,
        ms1_map_rt_bins=50,
        ms1_map_mz_bins=50,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    collect_metrics_from_mzml.collect(args)
    return h5py.File(out_hdf5, "r")


def _reference_metrics(exp):
    """Straightforward, unoptimized re-derivation of the metrics used as the
    ground truth - deliberately not sharing code with collect_metrics_from_mzml
    so a bug in one is unlikely to be mirrored in the other."""
    ms1_rt, ms1_tic, ms1_peaks = [], [], []
    ms2_rt, ms2_tic = [], []
    accumulated_ms1_tic = 0.0
    accumulated_ms2_tic = 0.0
    raw_ms1_peaks_above_threshold = []

    all_basepeaks = []
    for spectrum in exp.getSpectra():
        mz, intens = spectrum.get_peaks()
        tic = sum(intens)
        all_basepeaks.append(max(intens) if len(intens) else 0)
        if spectrum.getMSLevel() == 1:
            ms1_rt.append(spectrum.getRT())
            ms1_tic.append(tic)
            ms1_peaks.append(len(intens))
            accumulated_ms1_tic += tic
        elif spectrum.getMSLevel() == 2:
            ms2_rt.append(spectrum.getRT())
            ms2_tic.append(tic)
            accumulated_ms2_tic += tic

    base_peak_intensity_max = max(all_basepeaks)
    threshold = base_peak_intensity_max * FILTER_THRESHOLD
    for spectrum in exp.getSpectra():
        if spectrum.getMSLevel() != 1:
            continue
        mz, intens = spectrum.get_peaks()
        for m, i in zip(mz, intens):
            if i >= threshold:
                raw_ms1_peaks_above_threshold.append(i)

    return {
        "num_ms1": len(ms1_rt),
        "num_ms2": len(ms2_rt),
        "accumulated_ms1_tic": accumulated_ms1_tic,
        "accumulated_ms2_tic": accumulated_ms2_tic,
        "rt_first": exp.getSpectrum(0).getRT(),
        "rt_last": exp.getSpectrum(exp.getNrSpectra() - 1).getRT(),
        "base_peak_intensity_max": base_peak_intensity_max,
        "raw_ms1_peak_sum_above_threshold": sum(raw_ms1_peaks_above_threshold),
        "ms1_num_peaks": sorted(ms1_peaks),
    }


def test_metrics_match_reference(small_mzml, tmp_path):
    mzml_path, exp = small_mzml
    ref = _reference_metrics(exp)

    with _run_collect(mzml_path, tmp_path) as out:
        assert int(out["MS:4000059 ! nr_MS1"][0]) == ref["num_ms1"]
        assert int(out["MS:4000060 ! nr_MS2"][0]) == ref["num_ms2"]

        assert out["MS:4000029 ! accumulated_MS1_TIC"][()] == pytest.approx(ref["accumulated_ms1_tic"])
        assert out["MS:4000030 ! accumulated_MS2_TIC"][()] == pytest.approx(ref["accumulated_ms2_tic"])

        rt_range = out["MS:4000070 ! RT_range"][()]
        assert rt_range[0] == pytest.approx(ref["rt_first"])
        assert rt_range[1] == pytest.approx(ref["rt_last"])

        assert out["MS:4000202 ! base_peak_intensity_max"][()] == pytest.approx(ref["base_peak_intensity_max"])

        density_q = out["MS:4000061 ! MS1_density_quantiles"][()]
        assert list(density_q) == [
            int(np.quantile(ref["ms1_num_peaks"], 0.25)),
            int(np.quantile(ref["ms1_num_peaks"], 0.50)),
            int(np.quantile(ref["ms1_num_peaks"], 0.75)),
        ]


def test_freq_max_matches_naive_On2_reference():
    rng = np.random.default_rng(1)
    rt = np.sort(rng.uniform(0, 500, size=250))

    naive = []
    for i in range(len(rt)):
        naive.append(np.sum(np.logical_and(rt >= rt[i], rt <= rt[i] + 60)))
    expected = max(naive) / 60.0

    assert collect_metrics_from_mzml._freq_max_hz(list(rt)) == pytest.approx(expected)


def test_ms1_map_conserves_total_intensity_above_threshold(small_mzml, tmp_path):
    mzml_path, exp = small_mzml
    ref = _reference_metrics(exp)

    with _run_collect(mzml_path, tmp_path) as out:
        grid_intensity = out["LOCAL:rtMzIntensityMS1 ! MS1_map/intensity"][()]
        assert grid_intensity.sum() == pytest.approx(ref["raw_ms1_peak_sum_above_threshold"], rel=1e-6)
        # bounded by the fixed grid size, regardless of the number of peaks in the run
        assert len(grid_intensity) <= 50 * 50


def test_falls_back_to_full_load_for_unindexed_mzml(small_mzml, tmp_path):
    mzml_path, exp = small_mzml

    # Strip the <indexedmzML> wrapper to simulate an unindexed mzML (e.g.
    # some tdf2mzml-produced Bruker output), which OnDiscMSExperiment can't
    # open for streaming access.
    data = open(mzml_path, "rb").read()
    start = data.find(b"<mzML")
    end = data.find(b"</mzML>") + len(b"</mzML>")
    unindexed_path = tmp_path / "unindexed.mzML"
    unindexed_path.write_bytes(b'<?xml version="1.0" encoding="ISO-8859-1"?>\n' + data[start:end])

    ondisc = pyopenms.OnDiscMSExperiment()
    assert not ondisc.openFile(str(unindexed_path)), "fixture is not actually unindexed"

    with _run_collect(str(unindexed_path), tmp_path) as out:
        assert int(out["MS:4000059 ! nr_MS1"][0]) == 6
