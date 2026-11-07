import argparse
from pathlib import Path

import h5py
import numpy as np
import pytest

from macproqc_helpers.helpers.collect_metrics_from_pia_output import collect

RESOURCES = Path(__file__).parent / "data"


@pytest.mark.parametrize("store_all_infos", [False, True])
def test_header_only_pia_files_give_zero_identifications(tmp_path, store_all_infos):
    """PIA output files which only contain headers (no identifications) must be
    handled like empty files instead of raising."""
    out_hdf5 = tmp_path / "out.hdf5"
    args = argparse.Namespace(
        pia_peptides=str(RESOURCES / "pia_empty-peptides.csv"),
        pia_proteins=str(RESOURCES / "pia_empty-proteins.mzTab"),
        pia_PSMs=str(RESOURCES / "pia_empty-PSMs.mzTab"),
        out_hdf5=str(out_hdf5),
        store_all_infos=store_all_infos,
    )

    collect(args)

    with h5py.File(out_hdf5, "r") as f:
        assert f["MS:1003250 ! nr_peptides"][0] == 0
        assert f["MS:1003251 ! nr_PSMs"][0] == 0
        assert f["MS:1003327 ! nr_protein_groups"][0] == 0
        assert f["MS:4000214 ! nr_accessions"][0] == 0
        assert np.all(np.isnan(f["MS:4000206 ! filtered_psms_ppm_error_quartiles"][()]))
        assert np.isnan(f["MS:4000178 ! filtered_psms_ppm_error_mean"][0])
        assert np.isnan(f["MS:4000179 ! filtered_psms_ppm_error_sigma"][0])
