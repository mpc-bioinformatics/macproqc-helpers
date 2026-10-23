import argparse
import json
from pathlib import Path

import pytest

from macproqc_helpers.helpers import create_thermorawfileparser_xic_config

RESOURCES = Path(__file__).parent / "data"
SPIKEINS_MZTAB = RESOURCES / "spikeins_small.mzTab"
SPIKEINS_MZTAB_2 = RESOURCES / "spikeins_small_2.mzTab"


@pytest.fixture
def spikeins_csv(tmp_path):
    """Some spike-ins (RT and rt-tol in seconds): three identified in the mzTab
    resource, one with some PSM failing the score filter and one without an ID."""
    path = tmp_path / "spikeins.csv"
    path.write_text(
        "name,sequence,mz,RT,mz-tol,mz-tol-unit,rt-tol\n"
        "iRT02,YILAGVENSK/2,547.2980,3600,10,ppm,300\n"
        "iRT04,GTFIIDPAAVIR/2,636.8692,690,10,ppm,100\n"
        "iRT07-noRT,VEATFGVDESNAK/2,683.8279,-1,10,ppm,600\n"
        "iRT08-niID,TPVITGAPYEYR/2,683.8537,4500,10,ppm,300\n",
        encoding="utf-8",
    )
    return str(path)


def _run_create(spikeins_csv, tmp_path, iidents=None):
    out_json = tmp_path / "out.json"
    out_idents = tmp_path / "out_idents.csv"
    args = argparse.Namespace(
        icsv=spikeins_csv,
        iidents=iidents,
        ojson=str(out_json),
        oidentifications=str(out_idents),
    )

    # run the create_thermorawfileparser_xic_config
    create_thermorawfileparser_xic_config.create(args)

    # parse the created json
    with open(out_json, encoding="utf-8") as f:
        entries = {e["comment"]: e for e in json.load(f)}

    # parse the identification counts file
    idents = dict(line.split(",") for line in out_idents.read_text(encoding="utf-8").splitlines())
    return entries, {seq: int(n) for seq, n in idents.items()}


def test_without_identifications(spikeins_csv, tmp_path):
    entries, idents = _run_create(spikeins_csv, tmp_path)

    assert idents == {}

    # check values without found ID, directly from the fake-file
    assert entries["iRT02"]["rt_start"] == 55
    assert entries["iRT02"]["rt_end"] == 65
    assert entries["iRT02"]["tolerance"] == 10.0
    assert entries["iRT02"]["tolerance_unit"] == "ppm"
    assert "rt_tol" not in entries["iRT02"]

    # no initial RT given, no id
    assert entries["iRT07-noRT"]["rt_start"] == 0
    assert entries["iRT07-noRT"]["rt_end"] == 9999

    # no id, initial RT
    assert entries["iRT08-niID"]["rt_start"] == 70
    assert entries["iRT08-niID"]["rt_end"] == 80


def test_with_identifications(spikeins_csv, tmp_path):
    entries, idents = _run_create(spikeins_csv, tmp_path, iidents=[str(SPIKEINS_MZTAB)])

    # GTFIIDPAAVIR has 7 PSMs, but one below threshold
    assert idents == {"YILAGVENSK": 2, "VEATFGVDESNAK": 3, "GTFIIDPAAVIR": 6}

    # IDs at 630s and 690s -> centered at 11 min, given tolerance (1 min) is larger than half the spread
    assert entries["iRT02"]["rt_start"] == pytest.approx(57.9660683333)
    assert entries["iRT02"]["rt_end"] == pytest.approx(67.9660683333)

    # no initial RT given, no id
    assert entries["iRT07-noRT"]["rt_start"] == pytest.approx(49.5150188888)
    assert entries["iRT07-noRT"]["rt_end"] == pytest.approx(69.5150188888)

    # no id, initial RT
    assert entries["iRT08-niID"]["rt_start"] == 70
    assert entries["iRT08-niID"]["rt_end"] == 80


def test_with_multiple_identification_files(spikeins_csv, tmp_path):
    entries, idents = _run_create(
        spikeins_csv, tmp_path, iidents=[str(SPIKEINS_MZTAB), str(SPIKEINS_MZTAB_2)]
    )

    assert idents == {"YILAGVENSK": 2, "VEATFGVDESNAK": 3, "GTFIIDPAAVIR": 6, "TPVITGAPYEYR": 3}

    # IDs at 630s and 690s -> centered at 11 min, given tolerance (1 min) is larger than half the spread
    assert entries["iRT02"]["rt_start"] == pytest.approx(57.9660683333)
    assert entries["iRT02"]["rt_end"] == pytest.approx(67.9660683333)

    # no id, initial RT
    assert entries["iRT08-niID"]["rt_start"] == pytest.approx(70.4442816666)
    assert entries["iRT08-niID"]["rt_end"] == pytest.approx(80.4442816666)
