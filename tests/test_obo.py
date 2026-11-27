"""
Tests for loading the PSI-MS ontology (via pronto): a given local OBO file is always used,
otherwise the ontology is downloaded, with the bundled copy as fallback. The download itself
is mocked (no network access).
"""

import pytest

from macproqc_helpers.utils import obo

SMALL_OBO = b"""format-version: 1.2
data-version: 9.9.9

[Term]
id: MS:0000001
name:  term with leading space

[Term]
id: MS:0000002
name: duplicated name

[Term]
id: MS:0000003
name: duplicated name

[Term]
id: MS:0000004
name: obsolete term
is_obsolete: true

[Term]
id: MS:0000005
name: obsolete term

[Typedef]
id: has_units
name: has_units
"""


def _fail(url):
    raise OSError("offline (test)")


def test_parse_obo():
    ontology = obo.parse_obo(SMALL_OBO, source="test")

    assert ontology.version == "9.9.9"
    assert ontology.source == "test"
    assert ontology.name("MS:0000001") == "term with leading space"
    assert ontology.name("MS:0000004") == "obsolete term"
    assert ontology.ontology["MS:0000004"].obsolete
    # Typedefs are not terms
    assert ontology.name("has_units") is None
    assert ontology.name("MS:9999999") is None
    # an obsolete term does not make a name ambiguous
    assert ontology.ambiguous_names() == {"duplicated name"}


def test_given_obo_is_preferred(monkeypatch, tmp_path):
    def download_not_expected(url):
        raise AssertionError("no download expected, if a local OBO file is given")

    monkeypatch.setattr(obo, "_download", download_not_expected)
    local = tmp_path / "local.obo"
    local.write_bytes(SMALL_OBO.replace(b"9.9.9", b"1.0.0"))

    ontology = obo.load_psi_ms_ontology(str(local))

    assert ontology.version == "1.0.0"
    assert ontology.source == str(local)


def test_download_without_given_obo(monkeypatch):
    monkeypatch.setattr(obo, "_download", lambda url: SMALL_OBO)

    ontology = obo.load_psi_ms_ontology()

    assert ontology.version == "9.9.9"
    assert ontology.source == obo.PSI_MS_OBO_URL


def test_missing_given_obo_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(obo, "_download", lambda url: SMALL_OBO)

    with pytest.raises(FileNotFoundError):
        obo.load_psi_ms_ontology(str(tmp_path / "missing.obo"))


def test_fallback_to_bundled_when_offline(monkeypatch):
    monkeypatch.setattr(obo, "_download", _fail)

    assert obo.load_psi_ms_ontology().source == "bundled:psi-ms.obo"


def test_fallback_on_empty_download(monkeypatch):
    monkeypatch.setattr(obo, "_download", lambda url: b"<html>not an ontology</html>")

    assert obo.load_psi_ms_ontology().source == "bundled:psi-ms.obo"


@pytest.fixture
def bundled(monkeypatch):
    monkeypatch.setattr(obo, "_download", _fail)
    return obo.load_psi_ms_ontology()


def test_bundled_ontology(bundled):
    assert bundled.source == "bundled:psi-ms.obo"
    assert bundled.version
    assert bundled.name("MS:4000059") == "number of MS1 spectra"
    # the ontology itself has a leading space in these names: it is stripped
    assert bundled.name("MS:4000190") == "MS1 TIC quantile RT"
    # column / unit accessions of other ontologies
    assert bundled.name("UO:0000191") == "fraction"
    assert bundled.name("NCIT:C150827") == "Number of Occurrences"
    # MS:4000194 and MS:4000196 share their name
    assert "DIA isolation window count" in bundled.ambiguous_names()
