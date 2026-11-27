"""
Loading of the PSI-MS ontology (OBO format, parsed with pronto) to look up term names.

A local OBO file given by the user is always used. Otherwise, the current ontology is
downloaded from GitHub and, if this is not possible (e.g. in an offline container), the
copy bundled with this package (src/macproqc_helpers/data/psi-ms.obo) is used.
"""

import io
import logging
import urllib.request
from collections import Counter
from importlib import resources
from typing import Optional, Set

import pronto

logger = logging.getLogger(__name__)

PSI_MS_OBO_URL = "https://raw.githubusercontent.com/HUPO-PSI/psi-ms-CV/master/psi-ms.obo"
DOWNLOAD_TIMEOUT = 30  # seconds


class Ontology:
    """Term name lookup in a pronto ontology, with the information where it was loaded from."""

    def __init__(self, ontology: pronto.Ontology, source: str):
        self.ontology = ontology
        self.source = source
        self.version: str = ontology.metadata.data_version or ""

    def name(self, accession: str) -> Optional[str]:
        """The name of the term with the given accession, or None if unknown."""
        try:
            # only terms (no relationships / typedefs)
            name = self.ontology.get_term(accession).name
        except KeyError:
            return None
        return name.strip() if name else None

    def ambiguous_names(self) -> Set[str]:
        """The names used by more than one (not obsolete) term."""
        counts = Counter(
            term.name.strip() for term in self.ontology.terms() if term.name and not term.obsolete
        )
        return {name for name, count in counts.items() if count > 1}


def parse_obo(data: bytes, source: str = "") -> Ontology:
    """Parse an OBO file given as bytes."""
    return Ontology(pronto.Ontology(io.BytesIO(data)), source)


def _download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as response:
        return response.read()


def _read_bundled() -> Ontology:
    bundled = resources.files("macproqc_helpers").joinpath("data", "psi-ms.obo")
    return parse_obo(bundled.read_bytes(), source=f"bundled:{bundled.name}")


def load_psi_ms_ontology(obo_path: Optional[str] = None, url: str = PSI_MS_OBO_URL) -> Ontology:
    """
    Load the PSI-MS ontology: from the OBO file at `obo_path`, if given. Otherwise the
    current one from `url` or, if it cannot be downloaded, the copy bundled with this package.
    """
    if obo_path:
        with open(obo_path, "rb") as obo_file:
            return parse_obo(obo_file.read(), source=obo_path)

    try:
        ontology = parse_obo(_download(url), source=url)
        if len(ontology.ontology.terms()) == 0:
            raise ValueError("no terms found")
        return ontology
    except Exception as e:  # any download / parsing problem: use the bundled file
        logger.warning("Could not load the ontology from %s (%s), using the bundled copy.", url, e)
        return _read_bundled()
