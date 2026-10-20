"""Every external input in one place, read from the environment or from `.env` at the repository root
(see `.env.example`). Everything this repo writes goes under OUT."""
import os
from pathlib import Path

_ENV = Path(__file__).resolve().parents[1] / ".env"
if _ENV.exists():
    for _line in _ENV.read_text().splitlines():
        _key, _sep, _value = _line.partition("=")
        if _sep and not _key.lstrip().startswith("#"):
            os.environ.setdefault(_key.strip(), _value.strip())


def _env(key: str) -> str:
    return os.environ.get(key) or f"<{key} is not set>"  # fails on first use, naming the variable


MSG_TSV = Path(_env("SNA_MSG_TSV"))
CANDIDATES = {  # the two official MassSpecGym 1.5 retrieval lists; results use "mass"
    "mass": Path(_env("SNA_MSG_CANDIDATES")) / "MassSpecGym1.5_retrieval_candidates_mass.json",
    "formula": Path(_env("SNA_MSG_CANDIDATES")) / "MassSpecGym1.5_retrieval_candidates_formula.json",
}
AUTONMOL = Path(_env("SNA_AUTONMOL"))  # simulated-spectra store

ENCODERS = {name: _env(f"SNA_CKPT_{name.upper()}") for name in (
    "dreams_ssl", "dreams_contrastive", "dreams_vf", "dreams_vf_control",
    "msjepa", "msjepa_geom_base", "msjepa_geom_mcos", "msjepa_geom_vf")}
TOWERS = {  # Hugging Face id, pinned snapshot
    "molformer": ("ibm-research/MoLFormer-XL-both-10pct", "361063d0ad524ef77cf39b08469f6be770dc550f"),
    "chemberta": ("seyonec/ChemBERTa-zinc-base-v1", "761d6a18cf99db371e0b43baf3e2d21b3e865a20"),
}

OUT = Path(_env("SNA_OUT"))
PREP = OUT / "prep"
RUNS = OUT / "runs"
