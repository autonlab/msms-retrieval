"""Invariants of the scripts/prepare.py outputs; skipped until they are built."""
import os

os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from sna.paths import AUTONMOL, MSG_TSV, PREP, TOWERS

MAX_LENGTH = {"molformer": 202, "chemberta": 512}
rng = np.random.default_rng(0)


def load(name):
    if not (PREP / name).exists():
        pytest.skip(f"{PREP / name} not built")
    return np.load(PREP / name)


def check_csr(offsets, values):
    assert offsets.dtype == np.int64 and offsets[0] == 0 and offsets[-1] == len(values)
    assert np.all(np.diff(offsets) >= 0)


def strings(blob, offsets):
    check_csr(offsets, blob)
    raw = blob.tobytes()
    return [raw[a:b].decode() for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())]


def tokenizer(tower):
    from transformers import AutoTokenizer
    name, revision = TOWERS[tower]
    return AutoTokenizer.from_pretrained(name, revision=revision, trust_remote_code=tower == "molformer")


def check_tokens(tower, ids, offsets, rows, smiles):
    """Stored ids of ``rows`` equal a fresh encoding of ``smiles``; every sequence is <bos> ... <eos>."""
    lengths = np.diff(offsets)
    full = lengths > 0
    assert lengths.max() <= MAX_LENGTH[tower] and lengths[full].min() >= 3
    tok = tokenizer(tower)
    assert np.all(ids[offsets[:-1][full]] == tok.bos_token_id)
    assert np.all(ids[offsets[1:][full] - 1] == tok.eos_token_id)
    fresh = tok(smiles, truncation=True, max_length=MAX_LENGTH[tower])["input_ids"]
    for r, f in zip(rows, fresh):
        assert ids[offsets[r]:offsets[r + 1]].tolist() == f


def test_molecules_sorted_unique():
    m = load("molecules.npz")
    smiles = strings(m["smiles_blob"], m["smiles_offsets"])
    assert smiles[0] and all(a < b for a, b in zip(smiles, smiles[1:]))


def test_queries():
    n_molecules = len(load("molecules.npz")["smiles_offsets"]) - 1
    q = load("queries.npz")
    offsets, cand, molecule, positive = q["cand_offsets"], q["cand"], q["mol"], q["positive"]
    query = strings(q["smiles_blob"], q["smiles_offsets"])
    n = len(query)
    check_csr(offsets, cand)
    assert len(offsets) == n + 1 and len(q["fold"]) == len(molecule) == len(positive) == len(q["inchikey14"]) == n
    assert np.array_equal(np.diff(offsets), q["n_cand"]) and np.all(q["n_cand"] <= q["n_raw"])
    assert cand.min() >= 0 and cand.max() < n_molecules
    row = np.repeat(np.arange(n), q["n_cand"])
    assert len(np.unique(row * n_molecules + cand)) == len(cand)  # no molecule twice in a list
    assert np.all((positive >= 0) & (positive < q["n_cand"]))
    assert np.array_equal(cand[offsets[:-1] + positive], molecule)
    assert np.count_nonzero(cand == molecule[row]) == n  # exactly one positive per list

    tsv = pd.read_csv(MSG_TSV, sep="\t", usecols=["smiles", "fold"])
    sq = q["spectrum_query"]
    assert len(sq) == len(tsv) and np.array_equal(np.asarray(query, object)[sq], tsv.smiles.to_numpy())
    assert np.array_equal(q["fold"][sq], tsv.fold.map({"train": 0, "val": 1, "test": 2}).to_numpy())


@pytest.mark.parametrize("tower", ["molformer", "chemberta"])
def test_tokens(tower):
    m = load("molecules.npz")
    t = load(f"tokens_{tower}.npz")
    check_csr(t["offsets"], t["ids"])
    assert len(t["offsets"]) == len(m["smiles_offsets"]) and np.all(np.diff(t["offsets"]) > 0)
    rows = np.sort(rng.choice(len(t["offsets"]) - 1, 1000, replace=False))
    smiles = strings(m["smiles_blob"], m["smiles_offsets"])
    check_tokens(tower, t["ids"], t["offsets"], rows, [smiles[r] for r in rows])


def test_autonmol():
    z = load("autonmol.npz")
    train_end, excluded, weight = int(z["train_end"]), z["excluded"], z["weight"]
    fold = np.load(AUTONMOL / "molecules/fold.npy")
    molecule_row = np.load(AUTONMOL / "spectra/molecule_row.npy", mmap_mode="r")[:train_end]
    source = np.load(AUTONMOL / "spectra/source.npy", mmap_mode="r")[:train_end]
    assert len(excluded) == len(fold) and len(weight) == train_end
    assert np.all(fold[molecule_row] == 0)  # the train slice holds no val/test-fold molecule
    used = (fold == 0) & ~excluded
    used_row = used[molecule_row]
    assert np.all(weight[~used_row] == 0) and np.all(np.isfinite(weight)) and np.all(weight >= 0)
    total = weight.sum(dtype=np.float64)  # the builder's normalisation: mean 1, half per simulator
    assert abs(total / used_row.sum() - 1) < 1e-5
    for code in (0, 1):
        assert abs(weight[source == code].sum(dtype=np.float64) / total - 0.5) < 1e-5

    stored_key = np.load(AUTONMOL / "molecules/inchikey14.npy", mmap_mode="r")
    tsv = pd.read_csv(MSG_TSV, sep="\t", usecols=["inchikey", "fold"])
    heldout = tsv.inchikey[tsv.fold != "train"].to_numpy().astype("S14")
    assert not np.isin(stored_key[used], heldout).any()
    # A sample of val/test MASS-list candidates: none of their keys is a used store molecule's key.
    m, q = load("molecules.npz"), load("queries.npz")
    smiles = strings(m["smiles_blob"], m["smiles_offsets"])
    row = np.repeat(np.arange(len(q["fold"])), q["n_cand"])
    sample = rng.choice(q["cand"][q["fold"][row] > 0], 2000, replace=False)
    keys = np.array([Chem.MolToInchiKey(Chem.MolFromSmiles(smiles[i]))[:14] for i in sample], "S14")
    assert not np.isin(stored_key[used], keys).any()


def test_autonmol_tokens():
    excluded = load("autonmol.npz")["excluded"]
    t = load("autonmol_tokens_molformer.npz")
    fold = np.load(AUTONMOL / "molecules/fold.npy")
    used = (fold == 0) & ~excluded
    check_csr(t["offsets"], t["ids"])
    assert len(t["offsets"]) == len(fold) + 1 and np.array_equal(np.diff(t["offsets"]) > 0, used)
    blob = np.load(AUTONMOL / "molecules/smiles_blob.npy", mmap_mode="r")
    offsets = np.load(AUTONMOL / "molecules/smiles_offsets.npy", mmap_mode="r")
    rows = np.sort(rng.choice(np.nonzero(used)[0], 500, replace=False))
    smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(bytes(blob[offsets[r]:offsets[r + 1]]).decode())) for r in rows]
    check_tokens("molformer", t["ids"], t["offsets"], rows, smiles)
