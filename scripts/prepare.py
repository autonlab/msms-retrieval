"""One-time data preparation every training run reads. CPU only.

    PYTHONPATH=. python scripts/prepare.py [--sim] [--force]

Writes to sna.paths.PREP; a step whose outputs exist is skipped unless --force:
  molecules.npz      sorted unique RDKit-canonical SMILES of every MassSpecGym 1.5 query and every
                     MASS-list candidate of those queries (smiles_blob, smiles_offsets)
  queries.npz        one row per query SMILES of the TSV: raw SMILES, fold, mol (molecule id), InChIKey-14 of
                     its first TSV row (10 query SMILES carry two keys in the TSV),
                     its MASS list as CSR into molecules (cand_offsets, cand; RDKit-invalid entries and
                     duplicates after canonicalisation dropped), positive = position of the query in its
                     list, n_cand / n_raw = list length after / before dropping, and spectrum_query =
                     query row of every TSV row (read_msg_tsv order)
  tokens_{molformer,chemberta}.npz   token ids of every molecule, special tokens included (ids, offsets)
  autonmol.npz       the AutonMol store: train_end, excluded (per store molecule: a MassSpecGym val/test
                     query or candidate), weight (per train spectrum, builder's rule without them)
  autonmol_tokens_molformer.npz      MoLFormer ids of every used store molecule, CSR over store rows
  graphs.npz         JESTR's molecular graph of every molecule (sna.jestr.featurize): atom one-hot indices
                     (index), bond-type bits (bits), mass (mass) over atom_offsets; local bond pairs in both
                     directions (edges) over edge_offsets
  jestr.npz          JESTR's regulariser lists: each train query's MASS list without the query, sorted by
                     Morgan Tanimoto (radius 2, 4,096 bits) to it, descending, ties in list order (molecule over
                     offsets; empty for val/test queries)
Strings are uint8 blobs with int64 offsets; token ids are uint16 with int64 offsets.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # the parallelism is the process pool

import numpy as np
import transformers
from rdkit import Chem, RDLogger

from sna.jestr import featurize
from sna.paths import AUTONMOL, CANDIDATES, MSG_TSV, PREP, TOWERS
from sna.spectra import read_msg_tsv

RDLogger.DisableLog("rdApp.*")
transformers.logging.set_verbosity_error()
FOLDS = ("train", "val", "test")
WORKERS = 48
MAX_LENGTH = {"molformer": 202, "chemberta": 512}  # MoLFormer: max_position_embeddings
# Must match the builder of the AutonMol store, whose weights stratify() recomputes.
MASS_BINS = np.arange(0.0, 1625.0, 25.0)
PROTON = 1.007276466812
SIMULATORS = ("glacier", "fragnnet")  # spectra/source codes


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def save(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def blob_of(strings) -> tuple[np.ndarray, np.ndarray]:
    encoded = [s.encode() for s in strings]
    offsets = np.zeros(len(encoded) + 1, np.int64)
    np.cumsum([len(e) for e in encoded], out=offsets[1:])
    return np.frombuffer(b"".join(encoded), np.uint8), offsets


def strings_of(blob, offsets) -> list[str]:
    raw = np.asarray(blob).tobytes()
    return [raw[a:b].decode() for a, b in zip(offsets[:-1].tolist(), offsets[1:].tolist())]


# ---- RDKit and tokenisers, in a process pool -------------------------------------------------

def canonical_and_key(smiles: str) -> tuple[str, str]:
    """RDKit canonical isomeric SMILES and first InChIKey block; '' where RDKit fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    try:
        key = Chem.MolToInchiKey(mol)[:14]
    except Exception:
        key = ""
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), key


def rdkit_map(smiles: list[str]) -> tuple[list[str], list[str]]:
    with get_context("fork").Pool(WORKERS) as pool:
        pairs = pool.map(canonical_and_key, smiles, chunksize=2000)
    return [p[0] for p in pairs], [p[1] for p in pairs]


def tokenizer(tower: str):
    name, revision = TOWERS[tower]
    return transformers.AutoTokenizer.from_pretrained(name, revision=revision,
                                                      trust_remote_code=tower == "molformer")


_TOKENIZER = None


def _load(tower: str) -> None:
    global _TOKENIZER
    _TOKENIZER = tokenizer(tower)


def _encode(job):
    smiles, max_length = job
    ids = _TOKENIZER(smiles)["input_ids"]
    full = np.fromiter(map(len, ids), np.int64, len(ids))
    # <unk> or a character the pre-tokeniser drops (MoLFormer's removes what its regex does not match)
    lossy = sum("".join(_TOKENIZER.convert_ids_to_tokens(x, skip_special_tokens=True)) != s
                for x, s in zip(ids, smiles))
    long = np.nonzero(full > max_length)[0].tolist()
    if long:  # re-encode with the tokeniser's own truncation, which keeps both special tokens
        cut = _TOKENIZER([smiles[i] for i in long], truncation=True, max_length=max_length)["input_ids"]
        for i, c in zip(long, cut):
            ids[i] = c
    lengths = np.fromiter(map(len, ids), np.int64, len(ids))
    flat = np.fromiter(itertools.chain.from_iterable(ids), np.uint16, int(lengths.sum()))
    return flat, lengths, full, lossy


def tokenize(tower: str, smiles: list[str], chunk: int = 10_000) -> tuple[np.ndarray, np.ndarray, dict]:
    """Token ids with special tokens, truncated to MAX_LENGTH[tower], as CSR; plus statistics."""
    tok, max_length = tokenizer(tower), MAX_LENGTH[tower]
    if len(tok) > 2 ** 16:
        raise ValueError(f"{tower}: vocabulary does not fit uint16")
    jobs = [(smiles[i:i + chunk], max_length) for i in range(0, len(smiles), chunk)]
    with get_context("fork").Pool(WORKERS, _load, (tower,)) as pool:
        parts = pool.map(_encode, jobs)
    lengths = np.concatenate([p[1] for p in parts])
    full = np.concatenate([p[2] for p in parts])
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    stats = {
        "tokenizer": TOWERS[tower][0], "revision": TOWERS[tower][1], "class": type(tok).__name__,
        "vocab": len(tok), "max_length": max_length, "sequences": len(lengths),
        "tokens": int(offsets[-1]), "truncated": int((full > max_length).sum()),
        "not_round_tripping": int(sum(p[3] for p in parts)),
        "length_percentiles_before_truncation": {
            str(q): float(np.percentile(full, q)) for q in (0, 1, 50, 90, 99, 99.9, 100)},
    }
    return np.concatenate([p[0] for p in parts]), offsets, stats


# ---- MassSpecGym ------------------------------------------------------------------------------

def build_msg() -> dict:
    tsv = read_msg_tsv(MSG_TSV)
    spectrum_fold = np.array([FOLDS.index(f) for f in tsv["fold"]], np.int8)
    query, first, spectrum_query = np.unique(tsv["smiles"], return_index=True, return_inverse=True)
    fold = spectrum_fold[first]
    if not np.array_equal(fold[spectrum_query], spectrum_fold):
        raise ValueError("a query SMILES spans two folds")
    keys_of = {}
    for q, key in zip(spectrum_query.tolist(), tsv["inchikey14"].tolist()):
        keys_of.setdefault(q, set()).add(key)
    several_keys = sorted(str(query[q]) for q, keys in keys_of.items() if len(keys) > 1)

    with open(CANDIDATES["mass"]) as handle:
        listed = json.load(handle)
    lists = [listed[s] for s in query]
    del listed
    raw = list(set(query.tolist()).union(*lists))
    log(f"canonicalising {len(raw):,} query and candidate SMILES")
    canonical, _ = rdkit_map(raw)
    canonical_of = dict(zip(raw, canonical))
    if not all(canonical_of[s] for s in query):
        raise ValueError("RDKit rejects a query SMILES")
    molecules = sorted(set(canonical) - {""})
    id_of = {s: i for i, s in enumerate(molecules)}

    cand, cand_offsets, positive = [], [0], []
    invalid = duplicate = 0
    for s, entries in zip(query, lists):
        start, seen = len(cand), set()
        for entry in entries:
            c = canonical_of[entry]
            if not c:
                invalid += 1
            elif id_of[c] in seen:
                duplicate += 1
            else:
                seen.add(id_of[c])
                cand.append(id_of[c])
        target = id_of[canonical_of[s]]
        if target not in seen:  # dedup leaves at most one copy, so exactly one
            raise ValueError(f"query {s} is missing from its own MASS list")
        positive.append(cand.index(target, start) - start)
        cand_offsets.append(len(cand))
    molecule = np.array([id_of[canonical_of[s]] for s in query], np.int32)
    cand_offsets = np.asarray(cand_offsets, np.int64)
    n_cand, n_raw = np.diff(cand_offsets), np.array([len(x) for x in lists])
    positive = np.asarray(positive, np.int32)

    # Uniformly random ranking: E[1/rank] = H_K / K for a list of K; averaged over spectra.
    harmonic = np.cumsum(1.0 / np.arange(1, n_cand.max() + 1))
    random_rr = harmonic[n_cand - 1] / n_cand
    save(PREP / "molecules.npz", **dict(zip(("smiles_blob", "smiles_offsets"), blob_of(molecules))))
    save(PREP / "queries.npz", **dict(zip(("smiles_blob", "smiles_offsets"), blob_of(query.tolist()))),
         fold=fold, mol=molecule, inchikey14=tsv["inchikey14"][first].astype("S14"),
         cand_offsets=cand_offsets, cand=np.asarray(cand, np.int32), positive=positive,
         n_cand=n_cand.astype(np.int32), n_raw=n_raw.astype(np.int32),
         spectrum_query=spectrum_query.astype(np.int32))
    return {
        "tsv_spectra": {f: int((spectrum_fold == i).sum()) for i, f in enumerate(FOLDS)},
        "queries": {f: int((fold == i).sum()) for i, f in enumerate(FOLDS)},
        "query_molecules": len(set(molecule.tolist())),
        "query_molecules_in_two_folds": len(set(zip(molecule.tolist(), fold.tolist())))
        - len(set(molecule.tolist())),
        "query_smiles_with_several_tsv_inchikey14": several_keys,
        "smiles_canonicalised": len(raw),
        "invalid_candidate_smiles": sum(not c for c in canonical),
        "invalid_candidate_slots": invalid,
        "duplicate_slots_after_canonicalisation": duplicate,
        "queries_with_dropped_slots": int((n_raw > n_cand).sum()),
        "molecules": len(molecules),
        "candidate_slots": {"raw": int(n_raw.sum()), "kept": len(cand)},
        "list_length": {"min": int(n_cand.min()), "median": float(np.median(n_cand)),
                        "max": int(n_cand.max()), "below_256": int((n_cand < 256).sum())},
        "positive_at_position_0": int((positive == 0).sum()),
        "uniform_random_mrr": {f: float(random_rr[spectrum_query[spectrum_fold == i]].mean())
                               for i, f in enumerate(FOLDS)},
    }


def build_tokens(tower: str) -> dict:
    z = np.load(PREP / "molecules.npz")
    ids, offsets, stats = tokenize(tower, strings_of(z["smiles_blob"], z["smiles_offsets"]))
    save(PREP / f"tokens_{tower}.npz", ids=ids, offsets=offsets)
    return stats


# ---- AutonMol store -------------------------------------------------------------------------

def stratify(train, molecule_row, source, exact_mass, real_precursor):
    """The store builder's weights, recomputed on the given train mask. Half the budget per simulator;
    within it 1/n_spectra(molecule) x p_real(bin)/n_molecules(bin) over 25 Da [M+H]+ bins."""
    edges = MASS_BINS
    n_bins, n_molecules = len(edges) - 1, len(exact_mass)
    p_real = np.histogram(np.clip(real_precursor, edges[0], edges[-1] - 1e-6), edges)[0]
    p_real = p_real / p_real.sum()
    grid = np.arange(0.0, edges[-1] + 1.0, 1.0)
    real_cdf = np.cumsum(np.histogram(real_precursor, grid)[0]) / len(real_precursor)
    weight = np.zeros(len(molecule_row), dtype=np.float64)
    report = {}
    for code, name in enumerate(SIMULATORS):
        rows = np.nonzero(train & (source == code))[0]
        molecule = molecule_row[rows]
        per_molecule = np.bincount(molecule, minlength=n_molecules).astype(np.float64)
        present = np.nonzero(per_molecule)[0]
        bin_of = np.clip(np.digitize(exact_mass[present] + PROTON, edges) - 1, 0, n_bins - 1)
        in_bin = np.bincount(bin_of, minlength=n_bins)
        molecule_bin = np.zeros(n_molecules, dtype=np.int64)
        molecule_bin[present] = bin_of
        b = molecule_bin[molecule]
        w = p_real[b] / np.maximum(in_bin[b], 1) / per_molecule[molecule]
        w *= 0.5 / w.sum()
        weight[rows] = w
        share = np.bincount(molecule, weights=w, minlength=n_molecules)[present] / 0.5
        precursor = exact_mass[molecule].astype(np.float64) + PROTON
        cdf = np.cumsum(np.histogram(precursor, grid, weights=w)[0]) / w.sum()
        used = (in_bin > 0) & (p_real > 0)
        report[name] = {
            "train_spectra": int(len(rows)),
            "train_molecules": int(len(present)),
            "real_mass_share_in_empty_bins": float(p_real[in_bin == 0].sum()),
            "fewest_molecules_in_a_used_bin": int(in_bin[used].min()),
            "effective_molecules": float(1.0 / np.sum(share ** 2)),
            "w1_to_real_train_precursor_da": float(np.abs(cdf - real_cdf).sum()),
            "w1_unweighted_da": float(np.abs(
                np.cumsum(np.histogram(precursor, grid)[0]) / len(precursor) - real_cdf).sum()),
        }
    weight *= train.sum() / weight.sum()        # mean 1 over train rows
    return weight.astype(np.float32), report


def compare_weights(new, old, train, molecule_row, source, exact_mass) -> dict:
    """New/stored weight ratio on rows both use: constant within a simulator x mass bin unless a bin moved."""
    n_bins = len(MASS_BINS) - 1
    bin_of = np.clip(np.digitize(exact_mass + PROTON, MASS_BINS) - 1, 0, n_bins - 1)
    keep = train & (old > 0)
    group = source[keep].astype(np.int64) * n_bins + bin_of[molecule_row[keep]]
    ratio = new[keep].astype(np.float64) / old[keep]
    count = np.bincount(group, minlength=2 * n_bins)
    mean = np.bincount(group, ratio, 2 * n_bins) / np.maximum(count, 1)
    deviation = np.abs(ratio / mean[group] - 1)
    scale = mean[count > 0] / (new[keep].sum(dtype=np.float64) / old[keep].sum(dtype=np.float64))
    return {
        "rows_compared": int(keep.sum()),
        "rows_zero_in_store": int((train & (old == 0)).sum()),
        "rows_zero_in_store_nonzero_now": int((train & (old == 0) & (new > 0)).sum()),
        "rows_nonzero_in_store_zero_now": int((keep & (new == 0)).sum()),
        "max_relative_deviation_within_simulator_bin": float(deviation.max()),
        "molecules_deviating_over_1e-5": int(len(np.unique(molecule_row[keep][deviation > 1e-5]))),
        "bin_ratio_over_global_ratio": [float(scale.min()), float(scale.max())],
    }


def build_autonmol() -> dict:
    """K = RDKit InChIKey-14 of every MASS and FORMULA candidate of every val/test query (TSV fold), plus
    the RDKit and TSV keys of those queries. A store molecule is excluded when its stored key or the RDKit
    key of its stored SMILES is in K; the used molecules are the train-fold ones not excluded."""
    with open(AUTONMOL / "provenance.json") as handle:  # 232 MB; only fold_slices is needed
        start, train_end = json.load(handle)["fold_slices"]["train"]
    if start != 0:
        raise ValueError("the store's train slice does not start at 0")
    tsv = read_msg_tsv(MSG_TSV)
    heldout = tsv["fold"] != "train"
    queries = set(tsv["smiles"][heldout].tolist())
    listed = {}
    for name in ("mass", "formula"):
        with open(CANDIDATES[name]) as handle:
            candidates = json.load(handle)
        listed[name] = set().union(*(candidates[q] for q in queries))
        del candidates
    raw = list(queries | listed["mass"] | listed["formula"])
    log(f"InChIKeys of {len(raw):,} val/test queries and candidates")
    _, keys = rdkit_map(raw)
    key_of = dict(zip(raw, keys))
    rdkit_query_keys = {key_of[q] for q in queries}
    tsv_query_keys = set(tsv["inchikey14"][heldout].tolist())
    exclusion = {"mass": rdkit_query_keys | tsv_query_keys | {key_of[s] for s in listed["mass"]}}
    exclusion["mass+formula"] = exclusion["mass"] | {key_of[s] for s in listed["formula"]}
    for keyset in exclusion.values():
        keyset.discard("")

    m, s = AUTONMOL / "molecules", AUTONMOL / "spectra"
    fold = np.asarray(np.load(m / "fold.npy", mmap_mode="r"))
    in_msg = np.asarray(np.load(m / "in_msg.npy", mmap_mode="r"))
    stored_key = np.load(m / "inchikey14.npy", mmap_mode="r")
    smiles = strings_of(np.load(m / "smiles_blob.npy", mmap_mode="r"),
                        np.load(m / "smiles_offsets.npy", mmap_mode="r"))
    log(f"canonical SMILES and InChIKeys of {len(smiles):,} store molecules")
    canonical, rdkit_key = rdkit_map(smiles)
    rdkit_key = np.array(rdkit_key, "S14")

    def matches(keyset):
        keyset = np.array(sorted(keyset), "S14")
        return np.isin(stored_key, keyset), np.isin(rdkit_key, keyset)

    by_stored, by_rdkit = matches(exclusion["mass+formula"])
    excluded = by_stored | by_rdkit
    excluded_mass = np.logical_or(*matches(exclusion["mass"]))
    train_fold = fold == 0
    used = train_fold & ~excluded

    log("weights")
    molecule_row = np.asarray(np.load(s / "molecule_row.npy", mmap_mode="r")[:train_end])
    source = np.asarray(np.load(s / "source.npy", mmap_mode="r")[:train_end])
    old = np.asarray(np.load(s / "sample_weight.npy", mmap_mode="r")[:train_end])
    if np.any(fold[molecule_row] != 0):
        raise ValueError("the store's train slice holds a val/test molecule")
    exact_mass = np.load(m / "exact_mass.npy", mmap_mode="r").astype(np.float64)
    train = used[molecule_row]
    real_precursor = tsv["precursor_mz"][tsv["fold"] == "train"].astype(np.float64)
    weight, report = stratify(train, molecule_row, source, exact_mass, real_precursor)
    comparison = compare_weights(weight, old, train, molecule_row, source, exact_mass)
    n_spectra = np.bincount(molecule_row, minlength=len(fold))

    rows = np.nonzero(used)[0]
    fallback = sum(not canonical[i] for i in rows.tolist())
    log(f"MoLFormer tokens of {len(rows):,} used store molecules")
    ids, row_offsets, stats = tokenize("molformer", [canonical[i] or smiles[i] for i in rows.tolist()])
    lengths = np.zeros(len(fold), np.int64)
    lengths[rows] = np.diff(row_offsets)
    save(PREP / "autonmol.npz", train_end=np.int64(train_end), excluded=excluded, weight=weight)
    save(PREP / "autonmol_tokens_molformer.npz", ids=ids,
         offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    return {
        "train_end": int(train_end),
        "store_molecules": len(fold),
        "train_fold_molecules": int(train_fold.sum()),
        "heldout_query_smiles": len(queries),
        "heldout_candidate_smiles": {k: len(v) for k, v in listed.items()},
        "rdkit_key_failures": sum(not k for k in keys),
        "tsv_query_keys_not_rdkit_keys": len(tsv_query_keys - rdkit_query_keys),
        "exclusion_keys": {k: len(v) for k, v in exclusion.items()},
        "store_rdkit_failures": sum(not c for c in canonical),
        "store_key_disagreements": int((stored_key != rdkit_key).sum()),
        "excluded": {
            "by_store_fold": [int((excluded & (fold == i)).sum()) for i in range(len(FOLDS))],
            "train_fold_molecules": int((excluded & train_fold).sum()),
            "train_fold_msg_molecules": int((excluded & train_fold & in_msg).sum()),
            "train_spectra": int(n_spectra[excluded & train_fold].sum()),
            "store_weight_share": float(old[~train].sum(dtype=np.float64) / old.sum(dtype=np.float64)),
            "train_fold_only_by_stored_key": int((train_fold & by_stored & ~by_rdkit).sum()),
            "train_fold_only_by_rdkit_key": int((train_fold & by_rdkit & ~by_stored).sum()),
            "train_fold_molecules_mass_list_only": int((excluded_mass & train_fold).sum()),
        },
        "used": {"molecules": len(rows), "spectra": int(train.sum())},
        "weights": report,
        "weights_vs_store": comparison,
        "token_fallbacks_to_stored_smiles": fallback,
        "tokens": stats,
    }


# ---- JESTR ------------------------------------------------------------------------------------

def _featurize_chunk(smiles):
    parts = [featurize(s) for s in smiles]
    return ([np.concatenate([p[i] for p in parts]) for i in range(4)],
            np.array([len(p[0]) for p in parts], np.int64), np.array([len(p[3]) for p in parts], np.int64))


def build_graphs() -> dict:
    z = np.load(PREP / "molecules.npz")
    smiles = strings_of(z["smiles_blob"], z["smiles_offsets"])
    with get_context("fork").Pool(WORKERS) as pool:
        parts = pool.map(_featurize_chunk, [smiles[i:i + 5000] for i in range(0, len(smiles), 5000)])
    index, bits, mass, edges = (np.concatenate([p[0][i] for p in parts]) for i in range(4))
    atoms, bonds = np.concatenate([p[1] for p in parts]), np.concatenate([p[2] for p in parts])
    save(PREP / "graphs.npz", index=index, bits=bits, mass=mass, edges=edges,
         atom_offsets=np.concatenate([[0], np.cumsum(atoms)]), edge_offsets=np.concatenate([[0], np.cumsum(bonds)]))
    return {"molecules": len(smiles), "atoms": int(atoms.sum()), "directed_bonds": int(bonds.sum()),
            "atoms_per_molecule": {"min": int(atoms.min()), "median": float(np.median(atoms)), "max": int(atoms.max())},
            "atoms_outside_a_one_hot_field": int((index == 255).any(1).sum()),
            "unknown_element_atoms": int((index[:, 0] == 14).sum()), "molecules_without_bonds": int((bonds == 0).sum())}


def _sort_by_tanimoto(job):
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    out = []
    for query, smiles, ids in job:
        fp = lambda s: generator.GetFingerprint(Chem.MolFromSmiles(s))
        similarity = np.array(DataStructs.BulkTanimotoSimilarity(fp(query), [fp(c) for c in smiles]))
        out.append(ids[np.argsort(-similarity, kind="stable")])
    return out


def build_jestr() -> dict:
    z, q = np.load(PREP / "molecules.npz"), dict(np.load(PREP / "queries.npz"))  # dict: an NpzFile re-reads per access
    smiles = strings_of(z["smiles_blob"], z["smiles_offsets"])
    jobs, chunk = [], []
    train = np.nonzero(q["fold"] == 0)[0]
    for i in train.tolist():
        ids = q["cand"][q["cand_offsets"][i]:q["cand_offsets"][i + 1]]
        ids = ids[ids != q["mol"][i]]  # the query itself is not a candidate to push away
        chunk.append((smiles[q["mol"][i]], [smiles[c] for c in ids.tolist()], ids))
        if len(chunk) == 200:
            jobs.append(chunk)
            chunk = []
    with get_context("fork").Pool(WORKERS) as pool:
        lists = [x for part in pool.map(_sort_by_tanimoto, jobs + [chunk]) for x in part]
    counts = np.zeros(len(q["fold"]), np.int64)
    counts[train] = [len(x) for x in lists]
    save(PREP / "jestr.npz", molecule=np.concatenate(lists).astype(np.int32),
         offsets=np.concatenate([[0], np.cumsum(counts)]))
    return {"train_queries": len(train), "list_length": {"min": int(counts[train].min()),
                                                          "median": float(np.median(counts[train]))}}


# ---- driver -----------------------------------------------------------------------------------

STEPS = {  # name: (outputs, build)
    "msg": (("molecules.npz", "queries.npz"), build_msg),
    "tokens_molformer": (("tokens_molformer.npz",), lambda: build_tokens("molformer")),
    "tokens_chemberta": (("tokens_chemberta.npz",), lambda: build_tokens("chemberta")),
    "autonmol": (("autonmol.npz", "autonmol_tokens_molformer.npz"), build_autonmol),
    "graphs": (("graphs.npz",), build_graphs),
    "jestr": (("jestr.npz",), build_jestr),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="rebuild outputs that exist")
    parser.add_argument("--sim", action="store_true", help="also prepare the simulated-spectra store")
    args = parser.parse_args()
    t0 = time.time()
    PREP.mkdir(parents=True, exist_ok=True)
    for name, (outputs, build) in STEPS.items():
        if name == "autonmol" and not args.sim:
            continue
        if not args.force and all((PREP / o).exists() for o in outputs):
            log(f"{name}: outputs exist, skipped")
            continue
        log(f"{name}: start")
        start = time.time()
        stats = build()
        log(f"{name}: {json.dumps(stats, indent=1)} ({time.time() - start:.1f} s)")
    log(f"finished in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
