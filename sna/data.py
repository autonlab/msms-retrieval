"""Training streams and evaluation groups.

Real: MassSpecGym 1.5; an epoch visits every train query molecule (SMILES) once,
with one of its spectra drawn at random. Simulated: the AutonMol store's train
fold, spectra drawn by the store's stratified weights recomputed after removing
val/test decoys (scripts/prepare.py), endless.
"""
from __future__ import annotations

import numpy as np
import torch

from sna import paths
from sna.spectra import ADDUCTS, INSTRUMENTS, POLARITIES, SpectraTable, read_msg_tsv

FOLDS = ("train", "val", "test")


def strings(blob, offsets):
    return [bytes(blob[a:b]).decode() for a, b in zip(offsets[:-1], offsets[1:])]


class MSG:
    """MassSpecGym spectra on one GPU, their query molecules and official mass candidate lists."""

    def __init__(self, device):
        d = read_msg_tsv(paths.MSG_TSV)
        q = np.load(paths.PREP / "queries.npz")
        row = {s: i for i, s in enumerate(strings(q["smiles_blob"], q["smiles_offsets"]))}
        self.query = np.array([row[s] for s in d["smiles"]])           # spectrum -> query
        self.fold = q["fold"]                                          # query -> 0/1/2
        assert (self.fold[self.query] == np.array([FOLDS.index(f) for f in d["fold"]])).all()
        self.identifier, self.inchikey14 = d["identifier"], d["inchikey14"]
        self.table = SpectraTable(d, device=device)
        self.mol = torch.from_numpy(q["mol"]).to(device)                # query -> molecule row
        self.key = torch.from_numpy(np.unique(q["inchikey14"], return_inverse=True)[1]).to(device)
        self.cand_offsets = torch.from_numpy(q["cand_offsets"]).to(device)
        self.cand = torch.from_numpy(q["cand"]).to(device)
        self.positive = torch.from_numpy(q["positive"]).to(device)
        self.by_query = np.argsort(self.query, kind="stable")         # spectra grouped by query
        self.start = np.searchsorted(self.query[self.by_query], np.arange(len(self.fold) + 1))

    def spectra(self, query):
        return self.by_query[self.start[query]:self.start[query + 1]]

    def epoch(self, epoch, seed, batch, rank, world):
        """Spectrum rows [steps, batch] of this rank: each train query once, one random spectrum each."""
        queries = np.random.default_rng([seed, epoch]).permutation(np.nonzero(self.fold == 0)[0])
        steps = len(queries) // (batch * world)
        queries = queries[:steps * batch * world].reshape(steps, world, batch)[:, rank]
        pick = np.random.default_rng([seed, epoch, rank]).random(queries.shape)
        start = self.start[queries]
        return self.by_query[start + (pick * (self.start[queries + 1] - start)).astype(np.int64)]

    def sample_candidates(self, queries, k, generator):
        """k molecules from each query's list: the positive and k-1 random others, without
        replacement (fewer, masked, when the list is shorter). Positive at index 0."""
        start = self.cand_offsets[queries]
        n = self.cand_offsets[queries + 1] - start
        position = torch.arange(int(n.max()), device=start.device)
        valid = position[None] < n[:, None]
        key = torch.rand(valid.shape, device=start.device, generator=generator).masked_fill(~valid, 2.0)
        key[torch.arange(len(queries), device=start.device), self.positive[queries]] = -1.0
        order = key.argsort(1)[:, :k]
        rows = self.cand[(start[:, None] + order).clamp_max(len(self.cand) - 1)]
        return rows, valid.gather(1, order)

    def candidates(self, query):
        return self.cand[self.cand_offsets[query]:self.cand_offsets[query + 1]], int(self.positive[query])


class Simulated(torch.utils.data.IterableDataset):
    """Endless batches of simulated [M+H]+ Orbitrap spectra (padded to the store's 100-peak cap)."""

    # GLACIER stores sqrt intensities; its floor (relative to base) is the store loader's
    # calibration to real peak counts. FraGNNet is linear and already floored at 1%.
    POWER, FLOOR = (2.0, 1.0), (0.00061, 0.0)

    def __init__(self, batch, seed, rank, width=100):
        super().__init__()
        self.batch, self.seed, self.rank, self.width = batch, seed, rank, width

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, self.rank, worker.id if worker else 0])
        s = {k: np.load(paths.AUTONMOL / "spectra" / f"{k}.npy", mmap_mode="r") for k in
             ("molecule_row", "source", "collision_energy", "precursor_mz", "peak_start", "n_peaks")}
        peak_mz = np.load(paths.AUTONMOL / "peaks" / "mz.npy", mmap_mode="r")
        peak_intensity = np.load(paths.AUTONMOL / "peaks" / "intensity.npy", mmap_mode="r")
        cdf = np.cumsum(np.load(paths.PREP / "autonmol.npz")["weight"], dtype=np.float64)
        b, w = self.batch, self.width
        meta = dict(adduct=np.full(b, ADDUCTS.index("[M+H]+")), charge=np.ones(b, np.int64),
                    polarity=np.full(b, POLARITIES.index("positive")),
                    instrument=np.full(b, INSTRUMENTS.index("Orbitrap")), ce_valid=np.ones(b, bool))
        while True:
            rows = np.sort(np.searchsorted(cdf, rng.random(b) * cdf[-1], side="right"))
            mz, intensity, mask = np.zeros((b, w), np.float32), np.zeros((b, w), np.float32), np.zeros((b, w), bool)
            for i, r in enumerate(rows):
                start, n, source = int(s["peak_start"][r]), int(s["n_peaks"][r]), int(s["source"][r])
                m, x = peak_mz[start:start + n], peak_intensity[start:start + n]
                keep = np.isfinite(m) & np.isfinite(x) & (m > 0) & (x > 0)
                m, x = m[keep], x[keep] ** self.POWER[source]
                keep = x >= self.FLOOR[source] * x.max(initial=0.0)
                m, x = m[keep][:w], x[keep][:w]
                mz[i, :len(m)], intensity[i, :len(m)], mask[i, :len(m)] = m, x, True
            yield dict(mz=mz, intensity=intensity, mask=mask, precursor_mz=s["precursor_mz"][rows].astype(np.float32),
                       ce=s["collision_energy"][rows].astype(np.float32),
                       molecule=s["molecule_row"][rows].astype(np.int64), **meta)
