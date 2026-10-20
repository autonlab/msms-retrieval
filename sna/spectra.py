"""MassSpecGym spectra and the batch every spectrum encoder reads.

Batch: a dict of tensors on one device, B spectra, N <= the encoder's ``max_peaks``.
  mz, intensity  float32 [B, N]  top-N peaks by intensity, sorted by m/z,
                                 intensity divided by the base peak, zero padded
  mask           bool    [B, N]
  precursor_mz   float32 [B]
  adduct         long    [B]     index into ADDUCTS (0 = unknown)
  charge         long    [B]     signed charge
  polarity       long    [B]     index into POLARITIES
  instrument     long    [B]     index into INSTRUMENTS (the MS-JEPA vocabulary)
  ce             float32 [B]     collision energy in eV, 0 where missing
  ce_valid       bool    [B]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

ADDUCTS = ("<unk>", "[M+H]+", "[M+Na]+")
POLARITIES = ("<unk>", "positive", "negative")
INSTRUMENTS = ("Orbitrap", "QTOF", "IT-FT", "Unknown")
META = ("precursor_mz", "adduct", "charge", "polarity", "instrument", "ce", "ce_valid")
# training augmentation (simulated re-acquisition): peak dropout, multiplicative log-normal
# intensity noise, relative m/z jitter. The precursor is never perturbed.
AUGMENT = dict(peak_dropout=0.2, intensity_sigma=0.3, mz_ppm=5.0)


def read_msg_tsv(path) -> dict[str, np.ndarray]:
    """The official TSV as flat arrays, peaks in CSR form (``offsets``).

    Peaks with intensity <= 0 (11 of 8.4M) are dropped. MassSpecGym 1.5 has
    only [M+H]+ and [M+Na]+, so charge is +1 and polarity positive throughout.
    """
    df = pd.read_csv(path, sep="\t")
    mz, intensity, count = [], [], []
    for m, i in zip(df.mzs, df.intensities):
        m, i = np.asarray(m.split(","), np.float32), np.asarray(i.split(","), np.float32)
        keep = i > 0
        mz.append(m[keep]), intensity.append(i[keep]), count.append(int(keep.sum()))
    ce = df.collision_energy.to_numpy(np.float32)
    instrument = df.instrument_type.fillna("Unknown")
    return dict(
        identifier=df.identifier.to_numpy(str), smiles=df.smiles.to_numpy(str),
        inchikey14=df.inchikey.to_numpy(str), fold=df.fold.to_numpy(str),
        offsets=np.concatenate([[0], np.cumsum(count)]).astype(np.int64),
        mz=np.concatenate(mz), intensity=np.concatenate(intensity),
        precursor_mz=df.precursor_mz.to_numpy(np.float32),
        adduct=np.array([ADDUCTS.index(a) if a in ADDUCTS else 0 for a in df.adduct], np.int64),
        charge=np.ones(len(df), np.int64),
        polarity=np.full(len(df), POLARITIES.index("positive"), np.int64),
        instrument=np.array([INSTRUMENTS.index(x) if x in INSTRUMENTS else INSTRUMENTS.index("Unknown")
                             for x in instrument], np.int64),
        ce=np.nan_to_num(ce, nan=0.0), ce_valid=np.isfinite(ce),
    )


def pad(offsets, mz, intensity, rows):
    """Padded [len(rows), max peaks] arrays and mask from CSR peaks."""
    rows = np.asarray(rows)
    start, n = offsets[rows], offsets[rows + 1] - offsets[rows]
    width = max(int(n.max(initial=0)), 1)
    valid = np.arange(width)[None] < n[:, None]
    index = np.where(valid, start[:, None] + np.arange(width)[None], 0)
    return np.where(valid, mz[index], 0).astype(np.float32), np.where(valid, intensity[index], 0).astype(np.float32), valid


def select_peaks(mz, intensity, mask, k, augment=None, generator=None):
    """Augment (training only), keep the top-k peaks by intensity, sort by m/z, divide by the base peak."""
    if augment:
        def rand(f):
            return f(mz.shape, device=mz.device, generator=generator)
        keep = mask & (rand(torch.rand) >= augment["peak_dropout"])
        lost = mask.any(1) & ~keep.any(1)  # at least one peak survives: a random one
        keep[lost, (rand(torch.rand) * mask)[lost].argmax(1)] = True
        mask = keep
        intensity = (intensity * torch.exp(augment["intensity_sigma"] * rand(torch.randn))).clamp_min(1e-8)
        mz = mz * (1 + augment["mz_ppm"] * 1e-6 * rand(torch.randn))
    order = intensity.masked_fill(~mask, -1.0).sort(dim=1, descending=True, stable=True).indices[:, :k]
    mz, intensity, mask = (t.gather(1, order) for t in (mz, intensity, mask))
    order = mz.masked_fill(~mask, float("inf")).sort(dim=1, stable=True).indices
    mz, intensity, mask = (t.gather(1, order) for t in (mz, intensity, mask))
    base = intensity.masked_fill(~mask, 0.0).amax(1, keepdim=True).clamp_min(1e-8)
    return mz.masked_fill(~mask, 0.0), (intensity / base).masked_fill(~mask, 0.0), mask


class SpectraTable:
    """Spectra as padded tensors on one device, one row per spectrum."""

    def __init__(self, data: dict, rows=None, device="cpu"):
        rows = np.arange(len(data["precursor_mz"])) if rows is None else np.asarray(rows)
        mz, intensity, valid = pad(data["offsets"], data["mz"], data["intensity"], rows)
        self.mz, self.intensity, self.mask = (torch.from_numpy(a).to(device) for a in (mz, intensity, valid))
        self.meta = {k: torch.from_numpy(np.asarray(data[k])[rows]).to(device) for k in META}

    def __len__(self):
        return len(self.mask)

    def batch(self, index, max_peaks, augment=None, generator=None) -> dict:
        index = torch.as_tensor(index, device=self.mask.device)
        mz, intensity, mask = select_peaks(self.mz[index], self.intensity[index], self.mask[index],
                                           max_peaks, augment, generator)
        return dict(mz=mz, intensity=intensity, mask=mask, **{k: v[index] for k, v in self.meta.items()})
