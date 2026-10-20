"""Frozen molecule towers and the pooled-embedding cache.

The towers are never trained. They stay in eval mode even inside a training
step: MoLFormer redraws its random attention features on every forward in
train mode (or with deterministic_eval=False), which makes its embedding a
random function of the SMILES.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from sna import paths

# Structural tokens, masked out of the fine-grained molecule side: BOS, EOS, PAD, MASK. <unk> is kept,
# because it stands for a real (out-of-vocabulary) atom; masking it would leave e.g. [195TlH2] with no token.
SPECIAL = {"molformer": (0, 1, 2, 3), "chemberta": (0, 1, 2, 4)}
PAD = {"molformer": 2, "chemberta": 1}


class TokenTable:
    """Pre-tokenised SMILES (CSR, one row per molecule) on one device; rows come back padded."""

    def __init__(self, path, device="cpu"):
        z = np.load(path)
        self.ids = torch.from_numpy(z["ids"].astype(np.int32)).to(device)
        self.offsets = torch.from_numpy(z["offsets"].astype(np.int64)).to(device)

    def __len__(self):
        return len(self.offsets) - 1

    def __call__(self, rows, pad):
        rows = torch.as_tensor(rows, device=self.offsets.device)
        start, n = self.offsets[rows], self.offsets[rows + 1] - self.offsets[rows]
        position = torch.arange(int(n.max()), device=rows.device)
        valid = position[None] < n[:, None]
        index = (start[:, None] + position[None]).clamp_max(len(self.ids) - 1)
        return torch.where(valid, self.ids[index], pad).long(), valid


class Tower(nn.Module):
    """A frozen pretrained SMILES encoder (pinned Hugging Face snapshot)."""

    def __init__(self, name):
        super().__init__()
        repo, revision = paths.TOWERS[name]
        extra = dict(trust_remote_code=True, deterministic_eval=True) if name == "molformer" else {}
        self.model = AutoModel.from_pretrained(repo, revision=revision, **extra).requires_grad_(False).eval()
        self.name, self.pad, self.dim = name, PAD[name], self.model.config.hidden_size
        self.register_buffer("special", torch.tensor(SPECIAL[name]), persistent=False)

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, ids, valid, pooled, chunk=2048):
        """Pooled [M, 1, dim] or token states [M, L, dim] (structural tokens masked), with a mask.

        Pooled is MoLFormer's own pooler_output (mean over every attended token,
        special tokens included) and ChemBERTa's <s> state. Rows are encoded in
        length-sorted chunks so padding stays small.
        """
        lengths = valid.sum(1)
        order = lengths.argsort()
        out = torch.zeros(len(ids), 1 if pooled else ids.shape[1], self.dim, device=ids.device)
        for i in range(0, len(ids), chunk):
            rows = order[i:i + chunk]
            width = int(lengths[rows].max())
            x, m = ids[rows, :width], valid[rows, :width]
            with torch.autocast("cuda", torch.bfloat16, enabled=ids.is_cuda):
                h = self.model(input_ids=x, attention_mask=m.long()).last_hidden_state.float()
            if pooled:
                h = ((h * m[..., None]).sum(1) / m.sum(1, keepdim=True) if self.name == "molformer" else h[:, 0])[:, None]
            out[rows, :h.shape[1]] = h
        if pooled:
            return out, torch.ones(len(ids), 1, dtype=torch.bool, device=ids.device)
        return out, valid & ~torch.isin(ids, self.special)


def build_cache(name, batch=8192):
    """Pooled embeddings of every molecule in prep/molecules.npz -> prep/pooled_<name>.npy (float16)."""
    tokens = TokenTable(paths.PREP / f"tokens_{name}.npz", "cuda")
    tower = Tower(name).cuda()
    out = paths.PREP / f"pooled_{name}.npy"
    tmp = out.with_suffix(".tmp.npy")
    cache = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(len(tokens), tower.dim))
    started = time.time()
    for i in range(0, len(tokens), batch):
        rows = torch.arange(i, min(i + batch, len(tokens)), device="cuda")
        cache[i:i + len(rows)] = tower(*tokens(rows, tower.pad), pooled=True)[0][:, 0].half().cpu().numpy()
        if i // batch % 100 == 0:
            print(f"[{name}] {i + len(rows):,}/{len(tokens):,} {time.time() - started:.0f}s", flush=True)
    cache.flush()
    assert np.isfinite(cache[:: max(1, len(tokens) // 100_000)]).all()
    del cache
    os.replace(tmp, out)
    print(f"[{name}] wrote {out} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=build_cache.__doc__)
    parser.add_argument("tower", choices=sorted(paths.TOWERS))
    build_cache(parser.parse_args().tower)
