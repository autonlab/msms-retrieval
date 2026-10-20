"""The aligner: a spectrum encoder and a frozen molecule tower, each followed by an
MSAlign adapter, scored by cosine (global) or MaxSim (fine-grained)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from sna.spectra import ADDUCTS, INSTRUMENTS


def rank_candidates(scores, positive):
    """Rank the positive candidate, rejecting non-finite scores before comparison."""
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite candidate scores during evaluation")
    target = scores[:, positive:positive + 1]
    return 1 + (scores > target).sum(1), (scores == target).sum(1) - 1


class Adapter(nn.Sequential):
    """MSAlign (Krzakala et al.) projection head."""

    def __init__(self, dim_in, hidden=2048, dim_out=1024, dropout=0.2):
        super().__init__(nn.Linear(dim_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
                         nn.Dropout(dropout), nn.Linear(hidden, dim_out))


class Conditions(nn.Module):
    """Acquisition conditions appended to the spectrum adapter's input (MSAlign's metadata path).

    Adduct and instrument are learned embeddings. Collision energy is MS-JEPA's code:
    sin/cos at 16 log-spaced frequencies (0.01-10 rad/eV) and a validity bit, so it is
    not clamped (simulated CE reaches 681 eV) and a missing CE maps to one learned vector.
    """

    def __init__(self, fields, dim=128):
        super().__init__()
        assert set(fields) <= {"adduct", "ce", "instrument"}, fields
        self.fields, self.dim = tuple(fields), dim * len(fields)
        if "adduct" in fields:
            self.adduct = nn.Embedding(len(ADDUCTS), dim)
        if "instrument" in fields:
            self.instrument = nn.Embedding(len(INSTRUMENTS), dim)
        if "ce" in fields:
            self.register_buffer("frequency", torch.logspace(-2, 1, 16), persistent=False)
            self.ce = nn.Sequential(nn.Linear(33, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, batch):
        out = []
        for field in self.fields:
            if field == "ce":
                valid = batch["ce_valid"].float()[:, None]
                phase = batch["ce"].float()[:, None] * self.frequency
                out.append(self.ce(torch.cat([phase.sin() * valid, phase.cos() * valid, valid], -1)))
            else:
                out.append(getattr(self, field)(batch[field]))
        return torch.cat(out, -1)


def maxsim(s, s_mask, m, m_mask):
    """Scores between spectrum vector sets s [B,S,D] and molecule vector sets m [C,T,D] -> [B,C],
    or, with grouped molecules m [B,K,T,D], -> [B,K]: each spectrum vector takes its best-matching
    molecule vector, averaged over the spectrum's vectors. One vector a side gives the cosine."""
    if m.dim() == 3:
        sim = torch.einsum("bsd,ctd->bcst", s, m).masked_fill(~m_mask[None, :, None, :], -torch.inf)
    else:
        sim = torch.einsum("bsd,bktd->bkst", s, m).masked_fill(~m_mask[:, :, None, :], -torch.inf)
    sim = torch.where(s_mask[:, None, :], sim.amax(-1), 0.0)
    return sim.sum(-1) / s_mask.sum(-1, keepdim=True)


class Aligner(nn.Module):
    def __init__(self, encoder, tower_dim, conditions=(), fine_grained=False, frozen_encoder=False):
        super().__init__()
        self.encoder = encoder.requires_grad_(not frozen_encoder)
        self.frozen_encoder, self.fine_grained = frozen_encoder, fine_grained
        self.conditions = Conditions(conditions) if conditions else None
        self.spectrum_adapter = Adapter(encoder.dim + (self.conditions.dim if self.conditions else 0))
        self.molecule_adapter = Adapter(tower_dim)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07)))

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_encoder:
            self.encoder.eval()
        return self

    @property
    def tau(self):
        return self.log_tau.exp()

    @torch.no_grad()
    def clamp_tau(self, floor=0.01):
        """CLIP's floor, applied to the parameter after each step (a clamp in the forward would zero
        the gradient at the floor, so tau could never come back up)."""
        self.log_tau.clamp_(min=math.log(floor))

    def adapter_input(self, batch):
        """[B,S,d] with mask [B,S]: the encoder's global vector and, fine-grained, its precursor
        and peak tokens too, each with the conditions appended."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.frozen_encoder):
            g, tokens, token_mask = self.encoder(batch)
        x, mask = g[:, None], torch.ones(len(g), 1, dtype=torch.bool, device=g.device)
        if self.fine_grained:
            x, mask = torch.cat([x, tokens], 1), torch.cat([mask, token_mask], 1)
        if self.conditions is not None:
            x = torch.cat([x, self.conditions(batch)[:, None].expand(-1, x.shape[1], -1).to(x.dtype)], -1)
        return x, mask

    def spectrum(self, x):
        return F.normalize(self.spectrum_adapter(x).float(), dim=-1)

    def molecule(self, h, mask):
        """Adapted molecule vectors; only valid positions go through the adapter (padding stays 0)."""
        out = torch.zeros(*mask.shape, self.molecule_adapter[-1].out_features, device=h.device)
        out[mask] = F.normalize(self.molecule_adapter(h[mask]).float(), dim=-1)
        return out

    def forward(self, batch, h, h_mask, groups=None):
        """Training logits: every spectrum against every molecule ([B,B], in-batch), or against
        its own `groups` candidates when h holds B*groups molecules ([B,K])."""
        x, s_mask = self.adapter_input(batch)
        s, m = self.spectrum(x), self.molecule(h, h_mask)
        if groups is not None:
            m, h_mask = m.view(len(s), groups, *m.shape[1:]), h_mask.view(len(s), groups, -1)
        with torch.autocast("cuda", enabled=False):  # scores in fp32
            return maxsim(s, s_mask, m, h_mask) / self.tau


def inbatch_loss(logits, key):
    """Spectrum-to-molecule InfoNCE over the batch; other rows of the same molecule are not negatives."""
    target = torch.arange(len(key), device=key.device)
    same = (key[:, None] == key[None]) & (target[:, None] != target[None])
    return F.cross_entropy(logits.masked_fill(same, -torch.inf), target)


def candidate_loss(logits, valid, positive):
    """InfoNCE over each query's own candidates (positive included in the denominator)."""
    return F.cross_entropy(logits.masked_fill(~valid, -torch.inf), positive)
