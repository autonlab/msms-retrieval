"""Pretrained MS-JEPA spectrum encoder (the online encoder, film_ion conditioning).

Token sequence: [CLS], [REG], precursor, fragments. The ion condition (adduct,
charge, polarity) scales every block's two residual updates per channel (FiLM
gains on LayerScale). Variant "A": global = h_mol, the K=1 set aggregation of
the [CLS] output with the full source condition (ion + collision energy +
instrument). Variant "B": global = z_spec, the [CLS] output; B builds nothing
its forward does not use.

No peak is masked downstream, so the masked-pretraining inputs (mask vectors,
rank projection) are not built; their checkpoint keys are skipped by name.
Every other module keeps its checkpoint name.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from sna.spectra import ADDUCTS, INSTRUMENTS, POLARITIES

# The architecture every supported checkpoint was trained with. A checkpoint
# stores only the fields that existed when it was written; a missing field
# means the value below. The Fourier banks are not in the state dict, so a
# wrong value here would load silently.
PINNED = dict(
    register_tokens=1, encoder_metadata_mode="film_ion", aggregator_source_condition=True,
    hmol_explicit_condition_paths=True, attention_position_mode="pair_bias",
    mass_coordinate_mode="absolute_and_neutral_loss", pair_mass_min_period_da=1e-3,
    pair_mass_max_period_da=1e3, pair_intensity_features=True, pair_rank_features=True,
    mass_low_period_count=1000, mass_high_period_count=5000, mass_period_floor_da=None,
    mass_period_ceiling_da=None, logit_temperature=False, attention_differential=False,
    qk_normalization=False, pair_complementary_features=False,
)
# Token bank: periods 1..1000 Da and (2k-1)*0.1 mDa, k=1..5000. Pair bank: 16
# periods log-spaced over 1 mDa..1 kDa. float64, because the phase reduction
# below is only as exact as the frequency it multiplies by.
MASS_FREQUENCIES = 1.0 / torch.cat([torch.arange(1, 1001, dtype=torch.float64),
                                    (2 * torch.arange(1, 5001, dtype=torch.float64) - 1) * 1e-4])
PAIR_FREQUENCIES = 1.0 / torch.logspace(-3, 3, 16, dtype=torch.float64)
# Checkpoint keys with no counterpart here: masked-pretraining inputs, and for
# variant B the acquisition condition, which only the aggregator reads.
MASKING_KEYS = tuple(f"online_encoder.peak_encoder.{k}" for k in
                     ("absolute_mass_mask", "neutral_loss_mask", "intensity_mask", "rank_projection."))
ACQUISITION_KEYS = tuple(f"condition_encoder.{k}" for k in
                         ("instrument.", "ce_fourier_frequencies", "ce_net.", "acquisition_projection.",
                          "full_projection."))


def _phases(values: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
    """2*pi*frac(values * frequencies) in float32, reduced in float64: a float32
    phase of 10^7 cycles (1 kDa at a 0.1 mDa period) has no precision left."""
    flat = values.reshape(-1, 1).double()
    frequencies = frequencies.to(values.device)
    out = torch.empty(len(flat), len(frequencies), device=values.device)
    step = max(1, (1 << 23) // len(frequencies))  # bounds the float64 temporary
    for i in range(0, len(flat), step):
        out[i:i + step] = torch.remainder(flat[i:i + step] * frequencies, 1.0).mul_(2 * math.pi)
    return out.view(*values.shape, -1)


def _attention(qkv: torch.Tensor, heads: int, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Multi-head SDPA on a fused projection [B, T, 3D] -> [B, T, D]."""
    b, t, _ = qkv.shape
    q, k, v = (x.transpose(1, 2) for x in qkv.view(b, t, 3, heads, -1).unbind(2))
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(b, t, -1)


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, rate: float):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if not self.training or self.rate == 0.0:
            return x
        keep = x.new_empty(len(x), *(1,) * (x.ndim - 1)).bernoulli_(1.0 - self.rate)
        return x * keep / (1.0 - self.rate)


class ConditionEncoder(nn.Module):
    """Ion condition -> per-block FiLM gains; with ``acquisition`` also the full source condition."""

    def __init__(self, cfg: dict, acquisition: bool):
        super().__init__()
        d, f = cfg["condition_dim"], max(16, cfg["condition_dim"] // 4)
        self.depth, self.hidden, self.max_charge = cfg["depth"], cfg["hidden_dim"], cfg["max_charge"]
        self.condition_dropout = cfg["condition_dropout"]
        self.adduct = nn.Embedding(len(cfg["adduct_vocab"]), f)
        self.charge = nn.Embedding(2 * self.max_charge + 2, f)
        self.polarity = nn.Embedding(len(cfg["polarity_vocab"]), f)
        self.ion_projection = nn.Sequential(nn.Linear(3 * f, d), nn.LayerNorm(d), nn.SiLU())
        self.to_film_gain = nn.Linear(d, self.depth * 2 * self.hidden)
        nn.init.zeros_(self.to_film_gain.weight)  # every gain starts at exactly 1
        nn.init.zeros_(self.to_film_gain.bias)
        if acquisition:
            self.instrument = nn.Embedding(len(cfg["instrument_vocab"]), f)
            self.register_buffer("ce_fourier_frequencies", torch.logspace(-2, 1, 16))
            self.ce_net = nn.Sequential(nn.Linear(33, 2 * f), nn.SiLU(), nn.Linear(2 * f, f))
            self.acquisition_projection = nn.Sequential(nn.Linear(2 * f, d), nn.LayerNorm(d), nn.SiLU())
            self.full_projection = nn.Sequential(nn.Linear(2 * d, d), nn.LayerNorm(d), nn.SiLU())

    def ion(self, batch):
        charge = batch["charge"]
        charge = torch.where(charge.abs() <= self.max_charge, charge + self.max_charge + 1, 0)  # 0: out of range
        return self.ion_projection(torch.cat(
            [self.adduct(batch["adduct"]), self.charge(charge), self.polarity(batch["polarity"])], -1))

    def gains(self, ion):
        """[B, depth, 2, D]; in training the whole condition drops to the identity gain."""
        gains = 1.0 + self.to_film_gain(ion).view(len(ion), self.depth, 2, self.hidden)
        if self.training and self.condition_dropout > 0:
            keep = torch.rand(len(ion), 1, 1, 1, device=ion.device) >= self.condition_dropout
            gains = torch.where(keep, gains, 1.0)
        return gains

    def full(self, ion, batch):
        ce = batch["ce"].float()
        valid = batch["ce_valid"].bool() & torch.isfinite(ce) & (ce >= 0)
        phase = torch.where(valid, ce, 0.0)[:, None] * self.ce_fourier_frequencies
        ce = self.ce_net(torch.cat([phase.sin(), phase.cos(), valid[:, None].float()], -1))
        acquisition = self.acquisition_projection(torch.cat([ce, self.instrument(batch["instrument"])], -1))
        return self.full_projection(torch.cat([ion, acquisition], -1))


class FourierMassEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12000, 512, bias=False), nn.GELU(), nn.Linear(512, 512, bias=False),
                                 nn.GELU(), nn.Linear(512, 512, bias=False), nn.GELU(),
                                 nn.Linear(512, 256, bias=False))

    def forward(self, masses):
        phases = _phases(masses, MASS_FREQUENCIES)
        return self.net(torch.cat([phases.sin(), phases.cos()], -1))


class PeakEncoder(nn.Module):
    """Token content: m/z and neutral loss through one Fourier MLP, plus four intensity features."""

    def __init__(self, dim: int):
        super().__init__()
        self.mass_encoder = FourierMassEncoder()
        self.intensity_encoder = nn.Sequential(nn.Linear(4, 128, bias=False), nn.GELU(),
                                               nn.Linear(128, 128, bias=False))
        self.absolute_mass_norm, self.neutral_loss_norm = nn.LayerNorm(256), nn.LayerNorm(256)
        self.intensity_norm = nn.LayerNorm(128)
        self.fusion = nn.Linear(640, dim, bias=False)
        self.type_embedding = nn.Embedding(2, dim)
        self.mask_embedding = nn.Embedding(4, dim)  # pretraining mask types; downstream reads row 0 (visible)
        self.norm = nn.LayerNorm(dim)

    def forward(self, masses, intensities, is_precursor, valid):
        absolute, neutral = self.mass_encoder(torch.cat([masses, masses[:, :1] - masses])).chunk(2)
        fragments = (intensities * (valid & ~is_precursor)).sum(1, keepdim=True).clamp_min(1e-8)
        safe = intensities.clamp_min(1e-8)
        features = torch.stack([intensities, safe.sqrt(), torch.log1p(1000.0 * safe) / math.log(1001.0),
                                intensities / fragments], -1)
        x = self.fusion(torch.cat([self.absolute_mass_norm(absolute), self.neutral_loss_norm(neutral),
                                   self.intensity_norm(self.intensity_encoder(features))], -1))
        x = x + self.type_embedding(is_precursor.long()) + self.mask_embedding.weight[0]
        return self.norm(x).masked_fill(~valid[..., None], 0.0)


class AttentionBiasBranch(nn.Module):
    def __init__(self, inputs: int, heads: int):
        super().__init__()
        self.in_proj, self.out_proj = nn.Linear(inputs, 32), nn.Linear(32, heads)
        self.head_scale = nn.Parameter(torch.ones(heads))

    def forward(self, x):
        x = self.out_proj(F.gelu(self.in_proj(x)))
        return x * self.head_scale.to(x.dtype)


class AdditiveAttentionBias(nn.Module):
    """Per-head bias [B, T, T, H] from the mass-difference, intensity and flag descriptors."""

    def __init__(self, heads: int):
        super().__init__()
        self.mass_difference = AttentionBiasBranch(33, heads)
        self.intensity = AttentionBiasBranch(2, heads)
        self.flags = AttentionBiasBranch(22, heads)

    def forward(self, mass, intensity, flags):
        return self.mass_difference(mass) + self.flags(flags) + self.intensity(intensity)


def _pair_descriptor(masses, intensities, is_precursor, measured, valid):
    """Bias inputs for every token pair (i, j). ``measured``: the precursor and real fragments."""
    t = masses.shape[1]
    pair = (measured[:, :, None] & measured[:, None, :])[..., None]
    dm = masses[:, :, None] - masses[:, None, :]
    phases = _phases(dm, PAIR_FREQUENCIES)
    mass = torch.where(pair, torch.cat([phases.sin(), phases.cos(), dm[..., None] / 1000.0], -1), 0.0)
    log_i = intensities.clamp_min(1e-8).log()
    scale = -math.log(1e-8)
    ratio = ((log_i[:, :, None] - log_i[:, None, :]) / scale).clamp(-1.0, 1.0)
    mean = (0.5 * (log_i[:, :, None] + log_i[:, None, :]) / scale).clamp(-1.0, 1.0)
    intensity = torch.where(pair, torch.stack([ratio, mean], -1), 0.0)

    def rows(x):
        return x.float()[:, :, None].expand(-1, -1, t)

    def cols(x):
        return x.float()[:, None, :].expand(-1, t, -1)
    flags = torch.stack([rows(is_precursor), cols(is_precursor), rows(measured), cols(measured),
                         rows(measured), cols(measured)], -1)  # flags 3-4 / 5-6: mass / intensity visible
    flags = torch.where((valid[:, :, None] & valid[:, None, :])[..., None], flags, 0.0)
    # 16 rank features lead the flags; they are nonzero only for masked peaks.
    return mass, intensity, F.pad(flags, (16, 0))


class SetTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, drop_path: float):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        x = x + self.drop_path(self.proj(_attention(self.qkv(self.norm1(x)), self.heads)))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Block(SetTransformerBlock):
    """Encoder block: each residual update is scaled by LayerScale x FiLM gain before DropPath."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, drop_path: float, layer_scale: float):
        super().__init__(dim, heads, mlp_dim, drop_path)
        self.layer_scale_attention = nn.Parameter(torch.full((dim,), layer_scale))
        self.layer_scale_mlp = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x, mask, gain, valid):
        out = self.proj(_attention(self.qkv(self.norm1(x)), self.heads, mask))
        x = x + self.drop_path(out * (self.layer_scale_attention.to(out.dtype) * gain[:, 0, None].to(out.dtype)))
        out = self.mlp(self.norm2(x))
        x = x + self.drop_path(out * (self.layer_scale_mlp.to(out.dtype) * gain[:, 1, None].to(out.dtype)))
        return x.masked_fill(~valid[..., None], 0.0)


class DinoMSEncoder(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        dim, heads, depth = cfg["hidden_dim"], cfg["num_heads"], cfg["depth"]
        self.peak_encoder = PeakEncoder(dim)
        rates = torch.linspace(0.0, cfg["drop_path_rate"], depth).tolist()
        self.blocks = nn.ModuleList(Block(dim, heads, cfg["mlp_dim"], r, cfg["layer_scale_init"]) for r in rates)
        self.attention_biases = nn.ModuleList(AdditiveAttentionBias(heads) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)

    def forward(self, prefix, masses, intensities, valid, gains):
        """prefix [B, P, D]; masses, intensities, valid [B, T] with the precursor in column 0;
        gains [B, depth, 2, D]. Returns [B, P + T, D]."""
        is_precursor = torch.zeros_like(valid)
        is_precursor[:, 0] = True
        x = torch.cat([prefix, self.peak_encoder(masses, intensities, is_precursor, valid)], 1)

        def pad(values, fill):  # the prefix carries no measurement but is always attendable
            return torch.cat([values.new_full((len(values), prefix.shape[1]), fill), values], 1)
        measured, valid = pad(valid, False), pad(valid, True)
        descriptor = _pair_descriptor(pad(masses, 0.0), pad(intensities, 0.0), pad(is_precursor, False),
                                      measured, valid)
        for block, bias, gain in zip(self.blocks, self.attention_biases, gains.unbind(1)):
            mask = bias(*descriptor).permute(0, 3, 1, 2).masked_fill(~valid[:, None, None, :], float("-inf"))
            x = block(x, mask, gain, valid)
        return self.norm(x).masked_fill(~valid[..., None], 0.0)


class SetAggregator(nn.Module):
    """K=1: a learned token attends to one (spectrum, source condition) token."""

    def __init__(self, cfg: dict):
        super().__init__()
        dim = cfg["hidden_dim"]
        self.agg_token = nn.Parameter(torch.empty(1, 1, dim))
        nn.init.normal_(self.agg_token, std=0.02)
        self.source_projection = nn.Linear(dim + cfg["condition_dim"], dim)
        rates = torch.linspace(0, cfg["drop_path_rate"], cfg["aggregator_depth"]).tolist()
        self.blocks = nn.ModuleList(SetTransformerBlock(dim, cfg["aggregator_heads"], cfg["aggregator_mlp_dim"], r)
                                    for r in rates)
        self.norm = nn.LayerNorm(dim)

    def forward(self, spectrum, condition):
        source = self.source_projection(torch.cat([spectrum, condition], -1))
        x = torch.stack([self.agg_token[0, 0].expand_as(source), source], 1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x[:, 0])


class MSJEPAEncoder(nn.Module):
    """forward(batch) -> (global [B, D], tokens [B, 1+N, D], token_mask [B, 1+N]).

    tokens: final outputs at the precursor, then the N fragment positions.
    """

    def __init__(self, cfg: dict, variant: str):
        super().__init__()
        if variant not in ("A", "B"):
            raise ValueError(f"variant must be 'A' or 'B', got {variant!r}")
        self.variant, self.dim, self.max_peaks = variant, cfg["hidden_dim"], cfg["max_fragments"]
        self.condition_encoder = ConditionEncoder(cfg, acquisition=variant == "A")
        self.online_encoder = DinoMSEncoder(cfg)
        self.prefix_tokens = nn.Parameter(torch.empty(1, 2, self.dim))  # [CLS], [REG]
        nn.init.normal_(self.prefix_tokens, std=0.02)
        if variant == "A":
            self.aggregator = SetAggregator(cfg)

    def forward(self, batch):
        b, n = batch["mz"].shape
        if n > self.max_peaks:
            raise ValueError(f"batch has {n} peak slots; this encoder reads at most {self.max_peaks}")
        ion = self.condition_encoder.ion(batch)
        # Precursor token: its m/z with intensity 1.1, above any base-peak-normalized fragment.
        masses = torch.cat([batch["precursor_mz"][:, None], batch["mz"]], 1)
        intensities = torch.cat([batch["intensity"].new_full((b, 1), 1.1), batch["intensity"]], 1)
        token_mask = torch.cat([batch["mask"].new_ones(b, 1), batch["mask"]], 1)
        out = self.online_encoder(self.prefix_tokens.expand(b, -1, -1), masses, intensities, token_mask,
                                  self.condition_encoder.gains(ion))
        z_spec, tokens = out[:, 0], out[:, self.prefix_tokens.shape[1]:]
        if self.variant == "B":
            return z_spec, tokens, token_mask
        return self.aggregator(z_spec, self.condition_encoder.full(ion, batch)), tokens, token_mask


def _checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    cfg = dict(checkpoint["hyper_parameters"]["model_config"])
    for key, value in PINNED.items():
        if cfg.setdefault(key, value) != value:
            raise ValueError(f"{path}: {key}={cfg[key]!r}; this port supports only {value!r}")
    vocab = (tuple(cfg["adduct_vocab"][1:]), tuple(cfg["polarity_vocab"][1:]), tuple(cfg["instrument_vocab"]))
    if vocab != (ADDUCTS[1:], POLARITIES[1:], INSTRUMENTS):
        raise ValueError(f"{path}: condition vocabularies {vocab} differ from sna.spectra's")
    return cfg, checkpoint["state_dict"]


def fresh_msjepa(path, variant="A") -> MSJEPAEncoder:
    """The checkpoint's architecture with its training-time random initialisation."""
    return MSJEPAEncoder(_checkpoint(path)[0], variant)


def load_msjepa(path, variant="A") -> MSJEPAEncoder:
    """Pretrained online encoder, condition encoder, prefix tokens and (A) aggregator, in eval mode.

    The EMA target, predictor and auxiliary heads in the checkpoint are never read.
    """
    cfg, state = _checkpoint(path)
    model = MSJEPAEncoder(cfg, variant)
    own = ("prefix_tokens", *(name for name, _ in model.named_children()))
    skip = MASKING_KEYS + (ACQUISITION_KEYS if variant == "B" else ())
    model.load_state_dict({k[6:]: v for k, v in state.items() if k.startswith("model.")
                           and k[6:].split(".")[0] in own and not k[6:].startswith(skip)}, strict=True)
    return model.eval()
