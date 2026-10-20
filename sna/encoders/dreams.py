"""The pretrained DreaMS spectrum encoder (Bushuiev et al., 2025).

Only the released architecture is ported: fixed Fourier m/z features, a peak MLP and a Fourier MLP
stem, 7 pre-norm Transformer blocks with the parameter-free Graphormer m/z bias, a final LayerNorm.
Module names follow the checkpoints, so loading is a prefix strip and a strict ``load_state_dict``.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

DIM, DEPTH, HEADS, MLP_DIM = 1024, 7, 8, 4096
PEAK_DIM, FOURIER_DIM, FOURIER_HIDDEN = 44, 980, 512
MAX_MZ, MIN_RESOLUTION = 1000.0, 1e-4
DROPOUT = 0.1
PRECURSOR_INTENSITY = 1.1
# Checkpoint options that select the code path ported here; a key missing from the checkpoint means this value.
RELEASED = dict(
    max_mz=MAX_MZ, hidden_dim=DIM, depth=DEPTH, num_heads=HEADS, mlp_dim=MLP_DIM, peak_branch_dim=PEAK_DIM,
    fourier_branch_dim=FOURIER_DIM, peak_mlp_depth=1, fourier_mlp_depth=5, fourier_mlp_hidden_dim=FOURIER_HIDDEN,
    dropout=DROPOUT, attention_dropout=DROPOUT, residual_dropout=DROPOUT, feedforward_dropout=DROPOUT,
    pre_norm=True, normalization="layernorm", transformer_bias=False, stem_bias=True,
    graphormer_mz_differences=True, graphormer_parametrized=False, padding_mask_mode="reference-query",
    fourier_strategy="lin_float_int", fourier_min_resolution=MIN_RESOLUTION, fourier_trainable=False,
)


class MLP(nn.Module):
    """Linear layers, each followed by dropout (not the last) and a ReLU (also the last)."""

    def __init__(self, dims):
        super().__init__()
        layers = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            layers += [nn.Linear(a, b)] + [nn.Dropout(DROPOUT)] * (i < len(dims) - 2) + [nn.ReLU()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FourierFeatures(nn.Module):
    """cos and sin of 2 pi mz / period for periods 2e-4, 4e-4, .., 0.9998 Da and 2, 3, .., 999 Da."""

    def __init__(self):
        super().__init__()
        periods = [MIN_RESOLUTION * i for i in range(2, math.ceil(1 / MIN_RESOLUTION), 2)]
        periods += list(range(2, math.ceil(MAX_MZ)))
        self.register_buffer("frequencies", torch.tensor([1.0 / p for p in periods], dtype=torch.float32))

    def forward(self, mz):
        # float32 in this op order, as trained: phases reach 3e7 rad
        phase = 2.0 * torch.pi * mz.unsqueeze(-1) * self.frequencies
        return torch.cat([phase.cos(), phase.sin()], -1)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(4 * DIM, DIM) * (0.4 / DIM) ** 0.5)  # q, k, v, out; no biases

    def forward(self, x, bias, query_pad):
        B, T, _ = x.shape
        q, k, v = F.linear(x, self.weights[:3 * DIM]).view(B, T, 3, HEADS, -1).permute(2, 0, 3, 1, 4)
        # DreaMS masks padded query rows to a uniform softmax: a zero query and a zero bias row do the same.
        q = q.masked_fill(query_pad, 0.0)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=DROPOUT if self.training else 0.0)
        return F.linear(out.transpose(1, 2).reshape(B, T, DIM), self.weights[3 * DIM:])


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_projection = nn.Linear(DIM, MLP_DIM, bias=False)
        self.output_projection = nn.Linear(MLP_DIM, DIM, bias=False)

    def forward(self, x):
        return self.output_projection(F.dropout(F.relu(self.input_projection(x)), DROPOUT, self.training))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm, self.feedforward_norm = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.attention, self.feedforward = Attention(), FeedForward()

    def forward(self, x, bias, query_pad):
        x = x + F.dropout(self.attention(self.attention_norm(x), bias, query_pad), DROPOUT, self.training)
        return x + F.dropout(self.feedforward(self.feedforward_norm(x)), DROPOUT, self.training)


class DreaMSEncoder(nn.Module):
    """Spectrum batch (sna.spectra) -> (global [B, D], tokens [B, 1+N, D], token_mask [B, 1+N]).

    Token 0 is the precursor (precursor_mz, intensity 1.1); global is its final-norm output.
    """

    dim = DIM

    def __init__(self, max_peaks: int):
        super().__init__()
        self.max_peaks = max_peaks
        self.peak_encoder = MLP([2, PEAK_DIM])
        self.fourier_features = FourierFeatures()
        self.fourier_encoder = MLP([2 * len(self.fourier_features.frequencies)] + [FOURIER_HIDDEN] * 4 + [FOURIER_DIM])
        self.blocks = nn.ModuleList(Block() for _ in range(DEPTH))
        self.final_norm = nn.LayerNorm(DIM)

    def forward(self, batch):
        mz, intensity, mask = batch["mz"], batch["intensity"], batch["mask"]
        B, N = mask.shape
        assert N <= self.max_peaks, f"{N} peaks, the encoder takes at most {self.max_peaks}"
        # Released DreaMS masks padded queries but not padded keys, so outputs depend on the slot count: use max_peaks.
        pad = (0, self.max_peaks - N)
        mz = torch.cat([batch["precursor_mz"][:, None], F.pad(mz, pad)], 1)
        intensity = torch.cat([intensity.new_full((B, 1), PRECURSOR_INTENSITY), F.pad(intensity, pad)], 1)
        valid = torch.cat([mask.new_ones(B, 1), F.pad(mask, pad)], 1)

        fourier = self.fourier_encoder(self.fourier_features(mz))
        x = torch.cat([self.peak_encoder(torch.stack([mz / MAX_MZ, intensity], -1)), fourier], -1)
        x = F.dropout(x, DROPOUT, self.training)
        # Graphormer bias S_query - S_key (S = summed Fourier features), shared by all heads and layers;
        # S_query is constant along a softmax row, so in effect it is a per-key bias.
        s = fourier.float().sum(-1)
        bias = (s[:, :, None] - s[:, None, :]).masked_fill(~valid[:, :, None], 0.0)[:, None]
        query_pad = ~valid[:, None, :, None]
        for block in self.blocks:
            x = block(x, bias, query_pad)
        x = self.final_norm(x)
        return x[:, 0], x[:, :1 + N], valid[:, :1 + N]


def load_dreams(path) -> DreaMSEncoder:
    """Encoder of a DreaMS checkpoint (SSL or contrastive stage); every other head is dropped."""
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    config = checkpoint["hyper_parameters"]["model_config"]
    other = {k: config[k] for k, v in RELEASED.items() if config.get(k, v) != v}
    if other:
        raise ValueError(f"{path} is not the released DreaMS architecture: {other}")
    encoder = DreaMSEncoder(config["max_fragments"])
    prefix = "model.encoder."
    encoder.load_state_dict({k[len(prefix):]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)})
    return encoder
