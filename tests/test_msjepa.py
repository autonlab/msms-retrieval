"""Fast checks of the MS-JEPA encoder on the first 512 MassSpecGym spectra."""
import io
import itertools

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from sna.encoders.msjepa import fresh_msjepa, load_msjepa
from sna.paths import ENCODERS, MSG_TSV
from sna.spectra import SpectraTable, read_msg_tsv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def data():
    with open(MSG_TSV) as f:  # header + 512 spectra: both adducts, missing CE, Unknown instrument, 1 to 299 peaks
        return read_msg_tsv(io.StringIO("".join(itertools.islice(f, 513))))


@pytest.fixture(scope="module")
def ep139():
    return {v: load_msjepa(ENCODERS["msjepa"], v).to(DEVICE) for v in "AB"}


def batch_of(data, rows, max_peaks):
    return SpectraTable(data, rows, device=DEVICE).batch(torch.arange(len(rows)), max_peaks)


def cos(a, b):
    return F.cosine_similarity(a.double(), b.double(), dim=-1)


def test_variants_share_the_encoder(data, ep139):
    batch = batch_of(data, np.arange(64), 100)
    with torch.inference_mode():
        h, tokens, mask = ep139["A"](batch)
        z, tokens_b, mask_b = ep139["B"](batch)
    assert h.shape == z.shape == (64, 768) and tokens.shape == (64, 101, 768) and mask[:, 0].all()
    assert torch.equal(tokens, tokens_b) and torch.equal(mask, mask_b) and not tokens[~mask].any()
    assert not hasattr(ep139["B"], "aggregator") and not hasattr(ep139["B"].condition_encoder, "full_projection")


def test_padding_is_invisible(data, ep139):
    n = np.diff(data["offsets"])
    few, many = np.flatnonzero(n <= 20)[:32], np.flatnonzero(n > 128)[:1]
    with torch.inference_mode():
        h, tokens, mask = ep139["A"](batch_of(data, few, 100))  # N = most peaks in `few`
        h_wide, tokens_wide, _ = ep139["A"](batch_of(data, np.concatenate([few, many]), 100))  # N = 100
    width = tokens.shape[1]
    assert width <= 21 and tokens_wide.shape[1] == 101
    assert cos(h, h_wide[:32]).min() >= 0.99999 and cos(tokens[mask], tokens_wide[:32, :width][mask]).min() >= 0.99999


def test_train_mode_keeps_the_regularisers(data):
    model = load_msjepa(ENCODERS["msjepa"], "A").to(DEVICE)
    batch = batch_of(data, np.arange(32), 100)
    with torch.no_grad():
        assert torch.equal(model(batch)[0], model(batch)[0])
        model.train()
        assert not torch.equal(model(batch)[0], model(batch)[0])  # drop path and condition dropout


@pytest.mark.parametrize("variant", "AB")
def test_every_parameter_gets_a_gradient(variant, data):
    model = fresh_msjepa(ENCODERS["msjepa"], variant).to(DEVICE).train()
    assert sum(p.numel() for p in model.parameters()) == sum(
        p.numel() for p in load_msjepa(ENCODERS["msjepa"], variant).parameters())
    with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
        g, tokens, mask = model(batch_of(data, np.arange(16), 100))
    g.float().square().mean().backward()  # the global vector alone must reach every parameter
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_rejects_more_peaks_than_it_was_trained_with(data, ep139):
    rows = np.flatnonzero(np.diff(data["offsets"]) > 128)[:2]
    with pytest.raises(ValueError):
        ep139["B"](batch_of(data, rows, 128))
