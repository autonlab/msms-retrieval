"""Fast checks of the DreaMS encoder on the first 256 MassSpecGym spectra."""
import io
import itertools

import numpy as np
import pytest
import torch

from sna.encoders.dreams import load_dreams
from sna.paths import ENCODERS, MSG_TSV
from sna.spectra import AUGMENT, SpectraTable, read_msg_tsv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def data():
    with open(MSG_TSV) as f:  # header + 256 spectra
        return read_msg_tsv(io.StringIO("".join(itertools.islice(f, 257))))


def test_narrow_batch_is_padded_to_max_peaks(data):
    rows = np.nonzero(np.diff(data["offsets"]) <= 20)[0]  # N <= 20 < max_peaks: padding is visible to every query
    encoder = load_dreams(ENCODERS["dreams_contrastive"]).to(DEVICE).eval()
    with torch.inference_mode():
        _, tokens, mask = encoder(SpectraTable(data, rows, device=DEVICE).batch(torch.arange(len(rows)), encoder.max_peaks))
    width = 1 + int(np.diff(data["offsets"])[rows].max())
    assert tokens.shape == (len(rows), width, 1024) and mask.shape == (len(rows), width) and mask[:, 0].all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 autocast training needs CUDA")
def test_bf16_train_step_reaches_every_parameter(data):
    encoder = load_dreams(ENCODERS["dreams_contrastive"]).cuda().train()
    batch = SpectraTable(data, device="cuda").batch(torch.arange(16), encoder.max_peaks, AUGMENT,
                                                    torch.Generator(device="cuda").manual_seed(0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        g, tokens, mask = encoder(batch)
        again = encoder(batch)[0]
    (g.float().square().mean() + tokens.float()[mask].square().mean()).backward()
    assert not torch.equal(g, again)  # dropout is on
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.any() for p in encoder.parameters())
