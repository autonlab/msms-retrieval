import torch
import torch.nn.functional as F

from sna.model import Conditions, candidate_loss, inbatch_loss, maxsim


def test_maxsim_one_vector_is_cosine():
    s, m = F.normalize(torch.randn(5, 1, 8), dim=-1), F.normalize(torch.randn(7, 1, 8), dim=-1)
    ones = lambda n: torch.ones(n, 1, dtype=torch.bool)
    assert torch.allclose(maxsim(s, ones(5), m, ones(7)), s[:, 0] @ m[:, 0].T, atol=1e-6)


def test_maxsim_masks_and_groups():
    s, m = torch.randn(3, 4, 8), torch.randn(3, 6, 5, 8)
    s_mask, m_mask = torch.rand(3, 4) > 0.3, torch.rand(3, 6, 5) > 0.3
    s_mask[:, 0] = m_mask[..., 0] = True
    got = maxsim(s, s_mask, m, m_mask)
    for b in range(3):
        for k in range(6):
            sim = s[b][s_mask[b]] @ m[b, k][m_mask[b, k]].T
            assert torch.allclose(got[b, k], sim.amax(1).mean(), atol=1e-5)
    # all-pairs form agrees with the grouped form on the diagonal blocks
    flat = maxsim(s, s_mask, m[0], m_mask[0])
    assert torch.allclose(flat[0], got[0], atol=1e-5)


def test_inbatch_loss_ignores_same_molecule():
    logits = torch.zeros(3, 3)
    logits[0, 1] = 50.0  # row 1 is the same molecule as row 0: must not count as a negative
    assert inbatch_loss(logits, torch.tensor([7, 7, 9])) < inbatch_loss(logits, torch.tensor([7, 8, 9]))


def test_candidate_loss_masks_padding():
    logits = torch.tensor([[1.0, 0.0, 100.0]])
    assert torch.isclose(candidate_loss(logits, torch.tensor([[True, True, False]]), torch.tensor([0])),
                         F.cross_entropy(logits[:, :2], torch.tensor([0])))


def test_conditions_missing_ce_is_one_vector():
    c = Conditions(("ce", "instrument"))
    batch = dict(ce=torch.tensor([0.0, 35.0, 0.0]), ce_valid=torch.tensor([False, True, False]),
                 instrument=torch.tensor([0, 0, 0]))
    out = c(batch)
    assert out.shape == (3, 256) and torch.equal(out[0], out[2]) and not torch.equal(out[0], out[1])


def test_molecule_adapter_skips_padding():
    from sna.model import Aligner
    encoder = torch.nn.Linear(1, 1)
    encoder.dim = 8
    model = Aligner(encoder, 6, frozen_encoder=True).eval()
    h, mask = torch.randn(3, 5, 6), torch.rand(3, 5) > 0.4
    out = model.molecule(h, mask)
    full = torch.nn.functional.normalize(model.molecule_adapter(h), dim=-1)
    assert torch.allclose(out[mask], full[mask], atol=1e-6) and (out[~mask] == 0).all()


def test_every_molecule_keeps_a_token():
    """An empty molecule scores -inf, and -inf / tau gives tau a NaN gradient."""
    import numpy as np
    from sna import paths
    from sna.towers import SPECIAL
    for table in ("tokens_molformer", "autonmol_tokens_molformer", "tokens_chemberta"):
        path = paths.PREP / f"{table}.npz"
        if table.startswith("autonmol_") and not path.exists():
            continue
        z = np.load(path)
        content = np.add.reduceat((~np.isin(z["ids"], SPECIAL[table.split("_")[-1]])).astype(np.int64), z["offsets"][:-1])
        assert (content[np.diff(z["offsets"]) > 0] > 0).all(), table


def test_tau_can_leave_the_floor():
    import math
    from sna.model import Aligner
    encoder = torch.nn.Linear(1, 1)
    encoder.dim = 8
    model = Aligner(encoder, 6, frozen_encoder=True)
    with torch.no_grad():
        model.log_tau.fill_(math.log(0.001))
    model.clamp_tau()
    assert math.isclose(model.tau.item(), 0.01, rel_tol=1e-6)
    (-model.tau).backward()  # a loss that wants a larger tau still gets a gradient at the floor
    assert model.log_tau.grad.item() < 0


def test_expected_score_is_the_mean_score():
    """Chunked, length-sorted bank averaging equals the plain mean of MaxSim over the bank; with one
    vector a side it equals the score against the mean vector (the global control's shortcut)."""
    from sna.train import expected_scores
    bank, m = torch.randn(150, 7, 8), torch.randn(5, 4, 8)
    bank_mask = torch.arange(7) < torch.randint(1, 8, (150, 1))  # padding at the end, as spectra have it
    m_mask = torch.rand(5, 4) > 0.3
    m_mask[:, 0] = True
    order = bank_mask.sum(1).argsort(stable=True)
    got = expected_scores(bank[order], bank_mask[order], m, m_mask)
    assert torch.allclose(got[0], maxsim(bank, bank_mask, m, m_mask).mean(0), atol=1e-5)
    one = torch.ones(150, 1, dtype=torch.bool)
    linear = expected_scores(bank[:, :1].mean(0, keepdim=True), one[:1], m[:, :1], one[:5])
    assert torch.allclose(linear[0], maxsim(bank[:, :1], one, m[:, :1], one[:5]).mean(0), atol=1e-5)
