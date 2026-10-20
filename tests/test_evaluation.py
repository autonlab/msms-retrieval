from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sna.model import rank_candidates


def test_finite_ranks_keep_existing_tie_rule():
    scores = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.2, 0.1]])
    ranks, ties = rank_candidates(scores, 1)
    assert ranks.tolist() == [2, 1]
    assert ties.tolist() == [0, 1]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("column", [0, 1])
def test_nonfinite_positive_or_decoy_is_rejected(value, column):
    scores = torch.tensor([[0.1, 0.2]])
    scores[0, column] = value
    with pytest.raises(ValueError, match="Non-finite"):
        rank_candidates(scores, 0)


@pytest.mark.parametrize("entrypoint", ["aligner", "jestr"])
def test_evaluators_reject_nan_before_producing_results(monkeypatch, entrypoint):
    from sna import jestr, train

    msg = SimpleNamespace(fold=np.array([2]), spectra=lambda q: np.array([0]),
                          candidates=lambda q: (torch.tensor([0, 1]), 0))
    model = SimpleNamespace(eval=lambda: None, molecule=lambda h, mask: h,
                            spectrum=lambda x: torch.full((len(x), 2), torch.nan))
    if entrypoint == "aligner":
        monkeypatch.setattr(train, "spectrum_vectors", lambda *a: (
            torch.full((1, 1, 2), torch.nan), torch.ones(1, 1, dtype=torch.bool)))
        molecules = lambda rows: (torch.ones(2, 1, 2), torch.ones(2, 1, dtype=torch.bool))
        with pytest.raises(ValueError, match="Non-finite"):
            train.evaluate(model, msg, molecules, 2, 0, 1)
    else:
        monkeypatch.setattr(jestr, "encode_molecules", lambda *a: torch.ones(2, 2))
        with pytest.raises(ValueError, match="Non-finite"):
            jestr.evaluate(model, msg, None, torch.zeros(1, 2), 2)
