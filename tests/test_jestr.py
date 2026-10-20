import numpy as np
import torch

from sna.jestr import ATOM_DIM, ELEMENTS, MoleculeEncoder, contrastive, featurize


def test_featurize_acetic_acid():
    index, bits, mass, edges = featurize("CC(=O)O")
    assert index.shape == (4, 12) and len(edges) == 6  # three bonds, both directions
    assert [ELEMENTS[i] for i in index[:, 0]] == ["C", "C", "O", "O"]
    assert bits.tolist() == [1, 3, 2, 1]  # single; single + double; double; single
    assert np.allclose(mass, [0.12011, 0.12011, 0.15999, 0.15999], atol=1e-4)
    assert ATOM_DIM == 78


def test_unknown_element_and_no_bonds():
    index, bits, _, edges = featurize("[Pt]")
    assert index[0, 0] == len(ELEMENTS) and bits[0] == 0 and len(edges) == 0


def test_max_pool_is_per_graph():
    model = MoleculeEncoder().eval()
    x = torch.randn(5, ATOM_DIM)
    node = torch.arange(5)
    graph = torch.tensor([0, 0, 0, 1, 1])
    both = model(x, node, node, graph, 2)  # self-loops only
    first = model(x[:3], node[:3], node[:3], graph[:3], 1)
    assert torch.allclose(both[:1], first, atol=1e-5)


def test_contrastive_is_symmetric_and_masks_duplicates():
    s, m = torch.randn(4, 8), torch.randn(4, 8)
    key = torch.tensor([0, 1, 2, 3])
    assert torch.isclose(contrastive(s, m, key, 0.05), contrastive(m, s, key, 0.05))
    dup = torch.tensor([0, 0, 2, 3])
    m2 = m.clone()
    m2[1] = m[0]
    assert contrastive(s, m2, dup, 0.05) < contrastive(s, m2, key, 0.05) + 1e-6
