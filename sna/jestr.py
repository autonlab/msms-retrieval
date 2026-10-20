"""JESTR (Kalia et al., Bioinformatics 2025) on MassSpecGym, with and without its candidate regulariser.

    python -m sna.jestr CONFIG.json   # (sna.config.JESTR); normally `sna run jestr`, which writes it

The published model and recipe (github.com/HassounLab/JESTR1, params.yaml, train_contr.py):
  spectrum   peaks binned to 1 Da below 1,001 Da, intensities scaled to a base peak of 999 and summed
             per bin, log10(x + 1) / 3; MLP 1000 -> 1024 -> 1024 -> 512, ReLU, dropout 0.4 (output too)
  molecule   dgllife GCN [64, 128, 256] (sum over neighbours and self, residual, ReLU, dropout 0.2,
             BatchNorm) on dgllife's 'full' atom features, max pool, 256 -> 1024 -> 512, dropout 0.4
  objective  CMC InfoNCE on cosine / 0.05, both directions summed; Adam 5e-4, batch 32, 1,000 epochs
  regulariser for the last 3% of epochs: 0.9 InfoNCE + 0.1 mean cos(spectrum, candidate) over 32
             candidates of the query's list, the list sorted by Morgan Tanimoto to the query and walked
             32 at a time each time the query is drawn
Here the regulariser walks the official MASS list (the one the benchmark
scores against; the query itself removed); both arms share epochs 0-969 (JESTR-NR and JESTR are the
same run until the regulariser starts), and each arm's last checkpoint is tested.
Deviations, for the benchmark: every molecule is encoded (JESTR drops molecules with elements outside
its list, '.' or one atom; unknown elements take the 'unknown' type here); an epoch draws one random
spectrum per train query, and same-molecule pairs in a batch are not negatives.
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from sna import config as configs, paths
from sna.config import JESTR
from sna.model import rank_candidates

# ---- molecular graphs (dgllife 'full' atom featuriser, JESTR's element list) ----------------------------

ELEMENTS = ("H", "C", "O", "N", "P", "S", "Cl", "F", "Br", "I", "Si", "B", "As", "Se")
BOND_TYPES = (Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC)
HYBRIDISATION = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3,
                 Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2)
CHIRAL = (Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
          Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_OTHER)
# One-hot fields after the mass and the bond-type multi-hot, in dgllife's order: (value of an atom, allowed values).
FIELDS = (
    (lambda a: a.GetSymbol() if a.GetSymbol() in ELEMENTS else None, (*ELEMENTS, None)),  # encode_unknown
    (lambda a: a.GetDegree(), range(11)),
    (lambda a: a.GetTotalDegree(), range(6)),
    (lambda a: a.GetValence(Chem.ValenceType.EXPLICIT), range(1, 7)),
    (lambda a: a.GetValence(Chem.ValenceType.IMPLICIT), range(7)),
    (lambda a: a.GetHybridization(), HYBRIDISATION),
    (lambda a: a.GetTotalNumHs(), range(5)),
    (lambda a: a.GetFormalCharge(), range(-2, 3)),
    (lambda a: a.GetNumRadicalElectrons(), range(5)),
    (lambda a: a.GetIsAromatic(), (False, True)),
    (lambda a: a.IsInRing(), (False, True)),
    (lambda a: a.GetChiralTag(), CHIRAL),
)
SIZES = tuple(len(allowed) for _, allowed in FIELDS)
ATOM_DIM = 1 + len(BOND_TYPES) + sum(SIZES)  # 78


def featurize(smiles):
    """Atom one-hot indices [n, 12] (255: value outside the allowed set), bond-type bits [n],
    mass x 0.01 [n], and bonds as local atom pairs [e, 2] in both directions."""
    mol = Chem.MolFromSmiles(smiles)
    atoms = list(mol.GetAtoms())
    fields = [tuple(allowed) for _, allowed in FIELDS]
    index = np.array([[fields[k].index(v) if (v := f(a)) in fields[k] else 255 for k, (f, _) in enumerate(FIELDS)]
                      for a in atoms], np.uint8).reshape(len(atoms), len(FIELDS))
    bits = np.array([sum(1 << BOND_TYPES.index(t) for t in {b.GetBondType() for b in a.GetBonds()} if t in BOND_TYPES)
                     for a in atoms], np.uint8)  # any() over the atom's bonds, per type
    mass = np.array([a.GetMass() * 0.01 for a in atoms], np.float32)
    pairs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    edges = np.array(pairs + [(j, i) for i, j in pairs], np.int16).reshape(-1, 2)
    return index, bits, mass, edges


class Graphs:
    """Every molecule of prep/molecules.npz as a graph, on one device (prep/graphs.npz, CSR)."""

    def __init__(self, device):
        z = np.load(paths.PREP / "graphs.npz")  # kept in their compact dtypes (5 GB for 7.2M molecules)
        self.atom_offsets, self.edge_offsets, self.index, self.bits, self.mass, self.edges = (
            torch.from_numpy(z[k]).to(device) for k in ("atom_offsets", "edge_offsets", "index", "bits", "mass", "edges"))
        self.columns = torch.tensor(np.cumsum((1 + len(BOND_TYPES), *SIZES[:-1])), device=device)
        self.bit_values = 1 << torch.arange(len(BOND_TYPES), device=device)

    def __call__(self, rows):
        """Node features [N, 78], edges (src, dst) incl. self-loops, graph of each node, graph count."""
        rows = torch.as_tensor(rows, device=self.atom_offsets.device)
        atoms, graph = self._gather(self.atom_offsets, rows)
        edges, edge_graph = self._gather(self.edge_offsets, rows)
        first = torch.zeros(len(rows) + 1, dtype=torch.long, device=rows.device)
        first[1:] = torch.bincount(graph, minlength=len(rows)).cumsum(0)
        x = torch.zeros(len(atoms), ATOM_DIM, device=rows.device)
        x[:, 0] = self.mass[atoms]
        x[:, 1:1 + len(BOND_TYPES)] = ((self.bits[atoms, None].long() & self.bit_values) > 0).float()
        index = self.index[atoms].long()
        x.scatter_add_(1, (self.columns + index).clamp_max(ATOM_DIM - 1), (index != 255).float())
        local = self.edges[edges].long() + first[edge_graph, None]
        node = torch.arange(len(atoms), device=rows.device)
        return x, torch.cat([local[:, 0], node]), torch.cat([local[:, 1], node]), graph, len(rows)

    @staticmethod
    def _gather(offsets, rows):
        start, n = offsets[rows], offsets[rows + 1] - offsets[rows]
        owner = torch.repeat_interleave(torch.arange(len(rows), device=rows.device), n)
        position = torch.arange(len(owner), device=rows.device) - torch.repeat_interleave(n.cumsum(0) - n, n)
        return start[owner] + position, owner


# ---- model -------------------------------------------------------------------------------------------

class GCNLayer(nn.Module):
    """dgllife GCNLayer: GraphConv(norm='none', ReLU) + ReLU(Linear) residual, dropout, BatchNorm."""

    def __init__(self, dim_in, dim_out, dropout=0.2):
        super().__init__()
        self.weight, self.residual = nn.Linear(dim_in, dim_out), nn.Linear(dim_in, dim_out)
        nn.init.xavier_uniform_(self.weight.weight), nn.init.zeros_(self.weight.bias)  # dgl GraphConv init
        self.dropout, self.norm = nn.Dropout(dropout), nn.BatchNorm1d(dim_out)

    def forward(self, h, src, dst):
        summed = torch.zeros_like(h).index_add_(0, dst, h[src])  # self-loops are among the edges
        return self.norm(self.dropout(F.relu(self.weight(summed)) + F.relu(self.residual(h))))


class MoleculeEncoder(nn.Module):
    def __init__(self, channels=(64, 128, 256), hidden=1024, dim=512, dropout=0.4):
        super().__init__()
        self.layers = nn.ModuleList(GCNLayer(a, b) for a, b in zip((ATOM_DIM, *channels[:-1]), channels))
        self.head = nn.Sequential(nn.Linear(channels[-1], hidden), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hidden, dim), nn.Dropout(dropout))

    def forward(self, x, src, dst, graph, n):
        for layer in self.layers:
            x = layer(x, src, dst)
        pooled = x.new_full((n, x.shape[1]), -torch.inf).scatter_reduce(0, graph[:, None].expand_as(x), x, "amax")
        return self.head(pooled)


class SpectrumEncoder(nn.Sequential):
    def __init__(self, bins=1000, hidden=1024, dim=512, dropout=0.4):
        super().__init__(nn.Linear(bins, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, dim), nn.Dropout(dropout))


def binned(msg, bins=1000):
    """JESTR's spectrum input for every MassSpecGym spectrum [S, 1000] (all peaks, not the encoder top-k)."""
    t = msg.table
    intensity = 999.0 * t.intensity / t.intensity.amax(1, keepdim=True).clamp_min(1e-12)
    index = torch.trunc(t.mz - 1.0).long()  # int((mz - 1) / 1 Da): mz in (0, 2) -> bin 0
    keep = t.mask & (t.mz < bins + 1)
    out = torch.zeros(len(t), bins, device=t.mz.device)
    out.scatter_add_(1, index.clamp(0, bins - 1), torch.where(keep, intensity, 0.0))
    return torch.log10(out + 1.0) / 3.0


def contrastive(z_spec, z_mol, key, temperature):
    """JESTR's CMC loss, spectrum->molecule plus molecule->spectrum; the positive stays in each denominator.
    Another row of the same molecule is not a negative."""
    logits = F.normalize(z_spec, dim=-1) @ F.normalize(z_mol, dim=-1).T / temperature
    target = torch.arange(len(key), device=key.device)
    logits = logits.masked_fill((key[:, None] == key[None]) & (target[:, None] != target[None]), -torch.inf)
    return F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target)


# ---- training and evaluation ---------------------------------------------------------------------------

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.spectrum, self.molecule = SpectrumEncoder(), MoleculeEncoder()


@torch.no_grad()
def encode_molecules(model, graphs, rows, chunk=8192):
    return torch.cat([F.normalize(model.molecule(*graphs(rows[i:i + chunk])), dim=-1) for i in range(0, len(rows), chunk)])


@torch.no_grad()
def evaluate(model, msg, graphs, spectra, fold, blind=False):
    """Same rank table as sna.train.evaluate. blind: every query scored as the mean normalized train
    spectrum vector, i.e. each candidate by its expected cosine over train spectra (exact: linear). JESTR
    scores unit-norm outputs with no conditions, so this is also its mean-embedding control."""
    model.eval()
    queries = np.nonzero(msg.fold == fold)[0]
    if blind:
        train = torch.from_numpy(np.nonzero(msg.fold[msg.query] == 0)[0]).to(spectra.device)
        mean = sum(F.normalize(model.spectrum(spectra[train[i:i + 8192]]), dim=-1).sum(0) for i in range(0, len(train), 8192))
        mean = mean / len(train)
    out = []
    for i in range(0, len(queries), 256):
        qs = queries[i:i + 256]
        lists = [msg.candidates(q) for q in qs]
        m = encode_molecules(model, graphs, torch.cat([c for c, _ in lists]))
        b = 0
        for q, (cands, positive) in zip(qs, lists):
            rows = msg.spectra(q)
            s = mean[None].expand(len(rows), -1) if blind else F.normalize(model.spectrum(spectra[torch.from_numpy(rows).to(spectra.device)]), dim=-1)
            scores = s @ m[b:b + len(cands)].T
            ranks, ties = rank_candidates(scores, positive)
            out.append(np.stack([rows, ranks.cpu().numpy(), ties.cpu().numpy(),
                                 np.full(len(rows), len(cands))], 1))
            b += len(cands)
    result = np.concatenate(out)
    return result[np.argsort(result[:, 0])]


def test(model, msg, graphs, spectra, out, trained, say):
    """Score the arm's last checkpoint (out/best.pt) on the test fold, with the spectrum-blind control."""
    from sna.train import metrics
    model.load_state_dict(torch.load(out / "best.pt", map_location=spectra.device)["state"])
    result, blind = (evaluate(model, msg, graphs, spectra, 2, b) for b in (False, True))
    test = dict(test=metrics(result), **trained, blind=metrics(blind), blind_paper=None,
                blind_spectra="all train spectra (exact)", checkpoint="last")
    np.savez(out / "test_ranks.npz", identifier=msg.identifier[result[:, 0]], inchikey14=msg.inchikey14[result[:, 0]],
             rank=result[:, 1], ties=result[:, 2], candidates=result[:, 3], blind_rank=blind[:, 1],
             blind_paper_rank=np.zeros(0))
    (out / "test.json").write_text(json.dumps(test, indent=1))
    say(f"TEST {out.name}: {json.dumps(test['test'])} blind {json.dumps(test['blind'])}")
    return test


def main(c: JESTR):
    from sna.data import MSG
    from sna.train import metrics, run_info

    torch.manual_seed(c.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    names = c.names
    outs = [paths.RUNS / n for n in names]
    if all((o / "DONE").exists() for o in outs):
        return
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    say = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)
    msg, graphs = MSG(device), Graphs(device)
    spectra = binned(msg)
    model = Model().to(device)
    z = np.load(paths.PREP / "jestr.npz")
    reg_offsets, reg = torch.from_numpy(z["offsets"]).to(device), torch.from_numpy(z["molecule"].astype(np.int64)).to(device)
    pointer = torch.zeros(len(msg.fold), dtype=torch.long, device=device)
    train_queries = np.nonzero(msg.fold == 0)[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
    regularised_from = int(c.epochs * (1 - c.regularised_fraction))
    steps_per_epoch = math.ceil(len(train_queries) / c.batch)
    writers = [SummaryWriter(o / "tb") for o in outs]
    for o in outs:
        (o / "run.json").write_text(json.dumps(run_info(model), indent=1))

    def epoch(e, regularised, logs):
        model.train()
        rng = np.random.default_rng([c.seed, e])
        order = rng.permutation(train_queries)
        start = msg.start[order]
        rows = msg.by_query[start + (rng.random(len(order)) * (msg.start[order + 1] - start)).astype(np.int64)]
        total = torch.zeros((), device=device)
        for i in range(0, len(order), c.batch):
            q = torch.from_numpy(order[i:i + c.batch]).to(device)
            z_spec = model.spectrum(spectra[torch.from_numpy(rows[i:i + c.batch]).to(device)])
            loss = contrastive(z_spec, model.molecule(*graphs(msg.mol[q])), msg.key[q], c.temperature)
            if regularised:  # the next `candidates` entries of each query's sorted list, cyclically
                n = reg_offsets[q + 1] - reg_offsets[q]
                k = torch.arange(c.candidates, device=device)
                cand = reg[reg_offsets[q, None] + (pointer[q, None] + k) % n[:, None]]
                pointer[q] = (pointer[q] + c.candidates) % n
                z_cand = model.molecule(*graphs(cand.flatten())).view(len(q), c.candidates, -1)
                penalty = F.cosine_similarity(z_spec[:, None], z_cand, dim=-1).mean()
                loss = (1 - c.weight) * loss + c.weight * penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach()
        loss = total.item() / steps_per_epoch
        for w in logs:
            w.add_scalar("train/loss", loss, (e + 1) * steps_per_epoch)
        if (e + 1) % c.val_every == 0 or e + 1 == c.epochs:
            val = metrics(evaluate(model, msg, graphs, spectra, 1))
            for w in logs:
                w.add_scalar("val/mrr", val["mrr"], (e + 1) * steps_per_epoch)
            say(f"epoch {e + 1}/{c.epochs} loss {loss:.4f} val mrr {val['mrr']:.4f} r1 {val['r1']:.4f}"
                f"{' [regularised]' if regularised else ''}")

    t0 = time.time()
    for e in range(regularised_from):  # shared by both arms
        epoch(e, False, writers)
    say(f"shared epochs done in {(time.time() - t0) / 60:.0f} min")
    shared = copy.deepcopy((model.state_dict(), optimizer.state_dict()))
    for arm, (out, writer) in enumerate(zip(outs, writers)):
        model.load_state_dict(shared[0]), optimizer.load_state_dict(shared[1])
        torch.manual_seed(c.seed + regularised_from)
        pointer.zero_()
        for e in range(regularised_from, c.epochs):
            epoch(e, arm == 1, [writer])
        torch.save(dict(state=model.state_dict(), epoch=c.epochs, config=configs.to_dict(c)), out / "best.pt")
        trained = dict(best_step=c.epochs * steps_per_epoch, steps=c.epochs * steps_per_epoch,
                       best_val_mrr=metrics(evaluate(model, msg, graphs, spectra, 1))["mrr"])
        writer.add_text("test", json.dumps(test(model, msg, graphs, spectra, out, trained, say)))
        writer.close()
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sna.jestr CONFIG.json")
    main(configs.load(sys.argv[1]))
