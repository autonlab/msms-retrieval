"""Train, select and test one arm:  torchrun --nproc-per-node 4 -m sna.train CONFIG.json

Normally launched by the CLI (`sna run NAME`), which writes runs/<name>/config.json (sna.config.Experiment).
One recipe for every arm: a frozen encoder is Encoder.lr = 0 (eval mode, no gradients); sampling,
augmentation, batch, schedule, validation and checkpoint selection are shared. Validation scores the
full val fold against the official mass lists; the val-best checkpoint is scored once on the test fold.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter

from sna import config as configs, paths
from sna.config import Experiment
from sna.data import MSG, Simulated
from sna.model import Aligner, candidate_loss, inbatch_loss, maxsim, rank_candidates
from sna.spectra import AUGMENT, select_peaks
from sna.towers import TokenTable, Tower


def build(c: Experiment, device):
    """The aligner of a configuration (pretrained encoder weights loaded, adapters fresh)."""
    e = c.encoder
    if e.checkpoint.startswith("dreams"):
        from sna.encoders.dreams import load_dreams
        encoder = load_dreams(paths.ENCODERS[e.checkpoint])
    else:
        from sna.encoders.msjepa import fresh_msjepa, load_msjepa
        encoder = (fresh_msjepa if e.fresh else load_msjepa)(paths.ENCODERS[e.checkpoint], e.variant)
    return Aligner(encoder, 768, c.adapter_conditions, c.objective.score == "fine", e.lr == 0).to(device)


class Molecules:
    """Frozen-tower inputs for rows of one molecule table: pooled vectors (from the cache when
    there is one) or token states, with a mask."""

    def __init__(self, tower, tokens=None, pooled=None, fine=False):
        self.tower, self.tokens, self.pooled, self.fine = tower, tokens, pooled, fine

    def __call__(self, rows):
        if not self.fine and self.pooled is not None:
            return self.pooled[rows].float()[:, None], torch.ones(len(rows), 1, dtype=torch.bool, device=rows.device)
        return self.tower(*self.tokens(rows, self.tower.pad), pooled=not self.fine)


def spectrum_vectors(model, msg, spectra, embedding=None, chunk=512):
    """Adapted spectrum vectors [N,S,D] with mask [N,S] for spectrum rows. embedding: the encoder's
    vector replaced by this one, conditions kept (the mean-embedding control)."""
    s, s_mask = [], []
    for j in range(0, len(spectra), chunk):
        with torch.autocast("cuda", torch.bfloat16):
            x, m = model.adapter_input(msg.table.batch(spectra[j:j + chunk], model.encoder.max_peaks))
            if embedding is not None:
                x = torch.cat([embedding.expand(len(x), x.shape[1], -1).to(x.dtype), x[..., embedding.shape[-1]:]], -1)
            s.append(model.spectrum(x)), s_mask.append(m)
    return torch.cat(s), torch.cat(s_mask)


@torch.no_grad()
def evaluate(model, msg, molecules, fold, rank, world, control=None, group=16):
    """Rank of the true molecule for every spectrum of `fold` among its query's official
    candidates (fp32 scores; rank = 1 + #strictly higher, ties counted separately).
    control: a spectrum-blind control of blind_controls instead of the real spectra."""
    model.eval()
    queries = np.nonzero(msg.fold == fold)[0][rank::world]
    out = []
    for i in range(0, len(queries), group):
        qs = queries[i:i + group]
        rows = [msg.spectra(q) for q in qs]
        if control is None or control[0] == "embedding":
            s, s_mask = spectrum_vectors(model, msg, np.concatenate(rows), None if control is None else control[1])
        lists = [msg.candidates(q) for q in qs]
        with torch.autocast("cuda", torch.bfloat16):
            h, h_mask = molecules(torch.cat([c for c, _ in lists]))
            m = model.molecule(h, h_mask)
        a = b = 0
        for (cands, positive), spectrum_rows in zip(lists, rows):
            n, mc, hc = len(spectrum_rows), m[b:b + len(cands)], h_mask[b:b + len(cands)]
            if control is not None and control[0] == "expected":
                scores = expected_scores(*control[1:], mc, hc).expand(n, -1)
            else:  # up to 542 spectra a query
                scores = torch.cat([maxsim(s[j:min(j + 64, a + n)], s_mask[j:min(j + 64, a + n)], mc, hc)
                                    for j in range(a, a + n, 64)])
            ranks, ties = rank_candidates(scores, positive)
            out.append(np.stack([spectrum_rows, ranks.cpu().numpy(), ties.cpu().numpy(),
                                 np.full(n, len(cands))], 1))
            a, b = a + n, b + len(cands)
    gathered = [None] * world
    dist.all_gather_object(gathered, np.concatenate(out))
    result = np.concatenate(gathered)
    return result[np.argsort(result[:, 0])]  # columns: spectrum row, rank, ties, candidates


def expected_scores(bank, bank_mask, m, m_mask, chunk=64):
    """[1,C]: each candidate's score averaged over the bank's spectra. The bank is sorted by length and
    its padding is at the end (global, precursor, then peaks), so each chunk is cut to its longest."""
    total = 0.0
    for j in range(0, len(bank), chunk):
        width = int(bank_mask[j:j + chunk].sum(1).max())
        total = total + maxsim(bank[j:j + chunk, :width], bank_mask[j:j + chunk, :width], m, m_mask).sum(0)
    return (total / len(bank))[None]


@torch.no_grad()
def blind_controls(model, msg, rank, world, sample, seed=0):
    """Spectrum-blind controls from the train fold.

    expected: every candidate scored by its expected score over train spectra, E_x[score(x, m)], blind to
      spectrum and conditions. A global score is linear in the spectrum vector, so this is the score against
      the mean normalized vector over ALL train spectra (exact). MaxSim is not linear, so fine-grained scoring
      averages the score over `sample` train spectra drawn without replacement (unbiased).
    embedding (global only): the mean-embedding control, the encoder's vector replaced by its train mean,
      each query's own conditions kept. (A mean adapter INPUT is not used: it is off-manifold, a blend of adduct embeddings.)
    """
    model.eval()
    train = np.nonzero(msg.fold[msg.query] == 0)[0]
    if model.fine_grained:
        rows = np.sort(np.random.default_rng(seed).choice(train, sample, replace=False))
        s, mask = spectrum_vectors(model, msg, rows)
        order = mask.sum(1).argsort(stable=True)
        return dict(expected=("expected", s[order], mask[order]))
    u = e = 0.0
    for j in range(0, len(train[rank::world]), 1024):
        with torch.autocast("cuda", torch.bfloat16):
            x = model.adapter_input(msg.table.batch(train[rank::world][j:j + 1024], model.encoder.max_peaks))[0]
            u = u + model.spectrum(x)[:, 0].sum(0)
            e = e + x[:, 0, :model.encoder.dim].float().sum(0)
    count = torch.tensor(float(len(train[rank::world])), device=u.device)
    dist.all_reduce(u), dist.all_reduce(e), dist.all_reduce(count)
    one = torch.ones(1, 1, dtype=torch.bool, device=u.device)
    return dict(expected=("expected", (u / count)[None, None], one), embedding=("embedding", (e / count)[None, None]))


HARMONIC = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, 4097))])


def metrics(result):
    rank = result[:, 1]
    return dict(mrr=float(np.mean(1.0 / rank)), r1=float(np.mean(rank <= 1)), r5=float(np.mean(rank <= 5)),
                r20=float(np.mean(rank <= 20)), spectra=int(len(rank)), tied=float(np.mean(result[:, 2] > 0)),
                random_mrr=float(np.mean(HARMONIC[result[:, 3]] / result[:, 3])))  # uniform ranking: E[1/rank] = H_K / K


def run_info(model):
    """Basic runtime information accompanying the saved configuration."""
    import transformers, rdkit
    return dict(
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
        gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        versions=dict(python=sys.version.split()[0], torch=torch.__version__, transformers=transformers.__version__,
                      rdkit=rdkit.__version__, numpy=np.__version__),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)),
    )


def main(c: Experiment):
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.manual_seed(c.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = paths.RUNS / c.name
    if (out / "DONE").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    say = (lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)) if rank == 0 else (lambda *m: None)

    msg = MSG(device)
    model = build(c, device)
    fine, sim = c.objective.score == "fine", c.data.source == "sim"
    tower = Tower(c.tower).to(device) if fine or sim else None
    pooled = None if fine else torch.from_numpy(np.load(paths.PREP / f"pooled_{c.tower}.npy", mmap_mode="r")[:]).to(device)
    real = Molecules(tower, TokenTable(paths.PREP / f"tokens_{c.tower}.npz", device) if fine else None, pooled, fine)
    if c.init:
        state = torch.load(paths.RUNS / c.init / "best.pt", map_location=device)["state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected and all(k.startswith("encoder.") for k in missing) and (not missing or model.frozen_encoder), (missing, unexpected)
    if sim:
        simulated = Molecules(tower, TokenTable(paths.PREP / "autonmol_tokens_molformer.npz", device), None, fine)
        stream = iter(torch.utils.data.DataLoader(Simulated(c.data.batch, c.seed, rank), batch_size=None, num_workers=6,
                                                  pin_memory=True, prefetch_factor=4, persistent_workers=True))
        steps = c.data.steps
    else:
        per_epoch = int((msg.fold == 0).sum()) // (c.data.batch * world)
        steps = c.data.epochs * per_epoch
    if rank == 0:
        (out / "run.json").write_text(json.dumps(run_info(model), indent=1))
    say(f"{c.name}: {steps} steps, {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    head = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.") and n != "log_tau"]
    body = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    o = c.optimizer
    groups = [dict(params=[p for p in ps if p.ndim >= 2], weight_decay=o.weight_decay, lr=lr) for ps, lr in ((head, o.lr), (body, c.encoder.lr))]
    groups += [dict(params=[p for p in ps if p.ndim < 2], weight_decay=0.0, lr=lr) for ps, lr in ((head, o.lr), (body, c.encoder.lr))]
    groups += [dict(params=[model.log_tau], weight_decay=0.0, lr=o.tau_lr)]
    optimizer = torch.optim.AdamW([g for g in groups if g["params"]])
    warmup = max(1, int(o.warmup * steps))
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda t: min((t + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(1.0, max(0, t - warmup) / max(1, steps - warmup))))))
    ddp = DistributedDataParallel(model, device_ids=[device.index])
    writer = SummaryWriter(out / "tb") if rank == 0 else None
    generator = torch.Generator(device).manual_seed(c.seed * 1000 + rank)
    best, best_step, epoch_rows, t0 = -1.0, -1, None, time.time()

    for step in range(steps):
        model.train()
        if sim:
            b = {k: v.to(device, non_blocking=True) for k, v in next(stream).items()}
            b["mz"], b["intensity"], b["mask"] = select_peaks(b["mz"], b["intensity"], b["mask"], model.encoder.max_peaks, AUGMENT, generator)
            key = b.pop("molecule")
            h, h_mask = simulated(key)
        else:
            if step % per_epoch == 0:
                epoch_rows = msg.epoch(step // per_epoch, c.seed, c.data.batch, rank, world)
            rows = epoch_rows[step % per_epoch]
            b = msg.table.batch(rows, model.encoder.max_peaks, AUGMENT, generator)
            queries = torch.from_numpy(msg.query[rows]).to(device)
            key = msg.key[queries]
            if c.objective.loss == "candidate":
                cand, valid = msg.sample_candidates(queries, c.objective.candidates, generator)
                h, h_mask = real(cand.flatten())
            else:
                h, h_mask = real(msg.mol[queries])
        with torch.autocast("cuda", torch.bfloat16):
            logits = ddp(b, h, h_mask, cand.shape[1] if c.objective.loss == "candidate" else None)
        loss = candidate_loss(logits, valid, torch.zeros_like(key)) if c.objective.loss == "candidate" else inbatch_loss(logits, key)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), o.clip, error_if_nonfinite=True)
        optimizer.step()
        model.clamp_tau()
        schedule.step()
        if writer and (step % 50 == 0 or step == steps - 1):
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/tau", model.tau.item(), step)
            writer.add_scalar("train/grad_norm", grad.item(), step)
            for i, g in enumerate(optimizer.param_groups):
                writer.add_scalar(f"lr/group{i}", g["lr"], step)
            writer.add_scalar("perf/steps_per_s", (step + 1) / (time.time() - t0), step)
            writer.add_scalar("perf/max_memory_gib", torch.cuda.max_memory_allocated() / 2**30, step)
        if (step + 1) % c.evaluation.val_every == 0 or step + 1 == steps:
            val = metrics(evaluate(model, msg, real, 1, rank, world))
            if writer:
                for k in ("mrr", "r1", "r5", "r20"):
                    writer.add_scalar(f"val/{k}", val[k], step)
            say(f"step {step + 1}/{steps} loss {loss.item():.4f} tau {model.tau.item():.4f} val mrr {val['mrr']:.4f} r1 {val['r1']:.4f}")
            if val["mrr"] > best:
                best, best_step = val["mrr"], step + 1
                if rank == 0:
                    state = {k: v for k, v in model.state_dict().items() if not (model.frozen_encoder and k.startswith("encoder."))}
                    torch.save(dict(state=state, step=step + 1, val=val, config=configs.to_dict(c)), out / "best.tmp")
                    os.replace(out / "best.tmp", out / "best.pt")
            dist.barrier()

    test(c, model, msg, real, out, rank, world, say, dict(best_step=best_step, steps=steps, best_val_mrr=best))
    if writer:
        writer.add_text("test", (out / "test.json").read_text())
        writer.close()
        (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    dist.barrier()
    dist.destroy_process_group()


def test(c: Experiment, model, msg, real, out, rank, world, say, trained):
    """Score best.pt on the test fold with both spectrum-blind controls (expected score: blind; mean embedding:
    blind_paper); writes test.json and test_ranks.npz."""
    model.load_state_dict(torch.load(out / "best.pt", map_location=model.log_tau.device)["state"], strict=not model.frozen_encoder)
    result = evaluate(model, msg, real, 2, rank, world)
    blind = {k: evaluate(model, msg, real, 2, rank, world, c)
             for k, c in blind_controls(model, msg, rank, world, c.evaluation.blind_spectra).items()}
    if rank == 0:
        paper = blind.get("embedding")
        test = dict(test=metrics(result), **trained, blind=metrics(blind["expected"]),
                    blind_paper=None if paper is None else metrics(paper),
                    blind_spectra=c.evaluation.blind_spectra if model.fine_grained else "all train spectra (exact)")
        np.savez(out / "test_ranks.npz", identifier=msg.identifier[result[:, 0]], inchikey14=msg.inchikey14[result[:, 0]],
                 rank=result[:, 1], ties=result[:, 2], candidates=result[:, 3], blind_rank=blind["expected"][:, 1],
                 blind_paper_rank=np.zeros(0) if paper is None else paper[:, 1])
        (out / "test.json").write_text(json.dumps(test, indent=1))
        say(f"TEST {c.name}: {json.dumps(test['test'])} blind {json.dumps(test['blind'])} blind_paper {json.dumps(test['blind_paper'])}")
    dist.barrier()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m sna.train CONFIG.json")
    main(configs.load(sys.argv[1]))
