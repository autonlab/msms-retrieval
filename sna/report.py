"""Collect every finished run into runs/results.json with molecule-level bootstrap CIs.

Test spectra are resampled by InChIKey-14 group (1,000 draws), so an interval reflects which
molecules happen to be in the test fold. Comparisons that change one thing (a ladder rung,
the training length, JESTR's regulariser) get paired intervals: both arms on the same draws.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np

from sna import config as configs, paths
from sna.config import Experiment

KEYS = ("mrr", "r1", "r5", "r20")


def per_spectrum(rank):
    return np.stack([1.0 / rank, rank <= 1, rank <= 5, rank <= 20], 1).astype(np.float64)


def bootstrap(groups, values, draws=1000, seed=0):
    """95% intervals of the spectrum-averaged metrics, resampling molecules (groups)."""
    _, g = np.unique(groups, return_inverse=True)
    sums = np.stack([np.bincount(g, weights=values[:, j]) for j in range(values.shape[1])], 1)
    counts = np.bincount(g).astype(np.float64)
    pick = np.random.default_rng(seed).integers(0, len(counts), (draws, len(counts)))
    stats = sums[pick].sum(1) / counts[pick].sum(1)[:, None]
    return np.percentile(stats, [2.5, 97.5], axis=0)


def paired(a, b, key="rank", draws=1000, seed=0):
    za, zb = (np.load(paths.RUNS / n / "test_ranks.npz") for n in (a, b))
    assert (za["identifier"] == zb["identifier"]).all()
    diff = per_spectrum(zb[key]) - per_spectrum(za[key])
    ci = bootstrap(za["inchikey14"], diff, draws, seed)
    return {k: dict(delta=float(diff[:, j].mean()), ci=[float(ci[0, j]), float(ci[1, j])]) for j, k in enumerate(KEYS)}


def base_of(configs_, name):
    """The run a ladder starts from: the shared recipe with the ladder's encoder."""
    base = Experiment("", configs_[name].encoder)
    return next((n for n, c in configs_.items() if isinstance(c, Experiment) and dataclasses.replace(c, name="") == base), None)


def curves(out):
    """Validation MRR, temperature and training loss against optimizer step, from the run's TensorBoard log."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(out / "tb"), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    return {t: [[e.step + 1, round(e.value, 5)] for e in ea.Scalars(t)] for t in ("val/mrr", "train/tau", "train/loss") if t in tags}


def main():
    runs, config = {}, {}
    for test in sorted(paths.RUNS.glob("*/test.json")):
        out = test.parent
        if not (out / "DONE").exists():
            continue
        t, info = json.loads(test.read_text()), json.loads((out / "run.json").read_text())
        config[out.name] = configs.load(out / "config.json")
        z = np.load(out / "test_ranks.npz", allow_pickle=False)
        ci = bootstrap(z["inchikey14"], per_spectrum(z["rank"]))
        row = dict(config=configs.to_dict(config[out.name]), best_step=t["best_step"], steps=t["steps"], val_mrr=t["best_val_mrr"],
                   **{k: t["test"][k] for k in (*KEYS, "spectra", "tied", "random_mrr")},
                   ci={k: [float(ci[0, j]), float(ci[1, j])] for j, k in enumerate(KEYS)},
                   trainable=info["parameters"]["trainable"],
                   blind_spectra=t.get("blind_spectra"), curves=curves(out))
        for key, suffix in (("blind", ""), ("blind_paper", "_embedding")):  # expected-score; mean-embedding control
            if t.get(key):
                row["blind_mrr" + suffix] = t[key]["mrr"]
                row["delta0" + suffix] = t[key]["mrr"] - t["test"]["random_mrr"]
                row["delta_s" + suffix] = t["test"]["mrr"] - t[key]["mrr"]
        runs[out.name] = row

    # One change at a time, per ladder ("A"/"B": the pretrained carrier; "fresh": a fresh encoder).
    ladder = {}
    for name in runs:
        if name.startswith("ladder_") and name.endswith("_fine") and name.count("_") == 2:
            p, base = name.split("_")[1], base_of(config, name)
            rung = lambda r: f"ladder_{p}_{r}"
            pairs = [(base, rung("fine")), (base, rung("sim")), (rung("fine"), rung("sim_fine")),
                     (rung("sim"), rung("sim_fine")), (rung("sim_fine"), rung("sim_fine_candidate"))]
            ladder[p] = [dict(frm=a, to=b, **paired(a, b)) for a, b in pairs if a in runs and b in runs]
    # Training length: 48 -> 100 -> 200 epochs.
    length = []
    for short in (n for n in runs if n.endswith("_ep48")):
        default, long = short.removesuffix("_ep48"), short.replace("_ep48", "_ep200")
        length += [dict(frm=a, to=b, **paired(a, b)) for a, b in ((short, default), (default, long)) if a in runs and b in runs]
    # JESTR's regulariser, on the full score and on the spectrum-blind control.
    jestr = [dict(frm="jestr_nr", to="jestr", full=paired("jestr_nr", "jestr"), blind=paired("jestr_nr", "jestr", "blind_rank"))
             ] if {"jestr_nr", "jestr"} <= set(runs) else []
    result = dict(runs=runs, ladder=ladder, length=length, jestr=jestr)
    (paths.RUNS / "results.json").write_text(json.dumps(result, indent=1))

    print(f"{'run':44s} {'MRR':>6s} {'R@1':>6s} {'R@5':>6s} {'R@20':>6s}  {'MRR 95% CI':>15s} {'blind':>6s} {'blindE':>6s}")
    for name, r in runs.items():
        cell = lambda k: f"{r[k]:.3f}" if k in r else ""
        print(f"{name:44s} {r['mrr']:.3f}  {r['r1']:.3f}  {r['r5']:.3f}  {r['r20']:.3f}  "
              f"[{r['ci']['mrr'][0]:.3f},{r['ci']['mrr'][1]:.3f}] {cell('blind_mrr'):>6s} {cell('blind_mrr_embedding'):>6s}")
    for group, rows in [*((f"ladder {p}", rows) for p, rows in ladder.items()), ("length", length)]:
        for d in rows:
            print(f"{group:10s} {d['frm']:>36s} -> {d['to']:36s} dMRR {d['mrr']['delta']:+.3f} "
                  f"[{d['mrr']['ci'][0]:+.3f},{d['mrr']['ci'][1]:+.3f}]")
    for d in jestr:
        for part in ("full", "blind"):
            m = d[part]["mrr"]
            print(f"JESTR regulariser, {part} MRR {m['delta']:+.3f} [{m['ci'][0]:+.3f},{m['ci'][1]:+.3f}]")


if __name__ == "__main__":
    main()
