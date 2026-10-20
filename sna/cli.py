"""Run the retrieval experiments: list, show, run, all, report.

Use --timeout-hours on run/all to limit each run (0 disables the timeout).
GPUs: --gpus or $SNA_GPUS; JESTR uses the first GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from sna import config as configs, paths
from sna.config import JESTR, Experiment
from sna.experiments import experiments, find

REPO = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", HF_HUB_OFFLINE="1", PYTHONPATH=str(REPO),
           OMP_NUM_THREADS="8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")


def say(*message):
    print(time.strftime("%F %T"), *message, flush=True)


def names_of(c) -> tuple[str, ...]:
    return c.names if isinstance(c, JESTR) else (c.name,)


def done(name) -> bool:
    return (paths.RUNS / name / "DONE").exists()


def execute(cmd, log, gpus, hours):
    with open(log, "a") as handle:
        job = subprocess.Popen(cmd, cwd=REPO, env=dict(ENV, CUDA_VISIBLE_DEVICES=gpus), stdout=handle,
                               stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return job.wait(timeout=hours * 3600 if hours else None)
        except subprocess.TimeoutExpired:  # torchrun starts each worker in its own session: kill the whole tree
            for pid in (job.pid, *descendants(job.pid)):
                os.kill(pid, signal.SIGKILL)
            job.wait()
            return "timeout"


def descendants(pid):
    children = [int(c) for c in subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()]
    return children + [d for c in children for d in descendants(c)]


def command(c, gpus):
    path = str(paths.RUNS / names_of(c)[-1] / "config.json")
    if isinstance(c, JESTR):
        return [sys.executable, "-m", "sna.jestr", path]
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node", str(len(gpus.split(","))),
            "-m", "sna.train", path]


def run(c, gpus, force=False, timeout_hours=0) -> bool:
    """Train and test one experiment unless it has finished. False on failure."""
    names = names_of(c)
    if all(done(n) for n in names) and not force:
        return True
    if isinstance(c, Experiment) and c.init and not done(c.init):
        say("FAILED", c.name, f"needs {c.init} first")
        return False
    for name in names:  # an earlier attempt: keep its log, start clean
        out = paths.RUNS / name
        if out.exists():
            (paths.RUNS / "_failed").mkdir(exist_ok=True)
            if (out / "train.log").exists():
                shutil.move(out / "train.log", paths.RUNS / "_failed" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log")
            shutil.rmtree(out)
        out.mkdir(parents=True)
        configs.save(c, out / "config.json")
    gpus = gpus.split(",")[0] if isinstance(c, JESTR) else gpus
    say("START", " ".join(names), f"(tensorboard: {' '.join(str(paths.RUNS / n / 'tb') for n in names)})")
    code = execute(command(c, gpus), paths.RUNS / names[-1] / "train.log", gpus, timeout_hours)
    if code or not all(done(n) for n in names):
        say("FAILED", " ".join(names), "exit", code, f"(log: {paths.RUNS / names[-1] / 'train.log'})")
        return False
    for n in names:
        say("DONE", n, json.loads((paths.RUNS / n / "test.json").read_text())["test"])
    return True


def resolve(name, assignments=(), new_name=None):
    """A registered experiment (or a config .json) with overrides; a changed registered experiment must be renamed."""
    c = configs.load(name) if name.endswith(".json") else find(name)
    for assignment in assignments:
        c = configs.override(c, assignment)
    if new_name:
        c = configs.override(c, f"{'names' if isinstance(c, JESTR) else 'name'}={new_name}")
    if assignments and not name.endswith(".json") and set(names_of(c)) & set(names_of(find(name))):
        raise SystemExit(f"overriding {name!r} changes it: give the result a new name with --as (JESTR: --as NR,REGULARISED)")
    return c


def main(argv=None):
    p = argparse.ArgumentParser(prog="sna", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("show", "run"):
        s = sub.add_parser(name)
        s.add_argument("name")
        s.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE")
        s.add_argument("--as", dest="new_name")
        if name == "run":
            s.add_argument("--force", action="store_true", help="discard a finished run and train it again")
    sub.add_parser("all")
    for name in ("run", "all"):
        sub.choices[name].add_argument("--timeout-hours", type=float, default=0,
                                      help="time limit per run in hours; 0 disables it (default)")
    sub.add_parser("report")
    for s in sub.choices.values():
        s.add_argument("--gpus", default=os.environ.get("SNA_GPUS", "0,1,2,3"))
    a = p.parse_args(argv)
    if hasattr(a, "timeout_hours") and (not 0 <= a.timeout_hours < float("inf")):
        p.error("--timeout-hours must be finite and nonnegative")

    if a.command == "list":
        width = max(len(n) for _, c in experiments() for n in names_of(c))
        group = None
        for g, c in experiments():
            if g != group:
                print(f"\n{g}")
                group = g
            for n in names_of(c):
                status = "done" if done(n) else "started" if (paths.RUNS / n).exists() else "pending"
                mrr = json.loads((paths.RUNS / n / "test.json").read_text())["test"]["mrr"] if done(n) else None
                print(f"  {n:{width}s}  {status:8s}" + (f"  test MRR {mrr:.3f}" if mrr is not None else ""))
    elif a.command == "show":
        print(json.dumps(configs.to_dict(resolve(a.name, a.set, a.new_name)), indent=1))
    elif a.command == "run":
        sys.exit(0 if run(resolve(a.name, a.set, a.new_name), a.gpus, a.force, a.timeout_hours) else 1)
    elif a.command == "all":
        for _, c in experiments():
            if not run(c, a.gpus, timeout_hours=a.timeout_hours):
                sys.exit(1)
        from sna import report
        report.main()
        say("ALL DONE")
    elif a.command == "report":
        from sna import report
        report.main()


if __name__ == "__main__":
    main()
