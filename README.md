# Separability Is Not Alignability

Retrieval experiments for *Separability Is Not Alignability: Representations for
Spectrum-to-Molecule Retrieval*. This release covers retrieval training and evaluation;
encoder pretraining, spectrum simulation, and geometry analyses are outside its scope.

## Setup

Use Python 3.12 or newer and CUDA GPUs. The paper runs used four A100 80 GB GPUs
(one for JESTR); using a different GPU count changes the training recipe.

```bash
pip install -e '.[test]'
cp .env.example .env
```

Fill in `.env` with the MassSpecGym 1.5 TSV, candidate-list directory, output directory,
and checkpoints for the encoders you want to use. Simulated experiments also need
`SNA_AUTONMOL`. Inputs must already be available locally.

Download the molecule towers once:

```bash
python -c "from huggingface_hub import snapshot_download as d; from sna.paths import TOWERS; [d(name, revision=rev) for name, rev in TOWERS.values()]"
```

Prepare real data and pooled molecule embeddings:

```bash
python scripts/prepare.py
CUDA_VISIBLE_DEVICES=0 python -m sna.towers molformer
CUDA_VISIBLE_DEVICES=0 python -m sna.towers chemberta
```

For simulated experiments, also run `python scripts/prepare.py --sim`.
Preparation reuses existing outputs; use a new output directory when changing inputs.
The full prepared data occupies about 29 GB, plus 42 GB for the simulated store.

## Run

```bash
sna list
sna show frozen_msjepa_B
sna run frozen_msjepa_B
sna run jestr
sna run fresh_A_lr1e-4 --set data.epochs=400 --as fresh_A_lr1e-4_ep400
sna all --timeout-hours 12
sna report
```

`run` executes one configuration; JESTR runs both regularized and unregularized arms.
`all` runs unfinished configurations in dependency order and stops on failure.
`--timeout-hours` limits each run; the default, `0`, disables the timeout.
Failed runs restart from scratch. `--force` replaces an existing run.

Configurations are in [sna/experiments.py](sna/experiments.py), with shared defaults in
[sna/config.py](sna/config.py). The paper's validation-selected carrier and fresh learning
rate are fixed in the registry. A configuration with `init` requires that run's completed
checkpoint; all other configurations can run independently.

Outputs go to `$SNA_OUT/runs/<name>/`: configuration, runtime information, logs,
TensorBoard events, checkpoint, test metrics, and per-spectrum ranks. `sna report`
writes `results.json` with molecule-bootstrap intervals and selected paired comparisons
for ladder rungs, training lengths, and JESTR.

Run `pytest tests` after preparing the data and configuring checkpoint paths.
Dependencies are pinned to the tested environment in [pyproject.toml](pyproject.toml).

## License

[MIT](LICENSE). Upstream acknowledgments and license notices are in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). External models and datasets retain their
own licenses.
