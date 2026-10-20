"""Experiment configurations: frozen dataclasses whose field defaults ARE the shared recipe.

An experiment states only how it differs from the recipe; everything else is a default here.
Configurations round-trip through JSON (runs/<name>/config.json) and take dotted overrides
(`data.epochs=200`), so a run is fully described by its config file.
"""
from __future__ import annotations

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Encoder:
    checkpoint: str                      # key of sna.paths.ENCODERS
    variant: str = "A"                   # MS-JEPA: A = aggregated h_mol (conditions in the encoder), B = [CLS]
    fresh: bool = False                  # the checkpoint's architecture, randomly initialized
    lr: float = 0.0                      # 0 freezes the encoder (eval mode, no gradients)

    def __post_init__(self):
        assert self.variant in ("A", "B"), self.variant
        assert not (self.fresh and self.checkpoint.startswith("dreams")), "no fresh DreaMS init"
        assert not (self.fresh and self.lr == 0), "a fresh encoder must be trained"


@dataclass(frozen=True)
class Data:
    source: str = "real"                 # real: MassSpecGym train fold | sim: the AutonMol store
    epochs: int = 100                    # real: molecule epochs (each train query once, one random spectrum)
    steps: int = 0                       # sim: optimizer steps
    batch: int = 64                      # per GPU; in-batch negatives are per GPU

    def __post_init__(self):
        assert self.source in ("real", "sim"), self.source
        assert self.source == "real" or self.steps > 0, "simulated data needs steps"


@dataclass(frozen=True)
class Objective:
    score: str = "global"                # global: one vector a side | fine: MaxSim over spectrum and token vectors
    loss: str = "inbatch"                # inbatch | candidate: InfoNCE over the query's official list
    candidates: int = 128                # per query, candidate loss

    def __post_init__(self):
        assert self.score in ("global", "fine") and self.loss in ("inbatch", "candidate"), (self.score, self.loss)


@dataclass(frozen=True)
class Optimizer:
    lr: float = 1e-4                     # adapters and conditions (the encoder's is Encoder.lr)
    tau_lr: float = 1e-2                 # log temperature: settles within warm-up, independent of run length
    weight_decay: float = 0.02
    warmup: float = 0.05                 # fraction of steps, then cosine
    clip: float = 1.0


@dataclass(frozen=True)
class Evaluation:
    val_every: int = 200                 # steps; validation scores the full val fold
    blind_spectra: int = 16384           # fine-grained: train spectra the expected-score control averages over


@dataclass(frozen=True)
class Experiment:
    """One MSAlign-style arm (sna.train)."""
    name: str
    encoder: Encoder
    tower: str = "molformer"             # frozen molecule tower, key of sna.paths.TOWERS
    conditions: tuple[str, ...] | None = None  # adapter conditions; None: what the encoder does not read itself
    data: Data = field(default_factory=Data)
    objective: Objective = field(default_factory=Objective)
    optimizer: Optimizer = field(default_factory=Optimizer)
    evaluation: Evaluation = field(default_factory=Evaluation)
    init: str = ""                       # experiment whose best.pt this one starts from
    seed: int = 42

    def __post_init__(self):
        assert set(self.conditions or ()) <= {"adduct", "ce", "instrument"}, self.conditions
        assert self.data.source == "real" or (self.tower == "molformer" and self.objective.loss == "inbatch")

    @property
    def adapter_conditions(self) -> tuple[str, ...]:
        """Default: DreaMS reads no conditions, so the adapter gets adduct and CE; MS-JEPA (A) reads all of
        them in the encoder; (B)'s [CLS] sees only the ion condition, so the adapter gets CE and instrument."""
        if self.conditions is not None:
            return self.conditions
        if self.encoder.checkpoint.startswith("dreams"):
            return ("adduct", "ce")
        return () if self.encoder.variant == "A" else ("ce", "instrument")


@dataclass(frozen=True)
class JESTR:
    """JESTR-NR and JESTR (sna.jestr): the published recipe; both arms share every epoch before the regulariser."""
    names: tuple[str, str] = ("jestr_nr", "jestr")  # without, with the regulariser
    epochs: int = 1000
    regularised_fraction: float = 0.03   # the last 3% of epochs
    candidates: int = 32                 # regulariser candidates per query and step
    weight: float = 0.1                  # regulariser weight; InfoNCE gets 1 - weight
    batch: int = 32
    lr: float = 5e-4                     # Adam
    temperature: float = 0.05
    val_every: int = 10                  # epochs; monitoring only (the last checkpoint is tested)
    seed: int = 42

    @property
    def name(self) -> str:
        return self.names[1]


CONFIGS = (Experiment, JESTR)


def to_dict(config) -> dict:
    return dict(kind=type(config).__name__, **dataclasses.asdict(config))


def from_dict(d: dict):
    d = dict(d)
    cls = {c.__name__: c for c in CONFIGS}[d.pop("kind")]
    return _build(cls, d)


def _build(cls, d: dict):
    hints = typing.get_type_hints(cls)
    unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
    assert not unknown, f"{cls.__name__}: unknown fields {sorted(unknown)}"
    kwargs = {}
    for name, value in d.items():
        kind = hints[name]
        if dataclasses.is_dataclass(kind):
            value = _build(kind, value)
        elif isinstance(value, list):
            value = tuple(value)
        kwargs[name] = value
    return cls(**kwargs)


def save(config, path: Path) -> None:
    path.write_text(json.dumps(to_dict(config), indent=1) + "\n")


def load(path: Path):
    return from_dict(json.loads(Path(path).read_text()))


def override(config, assignment: str):
    """A copy of config with one dotted field set, e.g. 'data.epochs=200', 'conditions=ce,instrument',
    'encoder.fresh=true'. The value is parsed by the field's declared type."""
    path, _, text = assignment.partition("=")
    assert _, f"expected key=value, got {assignment!r}"
    head, *rest = path.split(".")
    hints = typing.get_type_hints(type(config))
    assert head in hints, f"{type(config).__name__} has no field {head!r}"
    if rest:
        value = override(getattr(config, head), ".".join(rest) + "=" + text)
    else:
        value = _parse(hints[head], text)
    return dataclasses.replace(config, **{head: value})


def _parse(kind, text: str):
    args = typing.get_args(kind)
    if type(None) in args:  # Optional
        if text.lower() in ("none", "null", ""):
            return None
        kind = next(a for a in args if a is not type(None))
    origin = typing.get_origin(kind)
    if origin is tuple:
        return tuple(t for t in text.split(",") if t)
    if kind is bool:
        assert text.lower() in ("true", "false", "1", "0"), text
        return text.lower() in ("true", "1")
    return kind(text)
