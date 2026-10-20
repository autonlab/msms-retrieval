"""Paper configurations, available independently of previous runs.

The paper's validation-selected carrier (MS-JEPA A, LR 5e-5) and fresh LR (1e-4)
are fixed here. Only configurations with `init` require another run's checkpoint.
"""
from __future__ import annotations

from collections.abc import Iterator

from sna.config import JESTR, Data, Encoder, Evaluation, Experiment, Objective

LRS = ("5e-6", "1.5e-5", "5e-5")         # end-to-end and unfrozen encoder LRs
FRESH_LRS = ("1e-4", "3e-4")
EPOCH = 97                               # steps per molecule epoch: 25,046 train molecules // (64 x 4)
FINE = dict(objective=Objective(score="fine"), evaluation=Evaluation(val_every=5 * EPOCH))
SIM = dict(data=Data(source="sim", steps=46_000), evaluation=Evaluation(val_every=2000))


def ladder(prefix: str, encoder: Encoder) -> list[Experiment]:
    """Fine-grained (real only); simulated pretraining without, then with, fine-grained; the
    candidate objective on the model that has both."""
    name = lambda rung: f"ladder_{prefix}_{rung}"
    return [
        Experiment(name("fine"), encoder, **FINE),
        Experiment(name("sim_pretrain"), encoder, **SIM),
        Experiment(name("sim"), encoder, init=name("sim_pretrain")),
        Experiment(name("sim_fine_pretrain"), encoder, **SIM, objective=Objective(score="fine")),
        Experiment(name("sim_fine"), encoder, **FINE, init=name("sim_fine_pretrain")),
        Experiment(name("sim_fine_candidate"), encoder, evaluation=FINE["evaluation"], init=name("sim_fine_pretrain"),
                   objective=Objective(score="fine", loss="candidate")),
    ]


def experiments() -> Iterator[tuple[str, Experiment | JESTR]]:
    """(group, config), with initialization checkpoints before their dependent runs."""
    yield from (("frozen encoders", e) for e in (
        Experiment("frozen_dreams_ssl", Encoder("dreams_ssl")),
        Experiment("frozen_dreams_contrastive", Encoder("dreams_contrastive")),
        Experiment("frozen_msjepa_A", Encoder("msjepa", "A")),
        Experiment("frozen_msjepa_B", Encoder("msjepa", "B"))))
    yield from (("adapter prior", e) for e in (
        Experiment("prior_chemberta_inbatch", Encoder("dreams_contrastive"), tower="chemberta"),
        Experiment("prior_chemberta_candidate", Encoder("dreams_contrastive"), tower="chemberta",
                   objective=Objective(loss="candidate")),
        Experiment("prior_molformer_candidate", Encoder("dreams_contrastive"), objective=Objective(loss="candidate"))))
    yield from (("molecule-tower term", Experiment(f"frozen_{k}", Encoder(k))) for k in ("dreams_vf", "dreams_vf_control"))
    yield from (("molecule-tower term", Experiment(f"frozen_msjepa_geom_{a}", Encoder(f"msjepa_geom_{a}")))
                for a in ("base", "mcos", "vf"))
    yield from (("unfrozen DreaMS", Experiment(f"unfrozen_dreams_contrastive_lr{x}", Encoder("dreams_contrastive", lr=float(x))))
                for x in LRS)
    e2e = {(v, x): Experiment(f"e2e_msjepa_{v}_lr{x}", Encoder("msjepa", v, lr=float(x))) for v in "AB" for x in LRS}
    yield from (("end-to-end MS-JEPA", e) for e in e2e.values())

    carrier = e2e["A", "5e-5"].encoder
    fresh = {x: Experiment(f"fresh_{carrier.variant}_lr{x}", Encoder("msjepa", carrier.variant, fresh=True, lr=float(x)))
             for x in FRESH_LRS}
    yield from (("fresh encoder", e) for e in fresh.values())
    yield from (("ladder, pretrained carrier", e) for e in ladder(carrier.variant, carrier))
    yield "JESTR", JESTR()

    base = fresh["1e-4"]
    for epochs in (48, 200):  # the fresh base and fresh + fine-grained at 48 and 200 epochs (default 100)
        yield "training length", Experiment(f"{base.name}_ep{epochs}", base.encoder, data=Data(epochs=epochs))
        yield "training length", Experiment(f"ladder_fresh_fine_ep{epochs}", base.encoder, data=Data(epochs=epochs), **FINE)
    yield from (("ladder, fresh encoder", e) for e in ladder("fresh", base.encoder))


def find(name: str) -> Experiment | JESTR:
    for _, config in experiments():
        if name == config.name or isinstance(config, JESTR) and name in config.names:
            return config
    raise KeyError(f"no experiment {name!r}; see `sna list`")
