import dataclasses

import pytest

from sna import config as configs
from sna.config import JESTR, Data, Encoder, Experiment, Objective
from sna.experiments import experiments, find


def test_round_trip(tmp_path):
    c = Experiment("x", Encoder("msjepa", "B", fresh=True, lr=1e-4), conditions=("ce",), data=Data("sim", steps=10),
                   objective=Objective(score="fine"), init="y")
    configs.save(c, tmp_path / "c.json")
    assert configs.load(tmp_path / "c.json") == c
    assert configs.from_dict(configs.to_dict(JESTR(epochs=7))) == JESTR(epochs=7)


def test_override_parses_by_declared_type():
    c = Experiment("x", Encoder("msjepa"))
    c = configs.override(c, "data.epochs=200")
    c = configs.override(c, "optimizer.lr=3e-4")
    c = configs.override(c, "encoder.lr=1e-4")
    c = configs.override(c, "encoder.fresh=true")
    c = configs.override(c, "conditions=ce,instrument")
    assert (c.data.epochs, c.optimizer.lr, c.encoder.fresh, c.conditions) == (200, 3e-4, True, ("ce", "instrument"))
    assert configs.override(c, "conditions=none").conditions is None
    with pytest.raises(AssertionError):
        configs.override(c, "data.epoch=3")  # typo: no such field


def test_unknown_fields_are_rejected():
    d = configs.to_dict(Experiment("x", Encoder("msjepa")))
    d["data"]["epoch"] = 3
    with pytest.raises(AssertionError):
        configs.from_dict(d)


def test_default_adapter_conditions():
    """What the encoder does not read itself goes to the adapter."""
    assert Experiment("x", Encoder("dreams_ssl")).adapter_conditions == ("adduct", "ce")
    assert Experiment("x", Encoder("msjepa", "A")).adapter_conditions == ()
    assert Experiment("x", Encoder("msjepa", "B")).adapter_conditions == ("ce", "instrument")
    assert Experiment("x", Encoder("msjepa", "B"), conditions=()).adapter_conditions == ()


def test_invalid_configs_fail_early():
    with pytest.raises(AssertionError):
        Encoder("dreams_ssl", fresh=True, lr=1e-4)
    with pytest.raises(AssertionError):
        Data("sim")  # no step budget
    with pytest.raises(AssertionError):
        Objective(loss="contrastive")


def test_registry_names_are_unique_and_found():
    names = [n for _, c in experiments() for n in (c.names if isinstance(c, JESTR) else (c.name,))]
    assert len(names) == len(set(names))
    assert find("frozen_msjepa_B").encoder == Encoder("msjepa", "B")
    assert dataclasses.replace(find("prior_molformer_candidate"), name="") == \
        Experiment("", Encoder("dreams_contrastive"), objective=Objective(loss="candidate"))


def test_cli_override_needs_a_new_name():
    from sna.cli import resolve
    with pytest.raises(SystemExit):
        resolve("frozen_msjepa_A", ["data.epochs=3"])
    assert resolve("frozen_msjepa_A", ["data.epochs=3"], "short").name == "short"
