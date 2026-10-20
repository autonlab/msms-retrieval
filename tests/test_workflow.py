import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

from sna import cli, paths
from sna.config import Experiment
from sna.experiments import experiments, find


def test_registry_is_available_without_completed_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    entries = list(experiments())
    assert len(entries) == 40
    assert find("jestr").names == ("jestr_nr", "jestr")
    assert find("fresh_A_lr1e-4").encoder.fresh
    assert find("ladder_A_fine").encoder.lr == 5e-5
    assert find("ladder_fresh_fine").encoder.lr == 1e-4
    preceding = set()
    for _, c in entries:
        if isinstance(c, Experiment) and c.init:
            assert c.init in preceding
        preceding.update(cli.names_of(c))


@pytest.mark.parametrize("sim", [False, True])
def test_preparation_only_reads_sim_when_requested(tmp_path, monkeypatch, sim):
    spec = importlib.util.spec_from_file_location("prepare", Path(__file__).parents[1] / "scripts/prepare.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    real, simulated = Mock(return_value={}), Mock(return_value={})
    monkeypatch.setattr(prepare, "PREP", tmp_path)
    monkeypatch.setattr(prepare, "AUTONMOL", tmp_path / "missing_sim_store")
    monkeypatch.setattr(prepare, "STEPS", {
        "msg": (("queries.npz",), real), "autonmol": (("autonmol.npz",), simulated)})
    monkeypatch.setattr("sys.argv", ["prepare.py", *(["--sim"] if sim else [])])
    prepare.main()
    real.assert_called_once()
    assert simulated.call_count == int(sim)


@pytest.mark.parametrize("hours,seconds", [(0, None), (0.25, 900)])
def test_timeout_reaches_subprocess(tmp_path, monkeypatch, hours, seconds):
    job = Mock()
    job.wait.return_value = 0
    monkeypatch.setattr(cli.subprocess, "Popen", Mock(return_value=job))
    assert cli.execute(["dummy"], tmp_path / "log", "0", hours) == 0
    job.wait.assert_called_once_with(timeout=seconds)


@pytest.mark.parametrize("args", [["test", "jestr"], ["all", "--retest"],
                                  ["run", "jestr", "--timeout-hours", "-1"]])
def test_removed_commands_and_invalid_timeout_fail_before_launch(monkeypatch, args):
    launch = Mock()
    monkeypatch.setattr(cli, "execute", launch)
    with pytest.raises(SystemExit) as error:
        cli.main(args)
    assert error.value.code == 2
    launch.assert_not_called()


def test_run_requires_its_initialization_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    launch = Mock()
    monkeypatch.setattr(cli, "execute", launch)
    assert not cli.run(find("ladder_A_sim"), "0")
    launch.assert_not_called()
