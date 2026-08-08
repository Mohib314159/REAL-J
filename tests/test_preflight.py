"""Preflight must fail closed.

configs/heldout.yaml was documentation: nothing loaded it, so the frozen model
and the preregistration commit could be silently ignored -- and the README told
the reader to run held-out against a different model than the config named.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from realj.preflight import _load, run_checks

CONFIG = Path("configs/heldout.yaml")


def _names(checks):
    return {c.name: c for c in checks}


def test_shipped_config_is_not_runnable_as_is():
    """Nothing is recorded yet, so a confirmatory run must be refused."""
    checks = run_checks(_load(CONFIG), Path.cwd())
    assert any(not c.passed for c in checks)


def test_missing_preregistration_commit_fails():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    assert not checks["preregistration_commit"].passed


def test_mismatched_target_model_fails():
    config = _load(CONFIG)
    config["target_model"] = "anthropic/claude-sonnet-4-5"
    checks = _names(run_checks(config, Path.cwd()))
    assert not checks["target_is_the_mechanistic_model"].passed, (
        "a config naming Qwen as the model and Claude as the target must be "
        "refused; that is two organisms, not a dissociation"
    )


def test_matching_target_passes():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    assert checks["target_is_the_mechanistic_model"].passed


def test_matched_state_certificate_required():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    assert not checks["matched_state_certified"].passed


def test_empty_reference_bank_fails():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    assert not checks["realism_test_references"].passed
    assert not checks["reference_provenance_recorded"].passed


def test_r5_requires_replays_with_snapshots():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    assert not checks["r5_replays_present"].passed
    assert not checks["r5_repo_snapshots_present"].passed


def test_thinking_mode_and_sampling_are_frozen():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    for name in ("thinking_mode_frozen", "temperature_frozen", "max_tokens_frozen"):
        assert checks[name].passed, name


def test_jlens_gates_only_apply_when_enabled():
    config = _load(CONFIG)
    assert not config["jlens"]["enabled"]
    assert "jlens_band_frozen" not in _names(run_checks(config, Path.cwd()))

    config["jlens"]["enabled"] = True
    checks = _names(run_checks(config, Path.cwd()))
    assert not checks["jlens_band_frozen"].passed
    assert not checks["jlens_token_ids_frozen"].passed


@pytest.mark.parametrize("rung", [1, 3, 5])
def test_run_config_is_nested_inspect_schema(rung):
    """Flat keys would be silently ignored by Inspect."""
    cfg = _load(Path(f"configs/heldout-run-r{rung}.yaml"))
    assert cfg["task"]["task"] == "realj/eval.py@realj_by_rung"
    assert cfg["task"]["args"]["rung"] == rung
    assert cfg["task"]["args"]["split"] == "heldout"
    assert cfg["model"]["model"] == "hf/Qwen/Qwen3-14B"
    assert cfg["model"]["args"]["enable_thinking"] is True
    assert cfg["generate_config"]["temperature"] == 0.6
    assert cfg["eval_config"]["epochs"] == 5


def test_run_config_parity_is_checked():
    checks = _names(run_checks(_load(CONFIG), Path.cwd()))
    assert checks["runconfig_parity_model"].passed
    assert checks["runconfig_parity_target_model"].passed


def test_parity_catches_drift():
    config = _load(CONFIG)
    config["temperature"] = 1.0
    checks = _names(run_checks(config, Path.cwd()))
    assert not checks["runconfig_parity_temperature"].passed


def test_dirty_tree_is_reported():
    assert "working_tree_clean" in _names(run_checks(_load(CONFIG), Path.cwd()))
