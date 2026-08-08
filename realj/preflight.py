"""Fail-closed checks before a confirmatory run.

``configs/heldout.yaml`` was documentation, not configuration. Nothing loaded
it, so the frozen model, the frozen settings and the preregistration commit
could all be silently ignored -- and the README cheerfully told the reader to
run the held-out split against a different model than the config named. A
preregistration that cannot stop you is a note to yourself.

This module fails, it does not warn. Run it before the confirmatory eval:

    python -m realj.preflight configs/heldout.yaml

Generation settings themselves belong in an Inspect run-config
(``configs/heldout-run.yaml``), which Inspect applies natively. This file
checks what Inspect cannot know about: whether the preregistration is frozen,
whether the artifacts the analysis depends on exist, and whether the run about
to happen is the run that was preregistered.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str


def _load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ImportError:
        # Minimal top-level parser so preflight works without PyYAML. Nested
        # blocks are returned as raw strings; the checks below only need
        # top-level keys and a handful of nested ones read via _get.
        out: dict = {}
        stack: list[tuple[int, dict]] = [(-1, out)]
        for line in text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            indent = len(line) - len(line.lstrip())
            if ":" not in line:
                continue
            key, _, value = line.strip().partition(":")
            value = value.split("#")[0].strip()
            while stack and stack[-1][0] >= indent:
                stack.pop()
            parent = stack[-1][1]
            if value == "":
                child: dict = {}
                parent[key.strip()] = child
                stack.append((indent, child))
            else:
                if value in ("null", "~", ""):
                    parsed = None
                elif value in ("true", "false"):
                    parsed = value == "true"
                else:
                    parsed = value.strip("'\"")
                parent[key.strip()] = parsed
        return out


def _get(config: dict, *path, default=None):
    node = config
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def _git_clean() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out == ""
    except Exception:
        return False


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def run_checks(config: dict, root: Path) -> list[Check]:
    checks: list[Check] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(Check(name, bool(passed), detail))

    confirmatory = _get(config, "phase") == "confirmatory"

    # --- preregistration -------------------------------------------------
    commit = _get(config, "preregistration_commit")
    add(
        "preregistration_commit",
        bool(commit),
        "set it in the config and commit before running"
        if not commit
        else f"frozen at {commit}",
    )
    head = _git_commit()
    if commit and head:
        # This previously passed True unconditionally and merely printed
        # "differs" -- the exact inverse of what preflight exists to do.
        add(
            "working_tree_matches_prereg",
            str(head).startswith(str(commit)) or str(commit).startswith(str(head)),
            f"HEAD={head[:12]}, prereg={str(commit)[:12]}",
        )
    add(
        "working_tree_clean",
        _git_clean(),
        "uncommitted changes: the run would not match any committed state",
    )

    # --- model identity --------------------------------------------------
    model = _get(config, "model")
    target = _get(config, "target_model")
    add("model_specified", bool(model), str(model))
    add(
        "target_is_the_mechanistic_model",
        bool(target) and target == model,
        f"model={model} target_model={target}; Q, J, V and B must come from one model",
    )
    add("model_revision_frozen", bool(_get(config, "model_revision")), "pin the revision")
    add("judge_specified", bool(_get(config, "judge")), str(_get(config, "judge")))
    add(
        "thinking_mode_frozen",
        _get(config, "enable_thinking") is not None,
        "VEA needs an accessible reasoning trace; do not rely on a template default",
    )
    add("temperature_frozen", _get(config, "temperature") is not None, "")
    add("max_tokens_frozen", _get(config, "max_tokens") is not None, "")

    # --- matched state ---------------------------------------------------
    report = root / "out" / "matched_state.json"
    passed = False
    detail = f"missing {report}; run the matched-state gate on the target model"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        passed = bool(data.get("passed")) and data.get("hash_kind") == "token_ids"
        detail = f"passed={data.get('passed')} hash_kind={data.get('hash_kind')}"
    add("matched_state_certified", passed, detail)

    # --- deployment references -------------------------------------------
    from realj.realism.trajectories import bank_status

    status = bank_status()
    minimum = int(_get(config, "realism", "min_references", default=10) or 10)
    add(
        "realism_test_references",
        status["realism_test"] >= minimum,
        f"{status['realism_test']} references, need {minimum}",
    )
    add(
        "reference_provenance_recorded",
        status["provenance_recorded"]["realism_test"] == status["realism_test"]
        and status["realism_test"] > 0,
        "every reference needs model and harness recorded",
    )

    rungs = str(_get(config, "rungs") or "")
    if "5" in rungs:
        add("r5_replays_present", status["replays"] > 0, "R5 needs recorded sessions")
        add(
            "r5_repo_snapshots_present",
            status["replays"] > 0
            and status["replays_with_repo_snapshot"] == status["replays"],
            "every replay needs repo_snapshot_hash for the consistency check",
        )

    # --- J-lens ----------------------------------------------------------
    if _get(config, "jlens", "enabled"):
        add("jlens_band_frozen", bool(_get(config, "jlens", "band_fraction")), "run select_band")
        for key, path_key in (
            ("jlens_token_ids_frozen", "frozen_token_ids"),
            ("jlens_validation_report", "validation_report"),
        ):
            path = _get(config, "jlens", path_key)
            add(key, bool(path) and (root / str(path)).exists(), str(path))
        report_path = _get(config, "jlens", "validation_report")
        if report_path and (root / str(report_path)).exists():
            data = json.loads((root / str(report_path)).read_text(encoding="utf-8"))
            add("jlens_validation_passed", bool(data.get("passed")), str(data.get("failures")))

    # --- parity with the Inspect run config -------------------------------
    run_path = root / "configs" / f"heldout-run-r{_get(config, 'rung', default='1')}.yaml"
    if not run_path.exists():
        run_path = root / "configs" / "heldout-run-r1.yaml"
    if run_path.exists():
        run_cfg = _load(run_path)
        pairs = [
            ("model", _get(config, "model"), _get(run_cfg, "model", "model")),
            (
                "temperature",
                _get(config, "temperature"),
                _get(run_cfg, "generate_config", "temperature"),
            ),
            (
                "max_tokens",
                _get(config, "max_tokens"),
                _get(run_cfg, "generate_config", "max_tokens"),
            ),
            ("epochs", _get(config, "epochs"), _get(run_cfg, "eval_config", "epochs")),
            (
                "judge",
                _get(config, "judge"),
                _get(run_cfg, "task", "args", "judge"),
            ),
            (
                "target_model",
                _get(config, "target_model"),
                _get(run_cfg, "task", "args", "target_model"),
            ),
        ]
        for name, a, b in pairs:
            if a is None and b is None:
                continue
            add(
                f"runconfig_parity_{name}",
                str(a) == str(b),
                f"{run_path.name}: {b!r} vs heldout.yaml: {a!r}",
            )
    else:
        add("runconfig_present", False, "no configs/heldout-run-r*.yaml found")

    if not confirmatory:
        checks = [c for c in checks if c.name != "preregistration_commit" or c.passed]
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="realj.preflight")
    parser.add_argument("config", type=Path)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)

    config = _load(args.config)
    checks = run_checks(config, args.root)

    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"{'PASS' if c.passed else 'FAIL'}  {c.name:<{width}}  {c.detail}")

    failed = [c.name for c in checks if not c.passed]
    if failed:
        print(f"\n{len(failed)} check(s) failed. Not ready for a confirmatory run.")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
