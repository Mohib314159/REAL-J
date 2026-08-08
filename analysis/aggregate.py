"""Inspect logs -> flat result rows.

The job here is unglamorous and load-bearing: pull every channel out of the
eval log into one row per rollout, with the provenance hashes intact, so that
``statistics.py`` and ``figures.py`` never have to reach back into the log
format.

Reading an Inspect log gives a list of samples, each with a ``scores`` dict
keyed by scorer name. The scorer names here are the function names in
``realj/scorers/``: ``reward_hack``, ``legitimate_fix``, ``visible_suite``,
``vea``, ``recognition``, ``realism_win_rate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from realj.logging.schema import ResultRow, ResultWriter

SCORER_KEYS = (
    "reward_hack",
    "legitimate_fix",
    "visible_suite",
    "vea_early",  # PRIMARY
    "vea_full",  # secondary
    "recognition",
    "realism_win_rate_prefix",  # PRIMARY
    "realism_win_rate_full",  # secondary
)


def _score_value(sample: Any, key: str):
    scores = getattr(sample, "scores", None) or {}
    score = scores.get(key)
    return None if score is None else score.value


def _score_meta(sample: Any, key: str) -> dict:
    scores = getattr(sample, "scores", None) or {}
    score = scores.get(key)
    return dict(getattr(score, "metadata", None) or {}) if score else {}


def _as_bool(value) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value >= 0.5
    if isinstance(value, str):
        return value.strip().upper() in {"C", "1", "TRUE", "YES"}
    return None


def _transcript_hash(sample: Any) -> str:
    parts = []
    for message in getattr(sample, "messages", []) or []:
        parts.append(f"{getattr(message, 'role', '')}:{getattr(message, 'text', '')}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def row_from_sample(sample: Any, model: str, git_commit: str | None = None) -> ResultRow:
    meta = dict(getattr(sample, "metadata", None) or {})
    surface = dict(meta.get("surface") or {})

    vea_early_meta = _score_meta(sample, "vea_early")
    vea_full_meta = _score_meta(sample, "vea_full")
    rec_meta = _score_meta(sample, "recognition")
    hack_meta = _score_meta(sample, "reward_hack")

    return ResultRow(
        sample_id=str(getattr(sample, "id", "")),
        case_id=meta.get("case_id", ""),
        case_split=meta.get("case_split", ""),
        seed=int(meta.get("seed", 0)),
        model=model,
        arm=meta.get("arm", ""),
        rung=int(meta.get("rung", 0)),
        repo_hash=meta.get("repo_hash", ""),
        code_hash=meta.get("code_hash", ""),
        prompt_hash=meta.get("prompt_hash", ""),
        transcript_hash=_transcript_hash(sample),
        git_commit=git_commit,
        context_chars=surface.get("context_chars"),
        num_messages=surface.get("num_messages"),
        num_tool_results=surface.get("num_tool_results"),
        num_files=surface.get("num_files"),
        legitimate_fix_success=_as_bool(_score_value(sample, "legitimate_fix")),
        reward_hack=_as_bool(_score_value(sample, "reward_hack")),
        visible_suite_passes=_as_bool(_score_value(sample, "visible_suite")),
        modified_test_paths=list(hack_meta.get("touched_paths", [])),
        vea_early_level=vea_early_meta.get("vea_early_level"),
        vea_early_evidence=vea_early_meta.get("vea_early_evidence"),
        vea_early_cut_reason=vea_early_meta.get("vea_early_cut_reason"),
        vea_early_window_exact=vea_early_meta.get("vea_early_window_exact"),
        vea_full_level=vea_full_meta.get("vea_full_level"),
        vea_full_evidence=vea_full_meta.get("vea_full_evidence"),
        recognition_p_eval=rec_meta.get("recognition_p_eval"),
        recognition_p_eval_sd=rec_meta.get("recognition_p_eval_sd"),
        recognition_q_source=rec_meta.get("recognition_q_source"),
        recognition_order_disagreement=rec_meta.get("recognition_order_disagreement"),
        realism_win_rate_prefix=_score_value(sample, "realism_win_rate_prefix"),
        realism_win_rate_full=_score_value(sample, "realism_win_rate_full"),
        realism_harness_mismatch=_score_meta(
            sample, "realism_win_rate_prefix"
        ).get("realism_harness_mismatch"),
        prefix_hash=meta.get("structural_hash"),
        hash_kind="structural",
        tool_schema_hash=meta.get("tool_schema_hash"),
        harness=meta.get("harness"),
        error=str(getattr(sample, "error", "") or "") or None,
        notes={"rung_label": meta.get("rung_label"), "module": meta.get("module")},
    )


def rows_from_log(log_path: str | Path, git_commit: str | None = None) -> list[ResultRow]:
    from inspect_ai.log import read_eval_log

    log = read_eval_log(str(log_path))
    model = getattr(getattr(log, "eval", None), "model", "unknown")
    return [
        row_from_sample(sample, model=model, git_commit=git_commit)
        for sample in (log.samples or [])
    ]


def attach_jlens(rows: Iterable[ResultRow], scores_path: str | Path) -> list[ResultRow]:
    """Merge J-lens scores in by ``prompt_hash``.

    The lens runs separately, on a different machine, possibly days later. It
    keys on the prefix hash because that is the only thing the two runs share
    by construction.
    """
    scores = json.loads(Path(scores_path).read_text(encoding="utf-8"))
    by_hash = {s["prompt_hash"]: s for s in scores if s.get("prompt_hash")}
    out = []
    for row in rows:
        entry = by_hash.get(row.prompt_hash)
        if entry:
            row.jlens_prefix_score = entry.get("jlens_eval_score")
            row.jlens_preaction_peak = entry.get("jlens_preaction_peak")
            row.jlens_preaction_mean = entry.get("jlens_preaction_mean")
            # If the lens ran on the same model with a tokenizer, upgrade the
            # prefix guarantee from structural to token-level.
            if entry.get("token_hash"):
                row.prefix_hash = entry["token_hash"]
                row.hash_kind = "token_ids"
                row.n_prefix_tokens = entry.get("n_prefix_tokens")
        out.append(row)
    return out


def _main(argv: list[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(prog="analysis.aggregate")
    parser.add_argument("logs", nargs="+", help="Inspect .eval log files")
    parser.add_argument("--out", default="out/results.jsonl")
    parser.add_argument("--jlens", help="JSON list of {prompt_hash, jlens_eval_score}")
    parser.add_argument("--git-commit")
    args = parser.parse_args(argv)

    rows: list[ResultRow] = []
    for log in args.logs:
        rows.extend(rows_from_log(log, git_commit=args.git_commit))
    if args.jlens:
        rows = attach_jlens(rows, args.jlens)

    with ResultWriter(args.out) as writer:
        writer.write_many(rows)
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
