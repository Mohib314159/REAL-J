"""Genuine agentic trajectories: R5 replays and the deployment reference bank.

Two distinct uses, and conflating them would invalidate the realism metric:

``deployment_refs/replays/``
    Trajectories replayed as the R5 prefix. LURE (Ivanov & Africa, 2605.26438)
    is the three-step recipe: collect a real session on a real repository,
    adapt it with minimal targeted edits, append the alignment-relevant
    request. The adapted transcript is fed back as prefill so the model
    experiences an in-progress session.

``deployment_refs/realism_train/`` and ``realism_test/``
    Genuine benign sessions used as the *comparison* side of the realism win
    rate. Held out from each other so that hill-climbing on realism does not
    contaminate the reported number.

**A trajectory must never be both.** ``assert_disjoint()`` enforces it: if the
transcript being replayed is also in the comparison bank, the judge is being
asked to distinguish a transcript from itself and the win rate is meaningless.

None of this can be synthesised. A fabricated "genuine session" is exactly the
artefact the ladder is trying to escape, and R5 raises rather than falling back
to one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Literal

REPLAY_SCHEMA_VERSION = 1

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "deployment_refs"


@dataclass(frozen=True)
class ReplayTurn:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict | None = None


@dataclass(frozen=True)
class ReplayTrajectory:
    """A recorded, genuinely benign agentic coding session."""

    trajectory_id: str
    source: str  # e.g. "claude-code", "inspect-react", "codex"
    repository: str  # repository the session actually ran against
    recorded_at: str  # ISO 8601 date
    system_prompt: str
    turns: tuple[ReplayTurn, ...]
    adaptations: tuple[str, ...] = ()  # every edit made, one line each
    notes: str = ""

    #: Model and harness the session was actually recorded with. If the
    #: reference bank is Claude Code sessions and the target is Qwen inside
    #: Inspect, the realism judge can separate them on house style alone and
    #: the "realism" number is measuring provenance, not deployment-likeness.
    model: str = ""
    harness: str = ""

    #: Hash of the repository state the recorded tool outputs actually refer to.
    #: R5 appends a REAL-J task to this trajectory, so if the prior tool results
    #: describe a different tree than the one the agent is now handed, the rung
    #: is not one coherent world -- it is two spliced together, which is exactly
    #: the incoherence the ladder claims to remove.
    repo_snapshot_hash: str = ""

    @property
    def content_hash(self) -> str:
        blob = json.dumps(
            {
                "system_prompt": self.system_prompt,
                "turns": [asdict(t) for t in self.turns],
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def as_transcript(self) -> str:
        """Flat rendering, for the realism-win-rate judge."""
        parts = [f"<system>\n{self.system_prompt}\n</system>"]
        for t in self.turns:
            if t.role == "tool":
                parts.append(f"<tool_result name={t.tool_name}>\n{t.content}\n</tool_result>")
            else:
                parts.append(f"<{t.role}>\n{t.content}\n</{t.role}>")
        return "\n\n".join(parts)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["schema_version"] = REPLAY_SCHEMA_VERSION
        payload["content_hash"] = self.content_hash
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, blob: str) -> "ReplayTrajectory":
        raw = json.loads(blob)
        version = raw.pop("schema_version", 0)
        if version != REPLAY_SCHEMA_VERSION:
            raise ValueError(
                f"trajectory schema version {version} != {REPLAY_SCHEMA_VERSION}; "
                "re-record or migrate"
            )
        raw.pop("content_hash", None)
        turns = tuple(ReplayTurn(**t) for t in raw.pop("turns"))
        adaptations = tuple(raw.pop("adaptations", ()))
        return cls(turns=turns, adaptations=adaptations, **raw)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _load_dir(path: Path) -> list[ReplayTrajectory]:
    if not path.is_dir():
        return []
    out = []
    for f in sorted(path.glob("*.json")):
        out.append(ReplayTrajectory.from_json(f.read_text(encoding="utf-8")))
    return out


def load_replays(root: Path | None = None) -> list[ReplayTrajectory]:
    return _load_dir((root or DEFAULT_ROOT) / "replays")


def load_reference_bank(
    split: Literal["realism_train", "realism_test"], root: Path | None = None
) -> list[ReplayTrajectory]:
    return _load_dir((root or DEFAULT_ROOT) / split)


def pick_replay(
    case_id: str, seed: int, root: Path | None = None
) -> ReplayTrajectory | None:
    """Deterministically assign a recorded session to a case.

    Deterministic rather than random so that R5 for a given case is stable
    across runs and the ``prompt_hash`` means something.
    """
    replays = load_replays(root)
    if not replays:
        return None
    key = int(hashlib.sha256(f"{case_id}:{seed}".encode()).hexdigest(), 16)
    return replays[key % len(replays)]


def assert_disjoint(root: Path | None = None) -> None:
    """A replayed trajectory must not also be a comparison reference."""
    replay_hashes = {t.content_hash: t.trajectory_id for t in load_replays(root)}
    for split in ("realism_train", "realism_test"):
        for ref in load_reference_bank(split, root):  # type: ignore[arg-type]
            if ref.content_hash in replay_hashes:
                raise ValueError(
                    f"trajectory {ref.trajectory_id!r} appears both in {split} and "
                    f"in replays (as {replay_hashes[ref.content_hash]!r}). The "
                    "realism win rate would be comparing a transcript to itself."
                )
    train = {t.content_hash for t in load_reference_bank("realism_train", root)}
    test = {t.content_hash for t in load_reference_bank("realism_test", root)}
    overlap = train & test
    if overlap:
        raise ValueError(
            f"{len(overlap)} trajectories appear in both realism_train and "
            "realism_test; the held-out realism number would be contaminated."
        )


def assert_replay_repo_consistent(
    trajectory: ReplayTrajectory, code_hash: str, strict: bool = True
) -> str:
    """R5's recorded session must refer to the repository being evaluated.

    Returns "" when consistent, otherwise a description. With ``strict`` the
    inconsistency raises: a replay whose tool outputs describe another tree is
    not a more realistic version of the task, it is a different task with a
    confusing preamble.
    """
    if not trajectory.repo_snapshot_hash:
        message = (
            f"trajectory {trajectory.trajectory_id!r} has no repo_snapshot_hash, "
            "so there is no way to check that its recorded tool output refers to "
            "the repository being evaluated"
        )
    elif trajectory.repo_snapshot_hash != code_hash:
        message = (
            f"trajectory {trajectory.trajectory_id!r} was recorded against "
            f"repo {trajectory.repo_snapshot_hash[:12]} but the task ships "
            f"{code_hash[:12]}; the prior tool results describe a different tree"
        )
    else:
        return ""
    if strict:
        raise ValueError(message)
    return message


def harness_mismatch(
    references: list[ReplayTrajectory], model: str, harness: str
) -> str:
    """Warn when the reference bank does not match the target distribution.

    Not fatal -- with a small bank you may have no choice -- but it must be
    recorded, because a judge that can tell Claude Code from Inspect-plus-Qwen
    will separate them regardless of how realistic either environment is.
    """
    if not references or not model:
        return ""
    models = {r.model for r in references if r.model}
    harnesses = {r.harness for r in references if r.harness}
    problems = []
    if models and model not in models:
        problems.append(f"references are {sorted(models)}, target is {model!r}")
    if harnesses and harness and harness not in harnesses:
        problems.append(f"reference harnesses {sorted(harnesses)}, target {harness!r}")
    if not models and not harnesses:
        problems.append("references record no model/harness provenance at all")
    return "; ".join(problems)


def bank_status(root: Path | None = None) -> dict:
    root = root or DEFAULT_ROOT
    replays = load_replays(root)
    return {
        "replays": len(replays),
        "replays_with_repo_snapshot": sum(1 for r in replays if r.repo_snapshot_hash),
        "realism_train": len(load_reference_bank("realism_train", root)),
        "realism_test": len(load_reference_bank("realism_test", root)),
        "provenance_recorded": {
            split: sum(
                1 for r in load_reference_bank(split, root) if r.model and r.harness
            )
            for split in ("realism_train", "realism_test")
        },
    }


# --------------------------------------------------------------------------
# Recording helper
# --------------------------------------------------------------------------

_RECORD_HELP = """\
Recording a genuine session
===========================

R5 and the deployment reference bank both need real sessions. There is no way
to generate them here; you have to run an agent on a real repository doing
ordinary work and keep the log.

1. Take a sandboxed copy of a repository you control.
2. Run your usual coding agent on genuinely benign tasks -- add a unit test,
   refactor a plotting helper, fix a docstring, add a config option,
   investigate a bug you already understand.
3. Export the session (system prompt, user turns, assistant turns, tool calls
   and tool results) and convert it with `convert` below.
4. Split the results: roughly half into realism_train/, half into
   realism_test/. Replays go in replays/ and must not appear in either bank.

Record the adaptations. Every edit you make to a recorded transcript goes in
`adaptations` as one line. If that list is long, the trajectory is no longer a
replay -- it is a synthetic scenario wearing a replay's clothes, and you should
say so in the write-up.
"""


def convert(
    raw_messages: Iterable[dict],
    trajectory_id: str,
    source: str,
    repository: str,
    recorded_at: str,
    system_prompt: str,
    adaptations: Iterable[str] = (),
    notes: str = "",
    model: str = "",
    harness: str = "",
    repo_snapshot_hash: str = "",
) -> ReplayTrajectory:
    """Convert a list of {role, content, ...} dicts into a ReplayTrajectory."""
    turns = tuple(
        ReplayTurn(
            role=m["role"],
            content=m.get("content", ""),
            tool_call_id=m.get("tool_call_id"),
            tool_name=m.get("tool_name"),
            tool_arguments=m.get("tool_arguments"),
        )
        for m in raw_messages
    )
    return ReplayTrajectory(
        trajectory_id=trajectory_id,
        source=source,
        repository=repository,
        recorded_at=recorded_at,
        system_prompt=system_prompt,
        turns=turns,
        adaptations=tuple(adaptations),
        notes=notes,
        model=model,
        harness=harness,
        repo_snapshot_hash=repo_snapshot_hash,
    )


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="realj.realism.trajectories")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("record", help="how to record a genuine session")
    sub.add_parser("status", help="count what is in the bank")
    sub.add_parser("check", help="verify replays and reference banks are disjoint")
    args = parser.parse_args(argv)

    if args.cmd == "record":
        print(_RECORD_HELP)
    elif args.cmd == "status":
        print(json.dumps(bank_status(), indent=2))
    elif args.cmd == "check":
        assert_disjoint()
        print("ok: replays and reference banks are disjoint")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
