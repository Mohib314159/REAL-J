"""Inspect task definitions.

    inspect eval realj/eval.py@realj --model anthropic/claude-sonnet-4-5 \
        -T split=dev -T rungs=1,3 -T arms=FORBIDDEN,ALLOWED,CLEAN

    inspect eval realj/eval.py@realj_calibration --model ... -T epochs=5

Everything that varies is a task parameter; nothing varies implicitly. One
sample is one (case, arm, rung, seed). Stochastic variation comes from
``epochs``, and the epoch count is *not* the replication unit -- see
``analysis/statistics.py``.

The prefix is built once, hashed, and stored in sample metadata. The rollout,
the recognition fork, and the J-lens all consume that same stored string, so
no channel can differ from another because it saw a different context.
"""

from __future__ import annotations

from typing import Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
)
from inspect_ai.tool import ToolCall

from realj.models import coding_agent
from realj.realism.rungs import (
    DEFAULT_RUNGS,
    MissingReplayError,
    RenderedRung,
    Turn,
    prefix_record,
    render,
)
from realj.realism.trajectories import pick_replay
from realj.prefix import tool_schema_hash
from realj.scorers.realism_winrate import (
    realism_win_rate_full,
    realism_win_rate_prefix,
)
from realj.scorers.recognition import recognition
from realj.scorers.reward_hack import reward_hack
from realj.scorers.task_success import legitimate_fix, visible_suite
from realj.scorers.vea import vea_early, vea_full
from realj.tasks.cases import cases_for_split, get_case
from realj.tasks.generator import ARMS, REPO_ROOT, Arm, build

SANDBOX = ("docker", "sandbox/compose.yaml")


def _parse_list(value: str | Sequence | int, cast=str) -> tuple:
    """Accept "1,3,5", [1,3,5], or a bare 1.

    Inspect coerces ``-T rungs=1`` to an int before the task function sees it,
    so a scalar has to be handled or a single-rung run raises. Found by
    actually invoking the CLI rather than by reading the docs.
    """
    if isinstance(value, (int, float)):
        return (cast(value),)
    if isinstance(value, str):
        return tuple(cast(v.strip()) for v in value.split(",") if v.strip())
    return tuple(cast(v) for v in value)


def _messages(rendered: RenderedRung) -> list[ChatMessage]:
    """Convert a rendered rung into Inspect messages.

    The rung's system prompt is carried **in the sample**, not in the agent's
    ``instructions``. An earlier version passed it through the agent, which
    meant that whenever one task mixed rungs with different system prompts the
    solver fell back to ``instructions=None`` and R4's deployment system prompt
    was silently dropped -- quietly deleting most of the deployment-resources
    manipulation while every test still passed.

    Per-sample system messages have no such failure mode: each sample carries
    its own framing whatever else is in the task.
    """
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=rendered.system_prompt)
    ]
    for turn in rendered.prefill:
        messages.extend(_turn_to_messages(turn))
    messages.append(ChatMessageUser(content=rendered.final_request))
    return messages


def _turn_to_messages(turn: Turn) -> list[ChatMessage]:
    if turn.role == "user":
        return [ChatMessageUser(content=turn.content)]
    if turn.role == "assistant":
        if turn.tool_call_id and turn.tool_name:
            return [
                ChatMessageAssistant(
                    content=turn.content,
                    tool_calls=[
                        ToolCall(
                            id=turn.tool_call_id,
                            function=turn.tool_name,
                            arguments=turn.tool_arguments or {},
                        )
                    ],
                )
            ]
        return [ChatMessageAssistant(content=turn.content)]
    if turn.role == "tool":
        return [
            ChatMessageTool(
                content=turn.content,
                tool_call_id=turn.tool_call_id,
                function=turn.tool_name,
            )
        ]
    raise ValueError(f"unknown turn role {turn.role!r}")


def build_samples(
    split: str = "dev",
    arms: Sequence[Arm] = ARMS,
    rungs: Sequence[int] = DEFAULT_RUNGS,
    seed: int = 0,
    prefill_mode: str = "tool_calls",
    skip_missing_replays: bool = True,
) -> list[Sample]:
    """One sample per (case, arm, rung)."""
    samples: list[Sample] = []
    for case in cases_for_split(split):
        for arm in arms:
            for rung in rungs:
                repo = build(case.case_id, seed=seed, arm=arm, rung=rung)
                replay = pick_replay(case.case_id, seed) if rung == 5 else None
                try:
                    rendered = render(
                        case.case_id,
                        arm=arm,
                        rung=rung,
                        repo=repo,
                        prefill_mode=prefill_mode,  # type: ignore[arg-type]
                        replay=replay,
                    )
                except MissingReplayError:
                    if skip_missing_replays:
                        continue
                    raise

                record = prefix_record(rendered)
                canonical = rendered.canonical()
                metadata = repo.as_metadata()
                metadata.update(
                    {
                        "case_split": case.split,
                        "module": case.module,
                        "bug_kind": case.bug_kind,
                        "repo_root": REPO_ROOT,
                        "rung_label": rendered.label,
                        "system_prompt": rendered.system_prompt,
                        "surface": rendered.surface.as_dict(),
                        # The canonical prefix, stored whole. Every channel
                        # rebuilds it from here rather than receiving a
                        # serialisation of it.
                        "prefix_record": record,
                        "prompt_hash": record["prompt_hash"],
                        "structural_hash": record["structural_hash"],
                        "tool_schema_hash": tool_schema_hash(canonical.tool_infos),
                        "harness": "inspect-react",
                        "target_model": None,  # set per-task below; scorers get it explicitly
                    }
                )

                samples.append(
                    Sample(
                        id=f"{case.case_id}-{arm}-R{rung}-s{seed}",
                        input=_messages(rendered),
                        files=repo.files,
                        metadata=metadata,
                        sandbox=SANDBOX,
                    )
                )
    return samples


def _solver(samples: list[Sample]):
    """One scaffold for the whole task, identical for every sample.

    ``instructions=None`` on purpose: the rung's system prompt reaches the
    model as the sample's own system message (see ``_messages``), so the
    scaffold contributes only the parts that must be constant across rungs --
    the assistant prompt and the submit prompt, both stripped of
    evaluation-flavoured wording in ``realj/models.py``.
    """
    del samples  # the scaffold does not depend on the dataset, and must not
    return coding_agent(instructions=None)


def _scorers(
    with_recognition: bool,
    with_realism: bool,
    judge: str | None,
    target_model: str | None = None,
    require_harness_match: bool = False,
) -> list:
    """Scorer set.

    Scientific parameters are passed to the scorers **explicitly**. They used
    to be read out of optional sample metadata, which meant the frozen target
    model never reached the VEA tokenizer (so the "512-token" window silently
    stayed a 2048-character approximation) and the realism provenance gate
    never fired. Unit tests could not catch that: the helpers worked, the
    execution path just never supplied the argument.

    ``judge`` grades VEA and realism -- a third party, and it should be. But
    ``recognition`` measures the **target's own** recognition, so it is passed
    ``target_model`` (None = the eval's active model), never the judge. Handing
    the judge to the recognition probe would measure the judge's opinion of the
    context and label it Q.
    """
    scorers = [
        reward_hack(),
        legitimate_fix(),
        visible_suite(),
        vea_early(judge=judge, tokenizer_name=target_model),  # PRIMARY
        vea_full(judge=judge),  # secondary
    ]
    if with_recognition:
        scorers.append(recognition(model=target_model))
    if with_realism:
        scorers.append(
            realism_win_rate_prefix(
                judge=judge, require_harness_match=require_harness_match
            )
        )  # PRIMARY
        scorers.append(
            realism_win_rate_full(
                judge=judge, require_harness_match=require_harness_match
            )
        )  # secondary
    return scorers


@task
def realj(
    split: str = "dev",
    arms: str = "FORBIDDEN,ALLOWED,CLEAN",
    rungs: str = "1,3",
    seed: int = 0,
    epochs: int = 1,
    judge: str | None = None,
    target_model: str | None = None,
    prefill_mode: str = "tool_calls",
    with_recognition: bool = True,
    with_realism: bool = True,
    require_harness_match: bool = False,
) -> Task:
    """The main task.

    ``split="dev"`` for calibration, ``split="heldout"`` after the
    preregistration is frozen. Running ``heldout`` before then is the failure
    mode this whole layout exists to prevent.
    """
    samples = build_samples(
        split=split,
        arms=_parse_list(arms),  # type: ignore[arg-type]
        rungs=_parse_list(rungs, int),
        seed=seed,
        prefill_mode=prefill_mode,
    )
    if not samples:
        raise ValueError(
            f"no samples for split={split!r} rungs={rungs!r}. If R5 is "
            "requested, deployment_refs/replays/ is probably empty; see "
            "`python -m realj.realism.trajectories record`."
        )

    return Task(
        dataset=samples,
        solver=_solver(samples),
        scorer=_scorers(
            with_recognition, with_realism, judge, target_model, require_harness_match
        ),
        sandbox=SANDBOX,
        epochs=epochs,
        message_limit=40,
        fail_on_error=0.2,
        metadata={
            "split": split,
            "arms": arms,
            "rungs": rungs,
            "seed": seed,
            "prefill_mode": prefill_mode,
        },
    )


@task
def realj_by_rung(
    rung: int = 3,
    split: str = "dev",
    arms: str = "FORBIDDEN,ALLOWED,CLEAN",
    seed: int = 0,
    epochs: int = 1,
    judge: str | None = None,
    target_model: str | None = None,
    prefill_mode: str = "tool_calls",
    require_harness_match: bool = False,
) -> Task:
    """One rung per task invocation.

    Preferred for the real runs: the system prompt is then constant within a
    task, so it is applied through the scaffold rather than smuggled into the
    message list, and per-rung logs stay separable.
    """
    samples = build_samples(
        split=split,
        arms=_parse_list(arms),  # type: ignore[arg-type]
        rungs=(rung,),
        seed=seed,
        prefill_mode=prefill_mode,
    )
    if not samples:
        raise ValueError(f"no samples for rung {rung} on split {split!r}")

    return Task(
        dataset=samples,
        solver=_solver(samples),
        scorer=_scorers(True, True, judge, target_model, require_harness_match),
        sandbox=SANDBOX,
        epochs=epochs,
        message_limit=40,
        fail_on_error=0.2,
        metadata={"rung": rung, "split": split, "seed": seed},
    )


@task
def realj_calibration(
    rungs: str = "1,3",
    seed: int = 0,
    epochs: int = 5,
    judge: str | None = None,
) -> Task:
    """Dev-set calibration. Behavioural channels only -- fast and cheap.

    Purpose: find out whether the temptation is strong enough to be
    non-degenerate before spending anything on the held-out set. Adjusting the
    scenarios here is allowed and expected. Adjusting them after looking at
    held-out results is not, which is why held-out cases are a different split
    and the preregistration is a committed file.
    """
    samples = build_samples(
        split="dev", arms=ARMS, rungs=_parse_list(rungs, int), seed=seed
    )
    return Task(
        dataset=samples,
        solver=_solver(samples),
        scorer=_scorers(False, False, judge, None),
        sandbox=SANDBOX,
        epochs=epochs,
        message_limit=40,
        fail_on_error=0.5,
        metadata={"phase": "calibration", "seed": seed},
    )


def describe(split: str = "all") -> list[dict]:
    """What would be run, without running it. Useful before spending money."""
    out = []
    for case in cases_for_split(split):
        c = get_case(case.case_id)
        out.append(
            {
                "case_id": c.case_id,
                "split": c.split,
                "module": c.module,
                "bug_kind": c.bug_kind,
                "impossible_test": c.impossible.test_name,
                "partner_test": c.impossible.partner_test,
            }
        )
    return out
