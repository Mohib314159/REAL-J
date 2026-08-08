"""The wiring between the ladder and Inspect.

Both tests here exist because of a bug that got through everything else.

The system-prompt test: an earlier version passed each rung's system prompt
through the agent's ``instructions``. When one task mixed rungs with different
system prompts, the solver fell back to ``instructions=None`` and R4's
deployment system prompt was **silently dropped** -- deleting most of the
deployment-resources manipulation while every other test still passed. It is
now carried in the sample, and asserted here.

The task-arg test: Inspect coerces ``-T rungs=1`` to an int before the task
function sees it, so a single-rung run raised ``TypeError``. Found by
invoking the CLI, not by reading the code.
"""

from __future__ import annotations

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from realj.eval import _parse_list, build_samples, describe
from realj.realism.rungs import RUNG_LABELS
from realj.tasks.cases import CASES

RUNGS = (1, 2, 3, 4)


@pytest.mark.parametrize("rung", RUNGS)
@pytest.mark.parametrize("arm", ["FORBIDDEN", "ALLOWED", "CLEAN"])
def test_system_prompt_reaches_the_model(rung, arm):
    sample = build_samples(split="dev", rungs=(rung,), arms=(arm,))[0]
    head = sample.input[0]
    assert isinstance(head, ChatMessageSystem), (
        f"R{rung}/{arm}: the first message is {type(head).__name__}, not a "
        f"system message. The rung's framing is not reaching the model."
    )
    assert head.text.strip(), f"R{rung}/{arm}: empty system prompt"
    assert head.text == sample.metadata["system_prompt"]


def test_deployment_system_prompt_is_actually_different():
    """R4's whole point is the deployment-shaped system prompt."""
    low = build_samples(split="dev", rungs=(1,), arms=("FORBIDDEN",))[0]
    high = build_samples(split="dev", rungs=(4,), arms=("FORBIDDEN",))[0]
    assert high.input[0].text != low.input[0].text
    assert len(high.input[0].text) > 3 * len(low.input[0].text), (
        "R4's system prompt is barely longer than R1's; the deployment "
        "resources manipulation is not doing anything."
    )


@pytest.mark.parametrize("rung", RUNGS)
def test_last_message_is_the_request(rung):
    sample = build_samples(split="dev", rungs=(rung,), arms=("FORBIDDEN",))[0]
    assert isinstance(sample.input[-1], ChatMessageUser)


def test_prefill_grows_with_rung():
    counts = {
        rung: len(build_samples(split="dev", rungs=(rung,), arms=("FORBIDDEN",))[0].input)
        for rung in RUNGS
    }
    assert counts[1] == counts[2], "R1 and R2 should both be history-free"
    assert counts[3] > counts[2], "R3 should add a prior session"
    assert counts[4] > counts[3], "R4 should add to R3's session"


@pytest.mark.parametrize(
    "value,cast,expected",
    [
        (1, int, (1,)),
        ("1,3,5", int, (1, 3, 5)),
        ([1, 3], int, (1, 3)),
        ("CLEAN", str, ("CLEAN",)),
        ("FORBIDDEN,ALLOWED", str, ("FORBIDDEN", "ALLOWED")),
        ("1, 3 ,5", int, (1, 3, 5)),
    ],
)
def test_task_args_accept_scalars_and_strings(value, cast, expected):
    assert _parse_list(value, cast) == expected


def test_sample_ids_are_unique_and_legible():
    samples = build_samples(split="heldout", rungs=(1, 3))
    ids = [s.id for s in samples]
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith("case_")
    assert "-R1-" in ids[0]


def test_every_sample_carries_its_provenance():
    for sample in build_samples(split="dev", rungs=(1, 3)):
        meta = sample.metadata
        for key in ("case_id", "arm", "rung", "code_hash", "repo_hash", "prompt_hash"):
            assert meta.get(key), f"{sample.id} missing {key}"
        assert meta["prefix_record"], f"{sample.id} has no stored prefix"
        assert meta["prefix_record"]["messages"], "stored prefix has no messages"
        assert meta["tool_schema_hash"], f"{sample.id} missing tool_schema_hash"
        assert meta["rung_label"] == RUNG_LABELS[meta["rung"]]


def test_r5_is_skipped_not_faked_when_no_replays_exist():
    with_r5 = build_samples(split="dev", rungs=(1, 3, 5))
    assert all(s.metadata["rung"] != 5 for s in with_r5), (
        "an R5 sample was constructed with no recorded trajectory available"
    )


def test_describe_lists_every_case():
    assert len(describe("all")) == len(CASES)
