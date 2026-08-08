"""The shared-prefix claim must be checkable, not asserted.

The README used to state that the rollout, the recognition fork and the J-lens
all consumed the same stored prefix. They did not: the rollout received
structured Inspect messages plus tool schemas, while the other two received a
hand-written XML serialisation. ``P_rollout != P_Q = P_J``, in a study whose
entire subject is whether small contextual cues tell a model it is being
observed.

These tests make the claim mechanical. If the branches ever diverge again, this
file fails rather than the write-up.
"""

from __future__ import annotations

import pytest
from inspect_ai.model import ChatMessageSystem, ChatMessageUser

from realj.models import agent_tools
from realj.prefix import (
    CanonicalPrefix,
    PrefixMessage,
    assert_same_prefix,
    tool_infos,
    tool_schema_hash,
)
from realj.realism.rungs import prefix_from_record, prefix_record, render
from realj.tasks.cases import CASES
from realj.tasks.generator import build

CASE_IDS = sorted(CASES)
RUNGS = (1, 2, 3, 4)


def _canonical(case_id="case_01", arm="FORBIDDEN", rung=3):
    repo = build(case_id, arm=arm, rung=rung)
    return render(case_id, arm=arm, rung=rung, repo=repo)


@pytest.mark.parametrize("rung", RUNGS)
def test_rollout_branch_matches_canonical(rung):
    """The messages the rollout receives are exactly the canonical messages."""
    rendered = _canonical(rung=rung)
    canonical = rendered.canonical()
    from realj.eval import build_samples

    sample = build_samples(split="dev", rungs=(rung,), arms=("FORBIDDEN",))[0]
    built = canonical.to_inspect_messages()

    assert len(sample.input) == len(built)
    for a, b in zip(sample.input, built):
        assert type(a) is type(b)
        assert a.text == b.text


@pytest.mark.parametrize("rung", RUNGS)
def test_probe_branch_is_canonical_plus_one_message(rung):
    canonical = _canonical(rung=rung).canonical()
    branch = canonical.with_probe("which is it?")
    assert len(branch.messages) == len(canonical.messages) + 1
    assert branch.messages[:-1] == canonical.messages
    assert branch.messages[-1].role == "user"
    # Tool schemas must be carried through unchanged: the tool surface is part
    # of the context a model could read evaluation cues from.
    assert branch.tool_infos == canonical.tool_infos


@pytest.mark.parametrize("rung", RUNGS)
def test_continuation_branch_preserves_prefix(rung):
    canonical = _canonical(rung=rung).canonical()
    branch = canonical.with_continuation("let me look at the failing test")
    assert branch.messages[:-1] == canonical.messages
    assert branch.messages[-1].role == "assistant"


def test_probe_and_jlens_branch_from_the_same_state():
    canonical = _canonical().canonical()
    restored = prefix_from_record(prefix_record(_canonical()))
    assert_same_prefix(canonical, restored)


def test_record_roundtrip_is_lossless():
    for rung in RUNGS:
        rendered = _canonical(rung=rung)
        canonical = rendered.canonical()
        restored = prefix_from_record(prefix_record(rendered))
        assert restored.structural_hash == canonical.structural_hash
        assert restored.messages == canonical.messages
        assert restored.tool_infos == canonical.tool_infos


def test_divergent_prefixes_are_rejected():
    a = _canonical(rung=1).canonical()
    b = _canonical(rung=3).canonical()
    with pytest.raises(AssertionError, match="do not share a prefix"):
        assert_same_prefix(a, b)


def test_tool_schema_is_the_real_serialisation():
    infos = tool_infos(agent_tools())
    assert [i["name"] for i in infos] == ["bash", "text_editor"]
    for info in infos:
        assert "description" in info
        assert info["parameters"]["type"] == "object"
        assert info["parameters"]["properties"]


def test_tool_schema_hash_is_stable_and_sensitive():
    a = tool_schema_hash(tool_infos(agent_tools()))
    assert a == tool_schema_hash(tool_infos(agent_tools()))
    altered = list(tool_infos(agent_tools()))
    altered[0] = {**altered[0], "description": "something else"}
    assert tool_schema_hash(altered) != a


def test_hash_kind_is_honest_without_a_tokenizer():
    """A structural hash must never be reported as a token-level guarantee."""
    canonical = _canonical().canonical()
    identity = canonical.identity(tokenizer=None)
    assert identity["hash_kind"] == "structural"
    assert identity["n_prefix_tokens"] is None


def test_hf_conversion_serialises_tool_calls_stably():
    message = PrefixMessage(
        role="assistant",
        content="looking",
        tool_call_id="tc1",
        tool_name="bash",
        tool_arguments={"b": 2, "a": 1},
    )
    first = message.to_hf()
    assert first["tool_calls"][0]["function"]["arguments"] == '{"a": 1, "b": 2}'
    assert first == message.to_hf()


def test_debug_text_is_not_a_model_input_path():
    """debug_text includes the tool schemas; it exists for auditing only."""
    canonical = _canonical(rung=4).canonical()
    text = canonical.debug_text()
    assert "<tool_schema>" in text
    assert "<system>" in text


def test_minimal_prefix_converts_both_ways():
    prefix = CanonicalPrefix(
        messages=(PrefixMessage("system", "s"), PrefixMessage("user", "u")),
        tool_infos=tool_infos(agent_tools()),
    )
    msgs = prefix.to_inspect_messages()
    assert isinstance(msgs[0], ChatMessageSystem)
    assert isinstance(msgs[1], ChatMessageUser)
