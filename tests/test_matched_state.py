"""The gate: the harness must add nothing to the canonical prefix.

Do not run held-out cases until this passes with a real tokenizer on the
mechanistic target. Here it runs structurally, which catches the failure that
actually occurred -- the scaffold injecting its own system prompt and submit
tool -- without needing a GPU.
"""

from __future__ import annotations

import pytest

from realj.integrity import (
    CapturedCall,
    MatchedStateFailure,
    certify,
    check_matched_state,
)
from realj.models import agent_tools, coding_agent
from realj.prefix import tool_infos
from realj.realism.rungs import render
from realj.tasks.generator import build

RUNGS = (1, 2, 3, 4)


def _canonical(rung=3):
    repo = build("case_01", arm="FORBIDDEN", rung=rung)
    return render("case_01", arm="FORBIDDEN", rung=rung, repo=repo).canonical()


def _schemas(canonical, extra=None, tweak=None):
    """Captured tool surface, as provider-facing dicts."""
    out = [dict(t) for t in canonical.tool_infos]
    if tweak:
        out[0] = {**out[0], "description": tweak}
    if extra:
        out.append({"name": extra, "description": "x", "parameters": {"type": "object"}})
    return out


@pytest.mark.parametrize("rung", RUNGS)
def test_canonical_matches_a_faithful_call(rung):
    canonical = _canonical(rung)
    captured = CapturedCall(
        messages=canonical.to_inspect_messages(), tools=_schemas(canonical)
    )
    result = check_matched_state(captured, canonical)
    assert result["passed"], result["problems"]


def test_scaffold_submit_tool_fails_the_gate():
    """The exact regression: react() adding a submit tool of its own."""
    canonical = _canonical()
    captured = CapturedCall(
        messages=canonical.to_inspect_messages(),
        tools=_schemas(canonical, extra="submit"),
    )
    result = check_matched_state(captured, canonical)
    assert not result["passed"]
    assert any("tools" in p for p in result["problems"])


def test_same_tool_name_different_description_fails_the_gate():
    """Name-only comparison would pass a description that leaks context.

    A bash tool described as "Run a shell command as part of this evaluation."
    has the same name and a completely different experimental meaning.
    """
    canonical = _canonical()
    captured = CapturedCall(
        messages=canonical.to_inspect_messages(),
        tools=_schemas(
            canonical, tweak="Run a shell command as part of this evaluation."
        ),
    )
    result = check_matched_state(captured, canonical)
    assert not result["passed"]
    assert any("schemas differ" in p for p in result["problems"])


def test_differing_tokens_fail_certification():
    """The certificate must come from comparing tokens, not from assuming them."""

    class FakeTok:
        def apply_chat_template(self, messages, tools=None, **kw):
            blob = "|".join(m.get("content", "") for m in messages)
            blob += "|" + "|".join(sorted(t.get("name", "") for t in (tools or [])))
            return [ord(c) % 97 for c in blob] if kw.get("tokenize") else blob

    canonical = _canonical()
    captured = CapturedCall(
        messages=canonical.to_inspect_messages(),
        tools=_schemas(canonical),
    )
    ok = check_matched_state(captured, canonical, tokenizer=FakeTok())
    assert ok["passed"] and ok["hash_kind"] == "token_ids"
    assert ok["captured_token_hash"] == ok["canonical_token_hash"]


def test_scaffold_system_prompt_fails_the_gate():
    from inspect_ai.model import ChatMessageSystem

    canonical = _canonical()
    captured = CapturedCall(
        messages=[ChatMessageSystem(content="You are a helpful assistant.")]
        + canonical.to_inspect_messages(),
        tools=_schemas(canonical),
    )
    assert not check_matched_state(captured, canonical)["passed"]


def test_certify_fails_closed():
    with pytest.raises(MatchedStateFailure):
        certify({"passed": False, "problems": ["tools differ"]})


def test_certify_will_not_claim_token_ids_without_a_tokenizer():
    """hash_kind='token_ids' must be unreachable unless tokens were compared."""
    canonical = _canonical()
    captured = CapturedCall(
        messages=canonical.to_inspect_messages(), tools=_schemas(canonical)
    )
    identity = certify(check_matched_state(captured, canonical, tokenizer=None))
    assert identity["hash_kind"] == "structural"
    assert identity["n_prefix_tokens"] is None


def test_agent_supplies_only_the_canonical_tools():
    """prompt=None, submit=False means the scaffold contributes nothing."""
    coding_agent()
    names = sorted(t["name"] for t in tool_infos(agent_tools()))
    assert names == ["bash", "text_editor"]
    assert "submit" not in names


def test_harness_instructions_are_inside_the_prefix():
    from realj.models import HARNESS_INSTRUCTIONS

    canonical = _canonical()
    system = canonical.messages[0].content
    assert HARNESS_INSTRUCTIONS in system, (
        "the harness instructions are not in the canonical prefix, so the "
        "hashed state is not the state the model sees"
    )
