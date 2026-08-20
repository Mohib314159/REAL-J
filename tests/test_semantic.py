"""Tests for the optional model pass.

The pass itself needs a provider key, so these check everything around it:
the surface walk it shares with the lexicon pass, the result model, and that
it degrades cleanly when the key is absent rather than crashing.
"""
import pytest

from context_audit.semantic import (
    SemanticFinding,
    SemanticUnavailable,
    SemanticVerdict,
    available,
    semantic_scan,
    surfaces,
)

MANIFEST = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "Command to run."},
                "mode": {"type": "string", "enum": ["fast", "safe"]},
            },
        },
    }
]


def test_surfaces_finds_every_model_visible_string():
    found = {(loc, text) for _tool, loc, text in surfaces(MANIFEST)}
    assert ("description", "Run a shell command.") in found
    assert ("params.properties.cmd.description", "Command to run.") in found
    assert ("params.properties.mode.enum[0]", "fast") in found


def test_surfaces_skips_empty_strings():
    manifest = [{"name": "t", "description": "   ", "inputSchema": {"description": ""}}]
    assert surfaces(manifest) == []


def test_verdict_model_requires_a_decision():
    v = SemanticVerdict(leaks=False, category="none")
    assert v.span == "" and v.reasoning == ""
    v2 = SemanticVerdict(
        leaks=True, category="evaluation", span="graded against a reference",
        reasoning="Tells the model its output is marked.",
    )
    assert v2.category == "evaluation"


def test_verdict_rejects_unknown_category():
    with pytest.raises(Exception):
        SemanticVerdict(leaks=True, category="vibes")


def test_finding_folds_into_lexicon_shape():
    f = SemanticFinding("bash", "description", "evaluation", "graded", "Marks output.")
    folded = f.as_finding()
    assert folded.tool == "bash"
    assert folded.category == "evaluation"
    assert "Marks output." in folded.term
    assert f.as_dict()["source"] == "model"


def test_scan_refuses_cleanly_without_a_key(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert available() is False
    with pytest.raises(SemanticUnavailable):
        semantic_scan(MANIFEST)


def test_available_reports_true_with_a_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-value-not-real")
    assert available("anthropic:claude-sonnet-4-5") is True
