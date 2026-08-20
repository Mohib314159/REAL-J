"""Optional second pass: ask a model what the lexicon cannot see.

The lexicon in `audit.py` matches phrases. It cannot catch a tool description
that frames the interaction without using any listed word -- "your response
here will be compared against a reference answer" contains nothing on the
list and does exactly what the list exists to detect.

So this pass exists, and it is deliberately second. The lexicon pass is
deterministic, free, needs no network and is the default. This one costs an
API call and can itself be wrong, so it is opt-in and its findings are marked
as coming from a model rather than from a match.

Requires `pip install "context-audit-mcp[semantic]"` and a provider key
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or whichever the chosen model needs).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from .audit import AuditReport, Finding, audit_manifest, walk_schema

DEFAULT_MODEL = "anthropic:claude-sonnet-4-5"

CATEGORIES = ("evaluation", "surveillance", "instruction", "identity")

SYSTEM_PROMPT = """\
You inspect text that will be placed into a language model's context as part \
of a tool definition, and decide whether it frames the interaction in a way \
the application author probably did not intend.

Report a leak only when the text tells the model something about the nature \
of the interaction rather than about the tool's function. Four kinds:

- evaluation: the interaction is a test, benchmark, trial or graded exercise
- surveillance: the model is watched, recorded, reviewed or reported on
- instruction: someone other than the user is steering the model's behaviour
- identity: framing about what the model is or how it was made

Ordinary technical description is not a leak. A tool that runs unit tests, \
reads a log, or reports a score for some domain object is describing its \
function. Judge by whether the sentence changes what the model believes about \
its own situation.

Quote the exact substring you are objecting to. If nothing qualifies, say so.\
"""


class SemanticVerdict(BaseModel):
    """Structured result. The model must fill this shape, not write prose."""

    leaks: bool = Field(description="Whether this text frames the interaction.")
    category: Literal["evaluation", "surveillance", "instruction", "identity", "none"] = (
        Field(description="Which kind of framing, or none.")
    )
    span: str = Field(
        default="", description="Exact substring objected to. Empty when leaks is false."
    )
    reasoning: str = Field(
        default="", description="One sentence on what the text tells the model."
    )


@dataclass
class SemanticFinding:
    tool: str
    location: str
    category: str
    span: str
    reasoning: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "location": self.location,
            "category": self.category,
            "span": self.span,
            "reasoning": self.reasoning,
            "source": "model",
        }

    def as_finding(self) -> Finding:
        """Fold into the lexicon report shape so both passes print together."""
        return Finding(
            tool=self.tool,
            location=self.location,
            category=self.category,
            term=f"(model) {self.reasoning}",
            excerpt=self.span,
        )


class SemanticUnavailable(RuntimeError):
    """Raised when the optional dependency or a provider key is missing."""


def available(model: str = DEFAULT_MODEL) -> bool:
    """True when a semantic pass could actually run."""
    try:
        import pydantic_ai  # noqa: F401
    except ImportError:
        return False
    provider = model.split(":", 1)[0]
    key = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google-gla": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }.get(provider)
    return bool(key and os.environ.get(key))


def _agent(model: str):
    try:
        from pydantic_ai import Agent
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SemanticUnavailable(
            'pydantic-ai is not installed. pip install "context-audit-mcp[semantic]"'
        ) from exc
    return Agent(model, output_type=SemanticVerdict, system_prompt=SYSTEM_PROMPT)


def surfaces(tools: Sequence[dict[str, Any]]) -> list[tuple[str, str, str]]:
    """Every (tool, location, text) the model can read. Shared with the lexicon pass."""
    out: list[tuple[str, str, str]] = []
    for tool in tools:
        name = str(tool.get("name", "<unnamed>"))
        desc = tool.get("description")
        if isinstance(desc, str) and desc.strip():
            out.append((name, "description", desc))
        schema = tool.get("inputSchema") or tool.get("input_schema")
        if isinstance(schema, dict):
            for location, text in walk_schema(schema, name, "params"):
                if text.strip():
                    out.append((name, location, text))
    return out


def semantic_scan(
    tools: Sequence[dict[str, Any]], model: str = DEFAULT_MODEL
) -> list[SemanticFinding]:
    """Run the model pass over every model-visible string.

    One call per surface rather than one call for the whole manifest: a single
    call over a long manifest reliably loses the short strings, and enum values
    are exactly the short strings this tool exists to catch.
    """
    if not available(model):
        raise SemanticUnavailable(
            "No provider key found, or pydantic-ai is not installed. "
            "The lexicon pass runs without either."
        )
    agent = _agent(model)
    findings: list[SemanticFinding] = []
    for tool, location, text in surfaces(tools):
        result = agent.run_sync(text)
        verdict = result.output
        if verdict.leaks and verdict.category != "none":
            findings.append(
                SemanticFinding(
                    tool=tool,
                    location=location,
                    category=verdict.category,
                    span=verdict.span or text,
                    reasoning=verdict.reasoning,
                )
            )
    return findings


def audit_with_semantics(
    tools: Sequence[dict[str, Any]],
    server: str = "unknown",
    model: str = DEFAULT_MODEL,
) -> tuple[AuditReport, list[SemanticFinding]]:
    """Lexicon pass always; model pass on top.

    The lexicon report is returned unchanged so a caller can still see which
    findings are deterministic. Model findings that duplicate a lexicon hit on
    the same location are dropped -- the cheap pass already caught those.
    """
    report = audit_manifest(tools, server=server)
    already = {(f.tool, f.location) for f in report.findings}
    extra = [f for f in semantic_scan(tools, model=model) if (f.tool, f.location) not in already]
    return report, extra


def format_combined(report: AuditReport, extra: list[SemanticFinding]) -> str:
    from .audit import format_report

    text = format_report(report)
    if not extra:
        return text + "\n\nModel pass found nothing the lexicon missed."
    lines = [text, "", f"Model pass found {len(extra)} further:", ""]
    for f in extra:
        lines.append(f"  [{f.category}] {f.tool} @ {f.location}")
        lines.append(f"      {f.span!r}")
        lines.append(f"      {f.reasoning}")
    lines.append("")
    lines.append(
        "Model findings are judgements, not matches. Read the span and decide."
    )
    return "\n".join(lines)
