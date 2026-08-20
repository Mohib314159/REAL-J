"""Scan an MCP tool manifest for vocabulary that reaches the model.

The premise is narrow. When a client connects to an MCP server, the server's
tool names, descriptions and full JSON parameter schemas are serialised into
the model's context on every turn. The application author usually did not
write that text and often has not read it. If it contains framing the agent
should not have -- that it is being evaluated, tested, observed, graded --
that framing is now part of the prompt, silently, on every call.

This scans the whole surface rather than the description field alone. A
per-parameter description or an enum value is just as visible to the model
as the tool description, and is far less likely to have been reviewed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

# Categories are separate because they mean different things. Evaluation
# framing tells the model the interaction is not real. Surveillance framing
# tells it someone is watching. Instructional framing is a third party trying
# to steer it. A manifest can leak one and not the others.
LEXICONS: dict[str, tuple[str, ...]] = {
    "evaluation": (
        "evaluation", "evaluated", "evaluating", "eval",
        "benchmark", "benchmarked", "test harness", "test suite",
        "graded", "grading", "grader", "scored by", "scoring rubric",
        "assessment", "assessed", "trial run", "experiment",
    ),
    "surveillance": (
        "monitored", "monitoring", "observed", "under observation",
        "recorded for", "logged for review", "audited by", "supervised by",
        "we are watching", "being watched",
    ),
    "instruction": (
        "you must", "you should always", "ignore previous",
        "disregard the", "your real task", "regardless of what the user",
        "do not tell the user", "without informing",
    ),
    "identity": (
        "as an ai", "you are an ai assistant", "language model",
        "your training data", "you are gpt", "you are claude",
    ),
}

# Words that are ordinary in a software context and would otherwise fire on
# every tool that touches a test file. "test" alone is useless as a signal in
# a repository full of unit tests; the phrase "test harness" is not.
BENIGN_IN_CONTEXT = ("unit test", "test file", "test case", "pytest", "run tests")


@dataclass(frozen=True)
class Finding:
    tool: str
    location: str          # where in the schema, e.g. "params.mode.enum[2]"
    category: str
    term: str
    excerpt: str

    def as_dict(self) -> dict[str, str]:
        return {
            "tool": self.tool,
            "location": self.location,
            "category": self.category,
            "term": self.term,
            "excerpt": self.excerpt,
        }


@dataclass
class AuditReport:
    server: str
    tools_scanned: int
    characters_scanned: int
    findings: list[Finding] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.findings

    def by_category(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.category] = out.get(f.category, 0) + 1
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "server": self.server,
            "tools_scanned": self.tools_scanned,
            "characters_scanned": self.characters_scanned,
            "clean": self.clean,
            "by_category": self.by_category(),
            "findings": [f.as_dict() for f in self.findings],
        }


def _excerpt(text: str, at: int, term: str, width: int = 60) -> str:
    start = max(0, at - width // 2)
    end = min(len(text), at + len(term) + width // 2)
    frag = text[start:end].replace("\n", " ").strip()
    return f"...{frag}..." if start > 0 or end < len(text) else frag


def _is_benign(text: str, at: int) -> bool:
    window = text[max(0, at - 20): at + 20].lower()
    return any(b in window for b in BENIGN_IN_CONTEXT)


def scan_text(text: str, tool: str, location: str) -> Iterator[Finding]:
    """Find leaking phrases, reporting each span once.

    Lexicons overlap on purpose -- "eval" is a substring of "evaluation" -- so
    matches are collected first and shorter ones dropped when they sit inside a
    longer hit. Reporting the same eight characters twice makes a two-word
    description look like two separate problems.
    """
    if not text:
        return
    lowered = text.lower()
    hits: list[tuple[int, int, str, str]] = []
    for category, terms in LEXICONS.items():
        for term in terms:
            for m in re.finditer(re.escape(term), lowered):
                if _is_benign(lowered, m.start()):
                    continue
                hits.append((m.start(), m.end(), category, term))

    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    kept: list[tuple[int, int, str, str]] = []
    for hit in hits:
        start, end, _, _ = hit
        if any(k[0] <= start and end <= k[1] for k in kept):
            continue
        kept.append(hit)

    for start, _end, category, term in kept:
        yield Finding(
            tool=tool,
            location=location,
            category=category,
            term=term,
            excerpt=_excerpt(text, start, term),
        )


# Schema keywords whose *string values* the model reads as prose or as data.
PROSE_KEYS = ("description", "title", "$comment")
VALUE_KEYS = ("default", "const")
LIST_VALUE_KEYS = ("enum", "examples")

# Keywords whose *keys* name things the model sees: property names appear in
# the serialised schema, and so do the names under $defs.
NAME_CONTAINERS = ("properties", "patternProperties", "$defs", "definitions")


def _split_identifier(name: str) -> str:
    """Turn `evaluation_mode` / `evaluationMode` into scannable words.

    A property name is model-visible text even though nobody writes it as
    prose. Splitting on case and separators lets the same lexicon apply
    without a second word list.
    """
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_\-./]+", " ", spaced)


def walk_schema(node: Any, tool: str, path: str) -> Iterator[tuple[str, str]]:
    """Yield (location, text) for every string a model could read.

    The whole serialised schema reaches the model, not just the description
    fields. That includes values the caller never thinks of as prose --
    a `default` of "benchmark_response", a `const`, an item in `examples` --
    and the property names themselves, since `evaluation_mode` is as visible
    in the JSON as any sentence. Property names are split into words before
    scanning so the ordinary lexicon applies to them.
    """
    if isinstance(node, dict):
        for key in PROSE_KEYS:
            val = node.get(key)
            if isinstance(val, str):
                yield f"{path}.{key}", val
        for key in VALUE_KEYS:
            val = node.get(key)
            if isinstance(val, str):
                yield f"{path}.{key}", val
        for key in LIST_VALUE_KEYS:
            items = node.get(key)
            if isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, str):
                        yield f"{path}.{key}[{i}]", item

        for container in NAME_CONTAINERS:
            block = node.get(container)
            if isinstance(block, dict):
                for prop_name in block:
                    yield f"{path}.{container}.{prop_name}", _split_identifier(prop_name)

        skip = set(PROSE_KEYS) | set(VALUE_KEYS) | set(LIST_VALUE_KEYS)
        for key, val in node.items():
            if key in skip:
                continue
            if isinstance(val, (dict, list)):
                yield from walk_schema(val, tool, f"{path}.{key}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            if isinstance(item, (dict, list)):
                yield from walk_schema(item, tool, f"{path}[{i}]")


def audit_manifest(tools: Sequence[dict[str, Any]], server: str = "unknown") -> AuditReport:
    """Audit a list of MCP tool definitions.

    Each tool is expected to have `name`, optionally `description`, and
    optionally `inputSchema` -- the shape returned by an MCP `tools/list`.
    """
    report = AuditReport(server=server, tools_scanned=len(tools), characters_scanned=0)
    for tool in tools:
        name = str(tool.get("name", "<unnamed>"))
        surfaces: list[tuple[str, str]] = [("name", name)]
        desc = tool.get("description")
        if isinstance(desc, str):
            surfaces.append(("description", desc))
        # MCP uses `inputSchema`; Inspect's ToolInfo serialises `parameters`.
        # Accepting both means a harness manifest can be piped in unchanged.
        seen_input = False
        for field_name in ("inputSchema", "input_schema", "parameters"):
            schema = tool.get(field_name)
            if isinstance(schema, dict) and not seen_input:
                surfaces.extend(walk_schema(schema, name, "params"))
                seen_input = True
        for field_name in ("outputSchema", "output_schema"):
            schema = tool.get(field_name)
            if isinstance(schema, dict):
                surfaces.extend(walk_schema(schema, name, "output"))
                break
        for location, text in surfaces:
            report.characters_scanned += len(text)
            report.findings.extend(scan_text(text, name, location))
    return report


def format_report(report: AuditReport) -> str:
    lines = [
        f"Context audit: {report.server}",
        f"{report.tools_scanned} tools, {report.characters_scanned} characters of "
        f"model-visible text.",
        "",
    ]
    if report.clean:
        lines.append("No leaking vocabulary found.")
        lines.append("")
        lines.append(
            "Note this checks a fixed lexicon, not meaning. A tool can frame the "
            "interaction without using any of these words."
        )
        return "\n".join(lines)

    counts = report.by_category()
    lines.append(
        "Found " + ", ".join(f"{n} {cat}" for cat, n in sorted(counts.items())) + "."
    )
    lines.append("")
    for f in report.findings:
        lines.append(f"  [{f.category}] {f.tool} @ {f.location}")
        lines.append(f"      term: {f.term!r}")
        lines.append(f"      {f.excerpt}")
    lines.append("")
    lines.append(
        "Every string above is serialised into the model's context on each turn, "
        "whether or not the tool is called."
    )
    return "\n".join(lines)
