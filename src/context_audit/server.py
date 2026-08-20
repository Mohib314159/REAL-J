"""MCP server exposing the context audit.

Deliberately self-referential: this server's own tool descriptions are written
to pass its own audit, and `tests/test_self_audit.py` asserts that they do. A
tool that warns about leaking vocabulary while leaking it would be worth
nothing.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, Literal

from mcp.server import MCPServer
from pydantic import Field

from .audit import LEXICONS, audit_manifest, format_report
from .semantic import (
    SemanticUnavailable,
    audit_with_semantics,
    available as semantic_available,
    format_combined,
)

server = MCPServer(
    name="context-audit",
    version="0.1.0",
    instructions=(
        "Scans tool manifests and prompts for wording that reaches a model's "
        "context without the application author intending it."
    ),
)


@server.tool(
    description=(
        "Scan a list of tool definitions for wording that would reach a model's "
        "context. Reports the tool, the exact location in the schema, and the "
        "matched phrase."
    )
)
def audit_tool_manifest(
    tools: Annotated[
        list[dict[str, Any]],
        Field(description="Tool definitions, each with a name and optionally a description and inputSchema."),
    ],
    server_name: Annotated[
        str, Field(description="Label for the source, used in the report header.")
    ] = "unknown",
    output: Annotated[
        Literal["text", "json"], Field(description="Report shape to return.")
    ] = "text",
    deep: Annotated[
        bool,
        Field(
            description=(
                "Also ask a model to judge each string, catching framing that uses "
                "none of the listed phrases. Needs a provider key; off by default."
            )
        ),
    ] = False,
) -> str:
    if not deep:
        report = audit_manifest(tools, server=server_name)
        if output == "json":
            return json.dumps(report.as_dict(), indent=2)
        return format_report(report)

    try:
        report, extra = audit_with_semantics(tools, server=server_name)
    except SemanticUnavailable as exc:
        report = audit_manifest(tools, server=server_name)
        note = f"\n\nDeep pass skipped: {exc}"
        if output == "json":
            return json.dumps({**report.as_dict(), "deep_skipped": str(exc)}, indent=2)
        return format_report(report) + note

    if output == "json":
        return json.dumps(
            {**report.as_dict(), "model_findings": [f.as_dict() for f in extra]}, indent=2
        )
    return format_combined(report, extra)


@server.tool(
    description=(
        "Scan a single block of prose the same way, for checking a system prompt "
        "or one tool description before it ships."
    )
)
def audit_text(
    text: Annotated[str, Field(description="The prose to scan.")],
    label: Annotated[
        str, Field(description="Name to attach to findings from this text.")
    ] = "text",
) -> str:
    report = audit_manifest([{"name": label, "description": text}], server=label)
    return format_report(report)


@server.tool(
    description=(
        "Return the phrase categories this scanner matches on, so the caller can "
        "see what it does and does not cover."
    )
)
def list_lexicons() -> str:
    return json.dumps(
        {
            "categories": {cat: list(terms) for cat, terms in LEXICONS.items()},
            "deep_pass_available": semantic_available(),
        },
        indent=2,
    )


def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
