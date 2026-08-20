"""The server's own manifest must pass its own audit."""
import asyncio

from context_audit.audit import audit_manifest
from context_audit.server import server


def _manifest():
    tools = asyncio.run(server.list_tools())
    return [
        {
            "name": t.name,
            "description": t.description,
            "inputSchema": t.input_schema,
        }
        for t in tools
    ]


def test_own_manifest_is_clean():
    report = audit_manifest(_manifest(), server="context-audit")
    assert report.clean, [f.as_dict() for f in report.findings]


def test_server_exposes_three_tools():
    assert len(_manifest()) == 3
