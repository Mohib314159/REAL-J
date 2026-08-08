"""The matched-state gate.

The central claim of this project is that Q, J, V and B are measurements of
the *same* model state. Everything downstream -- every dissociation, every
channel comparison -- is only meaningful if that holds. It has been asserted
twice now and been false both times, so it gets a gate rather than a sentence.

    actual first natural-rollout model call   -> sha256(token_ids)
    offline J_prefix call                     -> sha256(token_ids)
    recognition branch before the probe       -> sha256(token_ids)

If those three do not match, the matched-state claim has failed and the
mechanistic result is not interpretable.

``hash_kind="token_ids"`` is only settable through ``certify()``, which
requires the equality check to have passed. A row can no longer *claim* a
token-level guarantee it does not have.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


class MatchedStateFailure(AssertionError):
    """Raised when the branches did not start from the same tokens."""


@dataclass(frozen=True)
class CapturedCall:
    """What a model actually received on its first call."""

    messages: list
    tools: list

    def tool_names(self) -> list[str]:
        return sorted(i["name"] for i in self.tool_schemas())

    def tool_schemas(self) -> list[dict]:
        """Full provider-facing schemas, not just names.

        Comparing names only would pass a bash tool whose description had been
        changed to "Run a shell command as part of this evaluation." Same name,
        completely different experimental context.
        """
        from realj.prefix import tool_infos

        out: list[dict] = []
        for t in self.tools:
            if isinstance(t, dict):
                out.append(t)
            elif hasattr(t, "model_dump"):
                out.append(t.model_dump(exclude_none=True))
            else:
                out.extend(tool_infos([t]))
        return sorted(out, key=lambda d: d.get("name", ""))

    def to_canonical(self):
        """Rebuild a CanonicalPrefix from what the model actually received."""
        from realj.prefix import CanonicalPrefix, PrefixMessage

        messages = []
        for m in self.messages:
            role = getattr(m, "role", "user")
            call = (getattr(m, "tool_calls", None) or [None])[0]
            messages.append(
                PrefixMessage(
                    role=role,
                    content=getattr(m, "text", "") or "",
                    tool_call_id=getattr(m, "tool_call_id", None)
                    or (call.id if call else None),
                    tool_name=getattr(m, "function", None)
                    or (call.function if call else None),
                    tool_arguments=(call.arguments if call else None),
                )
            )
        return CanonicalPrefix(
            messages=tuple(messages), tool_infos=tuple(self.tool_schemas())
        )


class CapturingModel:
    """Records the first generate() call and stops the rollout.

    Used by the integration test to see what the harness *actually* sends,
    rather than what the dataset contains. The gap between those two is exactly
    where the scaffold used to inject its own prompt and submit tool.
    """

    def __init__(self):
        self.captured: CapturedCall | None = None

    async def generate(self, input, tools=None, **kwargs):  # noqa: A002
        if self.captured is None:
            self.captured = CapturedCall(messages=list(input), tools=list(tools or []))
        raise StopRollout()


class StopRollout(Exception):
    """Ends a rollout once the first call has been captured."""


def _sha(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def token_hash(ids) -> str:
    return _sha(",".join(str(int(i)) for i in ids))


def structural_call_hash(call: CapturedCall) -> str:
    """Hash of a captured call, comparable to CanonicalPrefix.structural_hash."""
    payload = {
        "messages": [
            {
                "role": getattr(m, "role", ""),
                "content": getattr(m, "text", "") or "",
            }
            for m in call.messages
        ],
        "tools": call.tool_names(),
    }
    return _sha(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def check_matched_state(
    captured: CapturedCall,
    canonical,
    tokenizer=None,
) -> dict:
    """Compare the real first call against the canonical prefix.

    With a tokenizer the comparison is over token ids and the result may be
    certified. Without one it is structural, and it may not.
    """
    problems: list[str] = []

    canon_msgs = canonical.to_inspect_messages()
    if len(captured.messages) != len(canon_msgs):
        problems.append(
            f"message count {len(captured.messages)} != canonical {len(canon_msgs)}; "
            "the harness is adding or removing messages"
        )
    else:
        for i, (a, b) in enumerate(zip(captured.messages, canon_msgs)):
            if (getattr(a, "text", "") or "") != (getattr(b, "text", "") or ""):
                problems.append(f"message {i} text differs from canonical")

    canon_tools = sorted(canonical.tool_infos, key=lambda d: d.get("name", ""))
    got_tools = captured.tool_schemas()
    if [t.get("name") for t in got_tools] != [t.get("name") for t in canon_tools]:
        problems.append(
            f"tools {captured.tool_names()} != canonical "
            f"{[t.get('name') for t in canon_tools]}; the harness is supplying a "
            "different tool surface (a scaffold submit tool is the usual cause)"
        )
    elif got_tools != canon_tools:
        differing = [
            t.get("name")
            for t, c in zip(got_tools, canon_tools)
            if t != c
        ]
        problems.append(
            f"tool schemas differ for {differing} despite matching names; the "
            "description or parameter schema the model receives is not the one "
            "in the canonical prefix"
        )

    result = {
        "passed": not problems,
        "problems": problems,
        "hash_kind": "structural",
        "captured_hash": structural_call_hash(captured),
        "canonical_hash": canonical.structural_hash,
    }

    # Token certification: tokenize the CAPTURED call and compare against the
    # canonical one. The previous version tokenized only the canonical prefix
    # and labelled the result "token_ids" whenever the text and tool *names*
    # matched -- so it certified a comparison it had never made.
    if tokenizer is not None:
        canon_ids = canonical.token_ids(tokenizer)
        captured_ids = captured.to_canonical().token_ids(tokenizer)
        result["canonical_token_hash"] = token_hash(canon_ids)
        result["captured_token_hash"] = token_hash(captured_ids)
        if captured_ids != canon_ids:
            problems.append(
                f"token sequences differ: captured {len(captured_ids)} tokens "
                f"({token_hash(captured_ids)[:12]}) vs canonical "
                f"{len(canon_ids)} ({token_hash(canon_ids)[:12]})"
            )
            result["passed"] = False
            result["problems"] = problems
        else:
            result.update(
                {
                    "hash_kind": "token_ids",
                    "prefix_hash": token_hash(canon_ids),
                    "n_prefix_tokens": len(canon_ids),
                }
            )
    return result


def certify(result: dict) -> dict:
    """Return the identity fields a result row may use. Fails closed."""
    if not result.get("passed"):
        raise MatchedStateFailure(
            "matched-state check failed: "
            + "; ".join(result.get("problems", []))
            + ". Q, J, V and B are not measurements of the same state, so no "
            "dissociation between them is interpretable."
        )
    if result.get("hash_kind") != "token_ids":
        return {
            "prefix_hash": result["canonical_hash"],
            "hash_kind": "structural",
            "n_prefix_tokens": None,
        }
    return {
        "prefix_hash": result["prefix_hash"],
        "hash_kind": "token_ids",
        "n_prefix_tokens": result["n_prefix_tokens"],
    }
