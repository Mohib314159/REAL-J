"""The canonical prefix. One model state, four branches, provably identical.

This module exists because the previous design had a scientific bug that the
README actively misdescribed. It claimed the rollout, the recognition fork and
the J-lens all consumed the same stored prefix. They did not:

* the natural rollout received structured Inspect messages plus tool schemas;
* ``prefix_text`` hand-serialised those into an XML-ish flat string;
* recognition and the J-lens received the flat string.

So ``P_rollout != P_Q = P_J``. Semantically related, not the same model state.
In a study about whether tiny contextual cues tell a model it is being
observed, the chat template and the tool representation *are* some of those
cues. Comparing channels across two different serialisations is comparing
measurements taken on two different inputs.

The fix is to make one object the single source of truth:

    CanonicalPrefix
      .messages        structured messages, exactly as the rollout receives
      .tool_infos      tool schemas, exactly as the provider receives
      .render(tok)     the model's own chat template applied to both
      .token_ids(tok)  the resulting token sequence
      .token_hash(tok) sha256 of those ids

Every branch is then defined as an operation on this object rather than on a
string:

    P -> natural rollout          V_early, V_full, B, and J_preaction
    P + probe message -> Q
    P (offline forward pass) -> J_prefix

For the open-weight mechanistic arm, ``token_hash`` makes the shared-prefix
claim checkable rather than asserted: the same hash in the rollout record, the
recognition record and the J-lens record is a proof that all three started
from the same tokens.

For closed models there is no tokenizer, so ``token_hash`` is unavailable and
``structural_hash`` is used instead -- a hash over the structured messages and
tool schemas. Weaker, and labelled as such: ``hash_kind`` records which
guarantee a given row actually has. That is why the mechanistic arm is
open-weight only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

Role = Literal["system", "user", "assistant", "tool"]


def _sha(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PrefixMessage:
    """One message, in a form that converts losslessly to any target format."""

    role: Role
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_arguments: dict | None = None

    def as_dict(self) -> dict:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.tool_name:
            out["tool_name"] = self.tool_name
        if self.tool_arguments is not None:
            out["tool_arguments"] = self.tool_arguments
        return out

    # -- target-format conversions -------------------------------------

    def to_inspect(self):
        """Inspect chat message. What the natural rollout actually receives."""
        from inspect_ai.model import (
            ChatMessageAssistant,
            ChatMessageSystem,
            ChatMessageTool,
            ChatMessageUser,
        )
        from inspect_ai.tool import ToolCall

        if self.role == "system":
            return ChatMessageSystem(content=self.content)
        if self.role == "user":
            return ChatMessageUser(content=self.content)
        if self.role == "assistant":
            if self.tool_call_id and self.tool_name:
                return ChatMessageAssistant(
                    content=self.content,
                    tool_calls=[
                        ToolCall(
                            id=self.tool_call_id,
                            function=self.tool_name,
                            arguments=self.tool_arguments or {},
                        )
                    ],
                )
            return ChatMessageAssistant(content=self.content)
        if self.role == "tool":
            return ChatMessageTool(
                content=self.content,
                tool_call_id=self.tool_call_id,
                function=self.tool_name,
            )
        raise ValueError(f"unknown role {self.role!r}")

    def to_hf(self) -> dict:
        """HuggingFace chat-template message.

        Tool calls go in ``tool_calls`` in the OpenAI-ish shape that Qwen's and
        most other templates expect. Arguments are serialised with sorted keys
        so the token sequence is stable across runs -- dict ordering silently
        changing the tokens would defeat the entire point of hashing them.
        """
        if self.role == "assistant" and self.tool_call_id and self.tool_name:
            return {
                "role": "assistant",
                "content": self.content,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": self.tool_name,
                            "arguments": json.dumps(
                                self.tool_arguments or {}, sort_keys=True
                            ),
                        },
                    }
                ],
            }
        if self.role == "tool":
            return {"role": "tool", "name": self.tool_name, "content": self.content}
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class CanonicalPrefix:
    """The exact model state immediately before the target begins reasoning."""

    messages: tuple[PrefixMessage, ...]
    tool_infos: tuple[dict, ...]  # serialised ToolInfo, provider-facing
    meta: dict = field(default_factory=dict)

    # -- identity ------------------------------------------------------

    @property
    def structural_hash(self) -> str:
        """Hash over structured messages plus tool schemas.

        Used for closed models, where no tokenizer is available. Guarantees the
        same *structure* reached each branch, not the same tokens.
        """
        blob = json.dumps(
            {
                "messages": [m.as_dict() for m in self.messages],
                "tools": list(self.tool_infos),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return _sha(blob)

    def render(self, tokenizer, tools: Sequence[dict] | None = None) -> str:
        """The model's own chat template, tools included."""
        return tokenizer.apply_chat_template(
            [m.to_hf() for m in self.messages],
            tools=list(tools if tools is not None else self.tool_infos) or None,
            add_generation_prompt=True,
            tokenize=False,
        )

    def token_ids(self, tokenizer) -> list[int]:
        return list(
            tokenizer.apply_chat_template(
                [m.to_hf() for m in self.messages],
                tools=list(self.tool_infos) or None,
                add_generation_prompt=True,
                tokenize=True,
            )
        )

    def token_hash(self, tokenizer) -> str:
        ids = self.token_ids(tokenizer)
        return _sha(",".join(str(i) for i in ids))

    def identity(self, tokenizer=None) -> dict:
        """What goes in the result row so the shared-prefix claim is auditable."""
        if tokenizer is not None:
            return {
                "prefix_hash": self.token_hash(tokenizer),
                "hash_kind": "token_ids",
                "n_prefix_tokens": len(self.token_ids(tokenizer)),
                "structural_hash": self.structural_hash,
            }
        return {
            "prefix_hash": self.structural_hash,
            "hash_kind": "structural",
            "n_prefix_tokens": None,
            "structural_hash": self.structural_hash,
        }

    # -- branches ------------------------------------------------------

    def to_inspect_messages(self) -> list:
        """Branch 1: the natural rollout."""
        return [m.to_inspect() for m in self.messages]

    def with_probe(self, probe: str) -> "CanonicalPrefix":
        """Branch 2: recognition. Appends one user message, changes nothing else.

        Returns a new prefix rather than a flattened string, so the probe
        branch is provably the canonical state plus exactly one message.
        """
        return CanonicalPrefix(
            messages=self.messages + (PrefixMessage("user", probe),),
            tool_infos=self.tool_infos,
            meta={**self.meta, "probe_appended": True},
        )

    def with_continuation(self, generated: str) -> "CanonicalPrefix":
        """Branch 3: J_preaction. Appends the target's own early reasoning.

        The middle-realism hypothesis is that ambiguity forces the model to
        *work out* whether it is being observed. If that is right, the
        representation may not exist in the prefix at all -- it may be
        constructed during generation. Scoring only the prefix would miss
        precisely the case the hypothesis predicts.
        """
        return CanonicalPrefix(
            messages=self.messages + (PrefixMessage("assistant", generated),),
            tool_infos=self.tool_infos,
            meta={**self.meta, "continuation_chars": len(generated)},
        )

    # -- debugging only ------------------------------------------------

    def debug_text(self) -> str:
        """Human-readable dump. **Never** feed this to a model.

        The old ``prefix_text`` was exactly this string, and it was being used
        as a model input. It is kept for eyeballing diffs and for the leakage
        scan, and named so that using it as an input looks wrong.
        """
        parts = []
        for m in self.messages:
            if m.role == "tool":
                parts.append(f"<tool_result name={m.tool_name}>\n{m.content}\n</tool_result>")
            else:
                parts.append(f"<{m.role}>\n{m.content}\n</{m.role}>")
        for tool in self.tool_infos:
            parts.append(f"<tool_schema>\n{json.dumps(tool, sort_keys=True)}\n</tool_schema>")
        return "\n\n".join(parts)


def tool_infos(tools: Sequence[Any]) -> tuple[dict, ...]:
    """Serialise Inspect tools to the schema the provider actually sees.

    Not docstrings. Inspect sends the tool name, the description and the full
    JSON parameter schema, and any of the three can carry leakage -- which is
    how ``react()``'s "Submit an answer for evaluation." got as far as it did.
    """
    from inspect_ai.tool import ToolInfo
    from inspect_ai.tool._tool_def import ToolDef

    out = []
    for tool in tools:
        td = ToolDef(tool)
        info = ToolInfo(
            name=td.name,
            description=td.description,
            parameters=td.parameters,
            options=td.options,
        )
        out.append(info.model_dump(exclude_none=True))
    return tuple(out)


def tool_schema_hash(infos: Sequence[dict]) -> str:
    return _sha(json.dumps(list(infos), sort_keys=True, separators=(",", ":")))


def assert_same_prefix(a: CanonicalPrefix, b: CanonicalPrefix, tokenizer=None) -> None:
    """Fail loudly if two branches did not start from the same state."""
    if tokenizer is not None:
        ha, hb = a.token_hash(tokenizer), b.token_hash(tokenizer)
        kind = "token"
    else:
        ha, hb = a.structural_hash, b.structural_hash
        kind = "structural"
    if ha != hb:
        raise AssertionError(
            f"branches do not share a prefix ({kind} hash {ha[:12]} != {hb[:12]}). "
            "Any dissociation between channels measured from these is an "
            "artefact of them having seen different inputs."
        )
