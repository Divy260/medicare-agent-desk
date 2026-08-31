"""
The LangChain tool layer.

The whole point of this module is one sentence: **the framework gets a wrapped
tool, never the raw function.**

It is tempting to write `StructuredTool.from_function(lookup_policy)` and move
on. Do that and the guardrail in guardrails/rules.py is gone — LangChain's tool
node calls the function the moment the model emits a tool call, and nothing in
between checks the allow-list, validates the arguments, enforces the rate limit
or requires human approval. The security boundary is wherever execution
actually happens, so that is where the check has to live. `guarded()` below
closes over the policy and returns a tool that cannot be called around it.

The second thing worth noticing is that the JSON Schemas are *not* rewritten
here. `tools/registry.py` already holds them, because a tool schema is a
contract with the model, and having two copies of it is how they drift. The
same `input_schema` dict is handed to the Anthropic SDK by agents/base.py and
to LangChain by this module. One definition, two frameworks.
"""

from __future__ import annotations

import os
from typing import Callable

from langchain_core.tools import StructuredTool

from guardrails.rules import check_tool_call
from tools.registry import TOOLS_BY_NAME


def guarded(name: str, call_counts: dict[str, int], approvals: set[str],
            sink: Callable[[str, dict], None]) -> StructuredTool:
    """
    Wrap one registry tool as a LangChain tool, with the guardrail inside it.

    `sink(kind, data)` receives the same event dicts observability/trace.py
    consumes, so a tool call made through LangChain shows up in the trace
    identically to one made through the native loop. Two engines, one trace.
    """
    tool = TOOLS_BY_NAME[name]

    def _run(**kwargs) -> dict:
        verdict = check_tool_call(name, kwargs, call_counts, TOOLS_BY_NAME,
                                  human_approved=name in approvals)
        if not verdict.allowed:
            sink("guardrail", {"layer": "tool", "rule": verdict.rule,
                               "reason": verdict.reason, "tool": name})
            sink("tool_call", {"name": name, "args": kwargs, "is_error": True,
                               "blocked_by": verdict.rule})
            return {"error": verdict.rule, "detail": verdict.reason}

        # A tool failure is DATA, not an exception. Raising here would abort the
        # graph; returning the error lets the model read it and adapt, which is
        # the behaviour the error-recovery eval case depends on.
        try:
            payload = tool.fn(**kwargs)
            is_error = "error" in payload
        except TypeError as exc:
            payload, is_error = {"error": "bad_arguments", "detail": str(exc)}, True
        except Exception as exc:                              # noqa: BLE001
            payload, is_error = {"error": type(exc).__name__, "detail": str(exc)}, True

        call_counts[name] = call_counts.get(name, 0) + 1
        sink("tool_call", {"name": name, "args": kwargs, "is_error": is_error})
        return payload

    return StructuredTool.from_function(
        func=_run,
        name=tool.name,
        description=tool.description,      # the description IS the prompt
        args_schema=tool.input_schema,     # reused, not re-declared
    )


def lc_tools(names: list[str], call_counts: dict[str, int], approvals: set[str],
             sink: Callable[[str, dict], None]) -> list[StructuredTool]:
    """The tool subset for one agent. Narrow sets are a design decision — fewer
    tools means less context per call, better selection, smaller blast radius."""
    return [guarded(n, call_counts, approvals, sink) for n in names]


# ---------------------------------------------------------------------------
# The chat model
# ---------------------------------------------------------------------------

def build_chat_model(tool_names: list[str], call_counts: dict[str, int],
                     approvals: set[str], sink: Callable[[str, dict], None]):
    """
    `ChatAnthropic` bound to this agent's tools, or None when no key is set.

    Returning None rather than raising is what keeps the no-API-key promise: the
    graph nodes fall back to the same deterministic mock the native engine uses,
    so the eval suite scores routing, handoff, guardrails and budgets without
    model variance in the signal. `bind_tools` is the LangChain call that
    attaches the schemas to every request — it is the exact equivalent of
    passing `tools=` to the Anthropic SDK, and it is worth knowing that is all
    it does.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        return None
    model = ChatAnthropic(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5"),
        temperature=0.0,                   # you cannot regression-test a moving target
        max_tokens=1200,
    )
    return model.bind_tools(lc_tools(tool_names, call_counts, approvals, sink))
