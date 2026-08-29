"""
The agent loop.

This is the whole of "AI agent engineering" — everything else in the repo is a
decoration on it.

    messages = [user request]
    while True:
        response = model(messages, tools=TOOLS)

        if response.stop_reason != "tool_use":
            return response.text                # done

        messages.append(assistant turn with tool_use blocks)

        results = []
        for block in response.tool_uses():
            results.append(run_tool(block))      # YOUR code decides

        messages.append(user turn with tool_result blocks)

The decorations that make it production-shaped, all present below:

    budgets       iteration cap + token cap + wall-clock cap (each fails differently)
    retries       exponential backoff on transient errors, never on a 4xx
    guardrails    checked before every tool executes and on the final answer
    tracing       every model call and tool call, with timing and tokens
    truncation    stop_reason == "max_tokens" handled explicitly, never shown as complete
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

from agents.llm import get_client
from guardrails.rules import check_tool_call, check_output
from observability.trace import Trace
from tools.registry import TOOLS_BY_NAME, schemas

MAX_ITERATIONS = 8
MAX_TOKEN_BUDGET = 40_000
MAX_WALL_CLOCK_S = 60


@dataclass
class AgentResult:
    answer: str
    trace: Trace
    stopped_early: bool = False


class Agent:
    """
    One agent = a system prompt + a subset of the tools.

    Giving each agent only the tools it needs is not tidiness. Fewer tools means
    less context per call, better tool selection, and a smaller blast radius if
    the model is confused or the input is adversarial.
    """

    def __init__(self, name: str, system: str, tool_names: list[str],
                 prompt_version: str = "v1", client=None):
        self.name = name
        self.system = system
        self.tool_names = tool_names
        self.prompt_version = prompt_version
        self.client = client or get_client()

    # ---------------------------------------------------------------
    def _call_model_with_retry(self, messages, trace: Trace):
        delay, last = 1.0, None
        for _ in range(4):
            try:
                return self.client.create(
                    system=self.system,
                    messages=messages,
                    tools=schemas(self.tool_names),
                    max_tokens=1200,
                    temperature=0.0,
                )
            except Exception as exc:                          # noqa: BLE001
                # A real implementation distinguishes 429/5xx (retry) from 4xx
                # (our bug — never retry, log and alert).
                status = getattr(exc, "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    trace.log("model_error", fatal=True, detail=str(exc)[:200])
                    raise
                last = exc
                trace.log("model_retry", detail=str(exc)[:200])
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(f"model unavailable after retries: {last}")

    # ---------------------------------------------------------------
    def run(self, user_message: str, trace: Trace | None = None,
            approvals: set[str] | None = None) -> AgentResult:
        trace = trace or Trace()
        trace.agent = self.name
        trace.prompt_version = self.prompt_version
        trace.model = getattr(self.client, "model", "unknown")
        approvals = approvals or set()

        messages: list[dict] = [{"role": "user", "content": user_message}]
        call_counts: dict[str, int] = {}

        for iteration in range(1, MAX_ITERATIONS + 1):

            # --- budgets, checked BEFORE spending more --------------------
            if trace.total_tokens > MAX_TOKEN_BUDGET:
                trace.log("budget_exceeded", kind_detail="tokens")
                return AgentResult(
                    "I've reached the token budget for this request. Here's what I "
                    "found before stopping — please narrow the question.",
                    trace, stopped_early=True)
            if trace.elapsed > MAX_WALL_CLOCK_S:
                trace.log("budget_exceeded", kind_detail="wall_clock")
                return AgentResult(
                    "This request took longer than the time budget allows. Please "
                    "try a narrower question.", trace, stopped_early=True)

            response = self._call_model_with_retry(messages, trace)
            trace.record_usage(response.usage.input_tokens, response.usage.output_tokens)
            trace.log("model_call", iteration=iteration, stop_reason=response.stop_reason)

            # --- truncation is not success -------------------------------
            if response.stop_reason == "max_tokens":
                trace.log("truncated")
                return AgentResult(
                    "My answer was cut off before it finished. Please ask for one "
                    "part of that at a time.", trace, stopped_early=True)

            # --- finished ------------------------------------------------
            if response.stop_reason != "tool_use":
                answer = response.text()
                verdict = check_output(answer, trace.tools_called())
                if verdict.rule:
                    trace.log("guardrail", layer="output", rule=verdict.rule,
                              reason=verdict.reason, severity=verdict.severity)
                if not verdict.allowed:
                    return AgentResult(
                        "I can't provide that response. Let me connect you with a "
                        "licensed agent who can help properly.", trace)
                trace.log("final_answer")
                return AgentResult(verdict.rewritten or answer, trace)

            # --- execute the requested tools -----------------------------
            messages.append({
                "role": "assistant",
                "content": [
                    ({"type": "text", "text": b.text} if b.type == "text"
                     else {"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                    for b in response.content if b.type != "text" or b.text.strip()
                ],
            })

            results = []
            for block in response.tool_uses():
                payload, is_error = self._execute(block, call_counts, approvals, trace)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(payload),
                    "is_error": is_error,
                })

            # Tool results are framed as coming from the environment, not the
            # assistant — hence role "user".
            messages.append({"role": "user", "content": results})

        trace.log("iteration_limit")
        return AgentResult(
            "I wasn't able to finish this within the step limit. Here's where I "
            "got to — could you narrow the question, or shall I pass this to a "
            "licensed agent?", trace, stopped_early=True)

    # ---------------------------------------------------------------
    def _execute(self, block, call_counts: dict[str, int],
                 approvals: set[str], trace: Trace) -> tuple[dict, bool]:
        """
        Guardrail, then execute. A tool failure is DATA, not an exception — it
        goes back to the model so it can adapt. An agent that crashes on the
        first 500 from a downstream API is a script with extra steps.
        """
        verdict = check_tool_call(block.name, block.input, call_counts,
                                  TOOLS_BY_NAME, human_approved=block.name in approvals)
        if not verdict.allowed:
            trace.log("guardrail", layer="tool", rule=verdict.rule,
                      reason=verdict.reason, tool=block.name)
            trace.log("tool_call", name=block.name, args=block.input, is_error=True,
                      blocked_by=verdict.rule)
            return {"error": verdict.rule, "detail": verdict.reason}, True

        tool = TOOLS_BY_NAME[block.name]
        try:
            payload = tool.fn(**block.input)
            is_error = "error" in payload
        except TypeError as exc:
            payload, is_error = {"error": "bad_arguments", "detail": str(exc)}, True
        except Exception as exc:                              # noqa: BLE001
            payload, is_error = {"error": type(exc).__name__, "detail": str(exc)}, True

        call_counts[block.name] = call_counts.get(block.name, 0) + 1
        trace.log("tool_call", name=block.name, args=block.input, is_error=is_error)
        return payload, is_error
