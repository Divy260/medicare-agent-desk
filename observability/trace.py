"""
Tracing.

You cannot debug, evaluate or cost-manage an agent you cannot see. This is the
raw material for the eval suite, the cost dashboard, and every postmortem — so
it is written on day one rather than retrofitted after an incident.

Everything recorded here would go to OpenTelemetry / Application Insights in a
real deployment. The shape is the same.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# Illustrative per-token prices. Real numbers come from the provider's pricing
# page and change; the point is that cost is measured, not estimated afterwards.
INPUT_PRICE = 3.00 / 1_000_000
OUTPUT_PRICE = 15.00 / 1_000_000


@dataclass
class Trace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str = ""
    prompt_version: str = ""
    model: str = ""
    events: list[dict] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    started: float = field(default_factory=time.perf_counter)

    # ---- recording ------------------------------------------------------
    def log(self, kind: str, **data: Any) -> None:
        self.events.append({
            "t": round(time.perf_counter() - self.started, 3),
            "kind": kind,
            **data,
        })

    def record_usage(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

    # ---- reading --------------------------------------------------------
    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return self.input_tokens * INPUT_PRICE + self.output_tokens * OUTPUT_PRICE

    def tools_called(self) -> list[str]:
        """Ordered list of tool names. This is what trajectory evals score."""
        return [e["name"] for e in self.events if e["kind"] == "tool_call"]

    def guardrails_triggered(self) -> list[str]:
        return [e["rule"] for e in self.events if e["kind"] == "guardrail"]

    def summary(self) -> dict:
        tool_events = [e for e in self.events if e["kind"] == "tool_call"]
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "iterations": len([e for e in self.events if e["kind"] == "model_call"]),
            "tool_calls": len(tool_events),
            "tool_errors": len([e for e in tool_events if e.get("is_error")]),
            "tools_called": self.tools_called(),
            "guardrails": self.guardrails_triggered(),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "elapsed_s": round(self.elapsed, 2),
            "cost_usd": round(self.cost_usd, 6),
        }

    def timeline(self) -> str:
        """Human-readable trace, for the Streamlit sidebar and for debugging."""
        return render_timeline(self.events)


# ===========================================================================
# Rendering, as a free function
# ===========================================================================
# This lives outside the class because the LangGraph engine in `graph/` does not
# carry a Trace object through its state — graph state gets checkpointed to
# durable storage, so everything in it must be plain serialisable data. It
# accumulates the same event dicts in a list and renders them with this. One
# trace format, two engines, so the Streamlit panel and the evals do not care
# which one produced the run.

def render_timeline(events: list[dict]) -> str:
    lines = []
    for e in events:
        prefix = f"{e['t']:>6.2f}s  "
        kind = e["kind"]
        if kind == "model_call":
            lines.append(f"{prefix}model  iter={e.get('iteration')} "
                         f"stop={e.get('stop_reason')}")
        elif kind == "tool_call":
            mark = "ERR " if e.get("is_error") else "ok  "
            lines.append(f"{prefix}tool   {mark}{e['name']}"
                         f"({json.dumps(e.get('args', {}))})")
        elif kind == "guardrail":
            lines.append(f"{prefix}guard  [{e.get('layer')}] {e.get('rule')} "
                         f"— {e.get('reason', '')}")
        elif kind == "route":
            lines.append(f"{prefix}route  -> {e.get('to')} ({e.get('why', '')})")
        elif kind == "final_answer":
            lines.append(f"{prefix}answer")
        elif kind == "node":
            lines.append(f"{prefix}node   {e.get('name')}")
        elif kind == "handoff":
            lines.append(f"{prefix}handoff {e.get('frm')} -> {e.get('to')} "
                         f"({e.get('why', '')})")
        elif kind == "interrupt":
            lines.append(f"{prefix}PAUSE  {e.get('name')} — {e.get('reason', '')}")
        elif kind == "resume":
            lines.append(f"{prefix}RESUME {e.get('name')} — approved={e.get('approved')}")
        else:
            lines.append(f"{prefix}{kind}  {json.dumps({k: v for k, v in e.items() if k not in ('t', 'kind')})}")
    return "\n".join(lines)
