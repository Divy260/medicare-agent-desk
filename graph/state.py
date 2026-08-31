"""
The shared state — the thing that makes this multi-agent rather than sequential.

A chain passes one agent's output string into the next agent's input string.
That is not multi-agent, it is a pipeline with extra latency: the second agent
cannot see *why* the first concluded anything, and nothing accumulates.

A graph passes a typed, shared blackboard. Every node reads all of it and
appends to part of it. That difference is what buys you:

  - fan-in    two specialists contribute findings, one node composes the answer
  - handoff   an agent transfers control mid-run without losing what it learned
  - resume    the state is checkpointed, so a run can pause for a human and
              continue hours later in a different process

The reducers below are the load-bearing detail. Without them, LangGraph's
default behaviour for a state key is LAST WRITE WINS — so when the coverage
agent hands off to enrollment, enrollment's return value would silently erase
coverage's findings. `Annotated[list, operator.add]` makes the update an append
instead. Nearly every "my multi-agent graph loses data" bug is this.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Literal, TypedDict

# How many times control may pass between specialists in a single turn. Two
# agents that can each hand off to the other will ping-pong forever given the
# right input; this is the same argument as the iteration cap in
# agents/base.py, applied one level up. Every loop in an agent system needs a
# bound, and each bound needs its own reason for existing.
MAX_HANDOFFS = 3

AgentName = Literal["coverage", "enrollment", "escalation"]


class Finding(TypedDict):
    """
    One sourced claim produced by one agent.

    `source` is not decoration. The desk's hard rule is that every factual claim
    cites the tool that produced it, so a finding without a source is a finding
    the compose node must not use. Carrying the citation with the claim — rather
    than reconstructing it later from the trace — is what makes that enforceable.
    """
    agent: str
    claim: str
    tools: list[str]


class DeskState(TypedDict, total=False):
    # ---- input, written once ------------------------------------------
    message: str
    approvals: list[str]

    # ---- routing ------------------------------------------------------
    route: str
    route_why: str
    blocked_by: str | None

    # ---- the blackboard, appended to by every specialist ---------------
    # operator.add on a list is list concatenation. Order is preserved, so the
    # compose node can present findings in the order they were established.
    findings: Annotated[list[Finding], operator.add]
    visited: Annotated[list[str], operator.add]
    handoffs: int

    # ---- observability -------------------------------------------------
    # One clock for the whole run. Each specialist node runs the native agent
    # loop, which keeps its own Trace with its own zero — merging those events
    # without rebasing them onto a shared origin produces a timeline that goes
    # backwards, which is worse than no timeline at all.
    started: float
    # Plain dicts, deliberately. Graph state is checkpointed, and in production
    # that checkpointer is Postgres or Redis rather than the in-memory one — so
    # anything living here has to survive serialisation. A Trace object would
    # not. observability/trace.py::render_timeline renders these identically to
    # a Trace, which is how one trace format serves two engines.
    trace_events: Annotated[list[dict], operator.add]
    input_tokens: int
    output_tokens: int

    # ---- output --------------------------------------------------------
    answer: str
    output_rule: str


def new_state(message: str, approvals: set[str] | None = None) -> DeskState:
    return DeskState(
        started=time.perf_counter(),
        message=message,
        approvals=sorted(approvals or set()),
        route="",
        route_why="",
        blocked_by=None,
        findings=[],
        visited=[],
        handoffs=0,
        trace_events=[],
        input_tokens=0,
        output_tokens=0,
        answer="",
        output_rule="",
    )
