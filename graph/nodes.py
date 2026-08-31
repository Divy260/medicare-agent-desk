"""
The nodes.

A LangGraph node is just a function `state -> partial state`. That is the whole
abstraction, and it is worth saying plainly because the framework's reputation
suggests something heavier. What the framework adds is not the node, it is what
surrounds it: conditional edges, checkpointed state, interrupts, and a
control-flow graph you can draw and reason about instead of a call stack you
cannot.

Division of labour in this engine, which is the design decision to defend:

    LangGraph owns INTER-agent control flow
        who runs, in what order, who hands to whom, where a human interrupts.

    agents/base.py owns INTRA-agent tool calling
        the while-loop, the iteration cap, the token budget, the wall-clock cap,
        the 429-vs-4xx retry policy, truncation handling.

The prebuilt `create_react_agent` would replace the second column. It is not
used here, and that is deliberate rather than ignorant: this desk has to enforce
a token budget and a specific retry policy, and the loop that does it already
exists, is traced, and is covered by the eval suite. Reaching for a prebuilt
agent means adopting its budget semantics instead of yours. Use it to start;
replace it the moment your failure modes stop matching its defaults.
"""

from __future__ import annotations

import re
import time

from langgraph.types import Command, interrupt

from agents.roster import AGENTS
from graph.state import MAX_HANDOFFS, DeskState, Finding
from guardrails.rules import check_input, check_output
from observability.trace import Trace
from orchestrator import ROUTES, classify_with_model

# Reused verbatim from the native engine. The routing *rules* are not a property
# of the orchestration framework, so porting the engine must not fork them —
# otherwise the two engines drift and the eval suite silently stops comparing
# like with like.
ENROLLMENT_RE = next(p for n, p, _ in ROUTES if n == "enrollment")
COVERAGE_RE = next(p for n, p, _ in ROUTES if n == "coverage")


def _now(state: DeskState) -> float:
    """Seconds since the run began — not since this node began."""
    return round(time.perf_counter() - state["started"], 3)


def _ev(kind: str, state: DeskState, **data) -> dict:
    return {"t": _now(state), "kind": kind, **data}


def _rebase(events: list[dict], offset: float) -> list[dict]:
    """
    Shift a specialist's Trace onto the graph's clock.

    agents/base.py starts a fresh Trace per agent, so its events are relative to
    that agent's own zero. Concatenating two of them without this produces a
    timeline where the clock runs backwards halfway down — the kind of detail
    nobody notices until they are reading a trace at 2am trying to work out
    which call was slow.
    """
    return [{**e, "t": round(e["t"] + offset, 3)} for e in events]


# ===========================================================================
# 1. Input guardrail — before any model call, so an injection costs nothing
# ===========================================================================

BLOCK_REPLIES = {
    "prompt_injection":
        "I can't act on that request. If you have a question about plan "
        "benefits or enrollment, I'm happy to help.",
    "out_of_scope":
        "That's outside what this desk covers. I can help with plan benefits, "
        "policy status and enrollment periods.",
    "input_too_long":
        "That message is too long for me to process. Could you send the "
        "specific question on its own?",
}


def input_guard(state: DeskState) -> DeskState:
    verdict = check_input(state["message"])
    if verdict.allowed:
        return {"trace_events": [_ev("node", state, name="input_guard")]}

    events = [_ev("node", state, name="input_guard"),
              _ev("guardrail", state, layer="input", rule=verdict.rule,
                  reason=verdict.reason)]

    # A blocked request is not always a dead end. Regulated advice is exactly
    # what the escalation agent exists for, so the guardrail picks the ROUTE
    # rather than ending the conversation. Getting this wrong in either
    # direction is expensive: answer it and you have a regulatory finding,
    # refuse it flatly and the desk is useless for its actual users.
    if verdict.rule == "regulated_advice":
        events.append(_ev("route", state, to="escalation",
                          why="guardrail: regulated_advice"))
        return {"route": "escalation", "route_why": "guardrail: regulated_advice",
                "blocked_by": verdict.rule, "trace_events": events}

    return {"route": "blocked", "blocked_by": verdict.rule,
            "answer": BLOCK_REPLIES.get(verdict.rule, "I can't help with that."),
            "trace_events": events}


# ===========================================================================
# 2. Supervisor — rules first, model for the tail
# ===========================================================================

def supervisor(state: DeskState) -> DeskState:
    """
    The router, as a graph node.

    Rules run before the classifier because roughly 70% of support traffic is
    pattern-recognisable, and spending a model call to label it is latency and
    money for nothing. A regex also never hallucinates and a wrong route is
    reproducible, which means fixable.
    """
    message = state["message"]

    for name, pattern, why in ROUTES:
        if pattern.search(message):
            return {"route": name, "route_why": why,
                    "trace_events": [_ev("node", state, name="supervisor"),
                                     _ev("route", state, to=name, why=why)]}

    trace = Trace()
    name = classify_with_model(message, trace)      # fails safe to "escalation"
    return {
        "route": name, "route_why": "model classifier",
        "input_tokens": state.get("input_tokens", 0) + trace.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + trace.output_tokens,
        "trace_events": [_ev("node", state, name="supervisor"),
                         _ev("route", state, to=name, why="model classifier")],
    }


# ===========================================================================
# 3. Specialists — each runs the native agent loop, then decides about handoff
# ===========================================================================

def _run_specialist(state: DeskState, name: str) -> tuple[Finding, list[dict], Trace]:
    offset = _now(state)                     # where this node starts on the run clock
    trace = Trace()
    result = AGENTS[name].run(state["message"], trace=trace,
                              approvals=set(state.get("approvals", [])))
    finding = Finding(agent=name, claim=result.answer, tools=trace.tools_called())
    return finding, _rebase(trace.events, offset), trace


def _specialist_update(state: DeskState, name: str, finding: Finding,
                       events: list[dict], trace: Trace) -> dict:
    return {
        "findings": [finding],
        "visited": [name],
        "input_tokens": state.get("input_tokens", 0) + trace.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + trace.output_tokens,
        "trace_events": events,
    }


def coverage(state: DeskState) -> Command:
    """
    The coverage specialist, with a HANDOFF.

    "For P-1001, is the Part B deductible covered, and when can she switch?" is
    two questions wearing one coat. A router has to pick a single destination
    and will answer half of it. This node answers its half, keeps what it found
    on the shared blackboard, and transfers control — the enrollment agent then
    sees the coverage findings rather than starting cold.

    `Command(goto=...)` is the handoff primitive: a node returning it both
    updates state and names the next node, overriding the static edge. That is
    the difference between a graph and a chain — the *data* decides the path.
    """
    finding, events, trace = _run_specialist(state, "coverage")
    update = _specialist_update(state, "coverage", finding, events, trace)
    update["trace_events"] = [_ev("node", state, name="coverage")] + update["trace_events"]

    needs_enrollment = bool(ENROLLMENT_RE.search(state["message"]))
    can_hand_off = (needs_enrollment
                    and "enrollment" not in state.get("visited", [])
                    and state.get("handoffs", 0) < MAX_HANDOFFS)

    if can_hand_off:
        update["handoffs"] = state.get("handoffs", 0) + 1
        update["trace_events"] = update["trace_events"] + [
            _ev("handoff", state, frm="coverage", to="enrollment",
                why="question also asks about an enrollment window")]
        return Command(goto="enrollment", update=update)

    return Command(goto="compose", update=update)


def enrollment(state: DeskState) -> Command:
    """The enrollment specialist. Symmetric handoff, and the same bound applies —
    two nodes that can each hand to the other will ping-pong without one."""
    finding, events, trace = _run_specialist(state, "enrollment")
    update = _specialist_update(state, "enrollment", finding, events, trace)
    update["trace_events"] = [_ev("node", state, name="enrollment")] + update["trace_events"]

    needs_coverage = bool(COVERAGE_RE.search(state["message"]))
    if (needs_coverage and "coverage" not in state.get("visited", [])
            and state.get("handoffs", 0) < MAX_HANDOFFS):
        update["handoffs"] = state.get("handoffs", 0) + 1
        update["trace_events"] = update["trace_events"] + [
            _ev("handoff", state, frm="enrollment", to="coverage",
                why="question also asks what the plan covers")]
        return Command(goto="coverage", update=update)

    return Command(goto="compose", update=update)


# ===========================================================================
# 4. Escalation — the human-in-the-loop interrupt
# ===========================================================================

APPROVAL_TOOL = "escalate_to_licensed_agent"


def escalation(state: DeskState) -> DeskState:
    """
    Pause the graph and wait for a human.

    `interrupt()` is the reason a checkpointer is not optional. It raises,
    LangGraph persists the state exactly as it stands, and the run ends. Later —
    a different process, a different day — resuming with `Command(resume=True)`
    replays this node and `interrupt()` RETURNS that value instead of raising.

    That replay is the gotcha worth knowing: the node re-executes from its first
    line, so anything above the interrupt happens twice. Side effects belong
    below it, or in a node of their own. Here there is nothing above it but a
    dict literal, which is deliberate.
    """
    approvals = set(state.get("approvals", []))

    if APPROVAL_TOOL not in approvals:
        decision = interrupt({
            "tool": APPROVAL_TOOL,
            "reason": state.get("blocked_by") or "unresolved",
            "question": state["message"],
            "prompt": "A licensed-agent hand-off has downstream effects. Approve?",
        })
        if decision:
            approvals.add(APPROVAL_TOOL)

    offset = _now(state)
    trace = Trace()
    result = AGENTS["escalation"].run(state["message"], trace=trace,
                                      approvals=approvals)
    return {
        "findings": [Finding(agent="escalation", claim=result.answer,
                             tools=trace.tools_called())],
        "visited": ["escalation"],
        "approvals": sorted(approvals),
        "input_tokens": state.get("input_tokens", 0) + trace.input_tokens,
        "output_tokens": state.get("output_tokens", 0) + trace.output_tokens,
        "trace_events": ([_ev("node", state, name="escalation")]
                         + _rebase(trace.events, offset)),
    }


# ===========================================================================
# 5. Compose — the fan-in
# ===========================================================================

# Sentence splitting, with two exceptions that both cost real correctness here.
#
#   "[" is NOT a sentence start, so a trailing "[benefits-kb/...]" citation stays
#   attached to the claim it justifies. Splitting them lets the dedup drop a
#   citation on its own and leave a factual claim unsourced — the one output the
#   guardrails exist to prevent.
#
#   A single capital letter before the period is an INITIAL, not a sentence end.
#   Every holder in the policy admin system is recorded as "A. Rivera" style, so
#   without this every composed answer splits a member's name in half.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$])")
_ENDS_WITH_INITIAL = re.compile(r"(?:^|[\s(\[])[A-Z]\.$")


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(text.strip()):
        part = part.strip()
        if not part:
            continue
        if out and _ENDS_WITH_INITIAL.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"       # glue "A." back onto "Rivera holds…"
        else:
            out.append(part)
    return out


def _norm(sentence: str) -> str:
    """Compare on content, ignoring whitespace and case."""
    return " ".join(sentence.lower().split())


def compose(state: DeskState) -> DeskState:
    """
    Merge the blackboard into one answer.

    Deterministic string assembly, not a model call. A synthesis step that asks
    a model to "combine these findings" is a fresh opportunity to hallucinate a
    claim that no tool produced, and it would sit downstream of every guardrail
    that made the findings trustworthy. When the parts are already sourced,
    joining them is not a job for an LLM.

    Deduplication happens at SENTENCE level, not per finding. Two specialists
    handed the same question both call lookup_policy, so both open with the same
    policy record — concatenating their answers states it twice and reads like a
    machine talking to itself. Comparing whole answers does not catch that,
    because the answers diverge after the first sentence.
    """
    findings = state.get("findings", [])
    if not findings:
        return {"answer": "I don't have that in my sources.",
                "trace_events": [_ev("node", state, name="compose")]}

    kept: list[str] = []
    seen: set[str] = set()
    dropped = 0
    for f in findings:
        for sentence in _sentences(f["claim"]):
            key = _norm(sentence)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            kept.append(sentence)

    return {"answer": " ".join(kept),
            "trace_events": [_ev("node", state, name="compose",
                                 findings=len(findings),
                                 agents=[f["agent"] for f in findings],
                                 duplicate_sentences_dropped=dropped)]}


# ===========================================================================
# 6. Output guardrail — last thing before the user sees it
# ===========================================================================

def output_guard(state: DeskState) -> DeskState:
    answer = state.get("answer", "")
    tools = [t for f in state.get("findings", []) for t in f["tools"]]

    verdict = check_output(answer, tools)
    events = [_ev("node", state, name="output_guard")]
    if verdict.rule:
        events.append(_ev("guardrail", state, layer="output", rule=verdict.rule,
                          reason=verdict.reason, severity=verdict.severity))
    if not verdict.allowed:
        return {"answer": "I can't provide that response. Let me connect you with "
                          "a licensed agent who can help properly.",
                "output_rule": verdict.rule, "trace_events": events}
    return {"answer": verdict.rewritten or answer,
            "output_rule": verdict.rule, "trace_events": events}
