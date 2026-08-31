"""
The graph.

    START -> input_guard -> (blocked?) -----------------------> output_guard -> END
                         \
                          -> supervisor -> coverage   --+--> compose -> output_guard -> END
                                        -> enrollment --+
                                        -> escalation --+
                                              ^
                                        coverage <-> enrollment
                                        (handoff, via Command(goto=...))

Three things this buys over the straight-line engine in orchestrator.py, and
they are the three worth naming in any conversation about why a framework:

  1. CONDITIONAL EDGES are declared, not implied. `route_from_guard` and
     `route_from_supervisor` below are the entire control flow, in one place,
     readable without tracing a call stack. In the native engine the same logic
     is spread across an if-chain and a function call.

  2. STATE IS CHECKPOINTED. Every superstep is persisted, so a run can be
     paused, inspected, resumed, or replayed from any point. That is what makes
     `interrupt()` — the human approval gate — possible at all, and it is the
     single feature that most justifies the dependency.

  3. THE TOPOLOGY IS DATA. `.get_graph().draw_mermaid()` prints the real
     control flow rather than a diagram in a README that drifted three commits
     ago.

What it does NOT buy: correctness, guardrails, evals, or cost control. Those
are the same code as before, imported unchanged. A framework moves the
orchestration; it does not do the engineering.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph import nodes
from graph.state import DeskState


# ---------------------------------------------------------------------------
# Conditional edges — the whole control flow, in two functions
# ---------------------------------------------------------------------------

def route_from_guard(state: DeskState) -> str:
    """Blocked input skips every model call and goes straight to the output
    layer. A prompt injection therefore costs nothing, which is the point of
    putting the input guardrail first."""
    if state.get("route") == "blocked":
        return "output_guard"
    if state.get("route") == "escalation":       # regulated advice, pre-routed
        return "escalation"
    return "supervisor"


def route_from_supervisor(state: DeskState) -> str:
    return state.get("route") or "escalation"    # fail safe: uncertain -> human


# ---------------------------------------------------------------------------
def build(checkpointer=None):
    """
    Compile the desk graph.

    The checkpointer defaults to `MemorySaver`, which is right for tests and a
    demo and wrong for production — it is per-process, so a resume only works
    inside the same run. Swap in `PostgresSaver`/`RedisSaver` and the approval
    interrupt survives a restart and a load balancer. Nothing else changes,
    which is the argument for putting durable state behind an interface.
    """
    g = StateGraph(DeskState)

    g.add_node("input_guard", nodes.input_guard)
    g.add_node("supervisor", nodes.supervisor)
    # `destinations` is documentation for the drawing, not enforcement — a node
    # returning Command(goto=...) needs it for the topology to render honestly.
    g.add_node("coverage", nodes.coverage,
               destinations=("enrollment", "compose"))
    g.add_node("enrollment", nodes.enrollment,
               destinations=("coverage", "compose"))
    g.add_node("escalation", nodes.escalation)
    g.add_node("compose", nodes.compose)
    g.add_node("output_guard", nodes.output_guard)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_from_guard,
                            ["supervisor", "escalation", "output_guard"])
    g.add_conditional_edges("supervisor", route_from_supervisor,
                            ["coverage", "enrollment", "escalation"])
    g.add_edge("escalation", "compose")
    g.add_edge("compose", "output_guard")
    g.add_edge("output_guard", END)

    return g.compile(checkpointer=checkpointer or MemorySaver())


DESK_GRAPH = build()


def mermaid_diagram() -> str:
    """
    The real topology, generated from the compiled graph rather than drawn by
    hand. Mermaid rather than ASCII because it needs no extra dependency and
    GitHub renders it inline — so the diagram in the README is generated output,
    not a picture that drifted three commits ago.
    """
    return DESK_GRAPH.get_graph().draw_mermaid()
