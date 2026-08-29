"""
The tool layer.

Two things live here, and keeping them together is deliberate:

  1. The JSON Schema the model reads to decide *whether* to call a tool.
  2. The Python function that actually runs, and the guardrail policy around it.

The model never executes anything. It emits a JSON request; this module decides
whether to honour it. That boundary is the single most important security
property of tool use, and it is where authorisation, argument validation and
data minimisation belong.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from data.knowledge import POLICIES, PLAN_RULES, ENROLLMENT_PERIODS

# Fields a model has no business seeing. Stripped at the tool boundary, before
# the record is serialised into the context window. The cheapest way to keep PII
# out of a model is not to send it.
MINIMISED_FIELDS = {"member_phone"}


# ===========================================================================
# Implementations
# ===========================================================================

def lookup_policy(policy_id: str) -> dict:
    """Stand-in for a REST call to a policy admin system."""
    time.sleep(0.03)
    record = POLICIES.get(policy_id.upper().strip())
    if record is None:
        return {
            "error": "not_found",
            "policy_id": policy_id,
            "hint": f"Known policy IDs in this environment: {', '.join(POLICIES)}",
        }
    safe = {k: v for k, v in record.items() if k not in MINIMISED_FIELDS}
    return {"policy_id": policy_id.upper().strip(), **safe,
            "source": f"policy-admin/{policy_id.upper().strip()}"}


def check_coverage(plan: str, benefit: str) -> dict:
    """Stand-in for a benefits knowledge-base lookup."""
    rules = PLAN_RULES.get(plan)
    if rules is None:
        return {"error": "unknown_plan", "plan": plan,
                "known_plans": list(PLAN_RULES)}
    if benefit not in rules:
        return {"error": "no_rule_on_file", "plan": plan, "benefit": benefit,
                "available_benefits": list(rules)}
    return {
        "plan": plan,
        "benefit": benefit,
        "coverage": rules[benefit],
        "source": f"benefits-kb/{plan.lower().replace(' ', '-')}#{benefit}",
    }


def enrollment_window(period: str) -> dict:
    """Look up a Medicare enrollment period by code."""
    entry = ENROLLMENT_PERIODS.get(period.upper().strip())
    if entry is None:
        return {"error": "unknown_period", "period": period,
                "known_periods": list(ENROLLMENT_PERIODS)}
    return {"period": period.upper(), **entry,
            "source": f"enrollment-calendar#{period.lower()}"}


def calculate(expression: str) -> dict:
    """
    Models are unreliable at arithmetic because numbers tokenise unpredictably.
    Give them a calculator rather than hoping.

    Note the character allow-list: never pass model output to a bare eval().
    """
    if not re.fullmatch(r"[0-9\s+\-*/().]+", expression):
        return {"error": "illegal_characters",
                "detail": "Only digits and + - * / ( ) . are permitted."}
    try:
        value = eval(expression, {"__builtins__": {}})       # noqa: S307
    except Exception as exc:                                  # noqa: BLE001
        return {"error": type(exc).__name__, "detail": str(exc)}
    return {"expression": expression, "result": round(float(value), 2)}


def escalate_to_licensed_agent(reason: str, summary: str) -> dict:
    """
    The hand-off tool. Anything that constitutes individualised advice — which
    plan to buy, whether to switch, whether a specific condition is covered for
    a specific person — must route to a licensed human. That is a regulatory
    boundary, not a quality preference.

    Marked requires_approval so it never fires silently in a demo.
    """
    return {
        "escalated": True,
        "reason": reason,
        "summary": summary,
        "queue": "licensed-agent-desk",
        "sla_minutes": 30,
    }


# ===========================================================================
# Schemas + policy
# ===========================================================================

@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., dict]
    max_calls_per_run: int = 10
    requires_approval: bool = False

    def schema(self) -> dict:
        return {"name": self.name,
                "description": self.description,
                "input_schema": self.input_schema}


# The DESCRIPTION is a prompt. It is the only thing telling the model when to
# reach for a tool, and vague descriptions cause more wrong tool selection than
# any other single factor. Each one below says what it does, when to use it,
# when NOT to, and what to do about missing arguments.

TOOLS: list[Tool] = [
    Tool(
        name="lookup_policy",
        description=(
            "Retrieve a policy record by its policy ID. Use this whenever the "
            "user mentions a policy number, BEFORE answering anything about that "
            "policy. Returns holder name, plan name, status, monthly premium, "
            "state and effective date. It does NOT return member contact "
            "details — do not ask it for a phone number. If the user has not "
            "given a policy ID, ask them for it rather than guessing one."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "policy_id": {
                    "type": "string",
                    "description": "Policy identifier in the form P-nnnn, e.g. P-1001",
                },
            },
            "required": ["policy_id"],
        },
        fn=lookup_policy,
        max_calls_per_run=6,
    ),
    Tool(
        name="check_coverage",
        description=(
            "Look up whether a specific benefit is covered under a specific plan. "
            "You must know the exact plan name first — call lookup_policy if you "
            "only have a policy ID. Never state coverage from memory; always call "
            "this. Cite the returned `source` value in your answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "plan": {
                    "type": "string",
                    "description": "Exact plan name as returned by lookup_policy",
                },
                "benefit": {
                    "type": "string",
                    "enum": [
                        "part_a_deductible", "part_b_deductible",
                        "part_b_coinsurance", "part_b_excess_charges",
                        "skilled_nursing_coinsurance", "foreign_travel_emergency",
                        "prescription_drugs", "routine_dental_vision_hearing",
                    ],
                    "description": "The benefit to check",
                },
            },
            "required": ["plan", "benefit"],
        },
        fn=check_coverage,
        max_calls_per_run=10,
    ),
    Tool(
        name="enrollment_window",
        description=(
            "Look up the dates and rules for a Medicare enrollment period. Use "
            "this for any question about when someone can join, switch or drop a "
            "plan. Period codes: IEP (initial, around age 65), AEP (annual, "
            "Oct 15 - Dec 7), MA_OEP (Medicare Advantage open enrollment, "
            "Jan 1 - Mar 31), MEDIGAP_OE (Medigap guaranteed issue), SEP "
            "(special enrollment after a life event)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["IEP", "AEP", "MA_OEP", "MEDIGAP_OE", "SEP"],
                },
            },
            "required": ["period"],
        },
        fn=enrollment_window,
        max_calls_per_run=6,
    ),
    Tool(
        name="calculate",
        description=(
            "Evaluate an arithmetic expression. Use this for ALL arithmetic, "
            "including simple multiplication — do not compute in your head. "
            "Example input: '148.20 * 12'."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Digits and + - * / ( ) . only",
                },
            },
            "required": ["expression"],
        },
        fn=calculate,
        max_calls_per_run=6,
    ),
    Tool(
        name="escalate_to_licensed_agent",
        description=(
            "Hand the conversation to a licensed human agent. Use this whenever "
            "the request asks for individualised advice — which plan someone "
            "should buy, whether they should switch, or whether a named medical "
            "condition is covered for a named person. Also use it when the tools "
            "cannot answer and the caller needs a human."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "enum": ["regulated_advice", "unresolved", "member_request"],
                },
                "summary": {
                    "type": "string",
                    "description": "One or two sentences the human agent will read first",
                },
            },
            "required": ["reason", "summary"],
        },
        fn=escalate_to_licensed_agent,
        max_calls_per_run=1,
        requires_approval=True,      # never fires without a human saying yes
    ),
]

TOOLS_BY_NAME: dict[str, Tool] = {t.name: t for t in TOOLS}


def schemas(names: list[str] | None = None) -> list[dict]:
    """The tool list sent to the model. Each agent gets only the tools it needs."""
    if names is None:
        return [t.schema() for t in TOOLS]
    return [TOOLS_BY_NAME[n].schema() for n in names if n in TOOLS_BY_NAME]
