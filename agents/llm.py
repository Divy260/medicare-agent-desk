"""
The model client, with a mock backend.

Why the mock exists
-------------------
Two reasons, and the second matters more than the first.

  1. The whole application runs and demos with no API key and no network.
  2. It makes the agent loop *testable*. A scripted, deterministic model means
     the eval suite measures the orchestration — routing, tool selection,
     guardrails, budgets — without the model's variance drowning the signal.

Set ANTHROPIC_API_KEY to use the real API; leave it unset and the mock takes
over automatically. Nothing else in the codebase changes.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5")


# ===========================================================================
# A response shape both backends produce
# ===========================================================================

@dataclass
class Block:
    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Response:
    stop_reason: str                       # end_turn | tool_use | max_tokens
    content: list[Block]
    usage: Usage = field(default_factory=Usage)
    model: str = DEFAULT_MODEL

    def text(self) -> str:
        return "".join(b.text for b in self.content if b.type == "text")

    def tool_uses(self) -> list[Block]:
        return [b for b in self.content if b.type == "tool_use"]


# ===========================================================================
# Real backend
# ===========================================================================

class ClaudeClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        from anthropic import Anthropic
        self._client = Anthropic()
        self.model = model
        self.backend = "anthropic"

    def create(self, system: str, messages: list[dict], tools: list[dict],
               max_tokens: int = 1500, temperature: float = 0.0) -> Response:
        raw = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,       # 0 — you cannot regression-test a moving target
            system=system,
            tools=tools,
            messages=messages,
        )
        blocks = []
        for b in raw.content:
            if b.type == "text":
                blocks.append(Block(type="text", text=b.text))
            elif b.type == "tool_use":
                blocks.append(Block(type="tool_use", id=b.id, name=b.name,
                                    input=dict(b.input)))
        return Response(
            stop_reason=raw.stop_reason,
            content=blocks,
            usage=Usage(raw.usage.input_tokens, raw.usage.output_tokens),
            model=raw.model,
        )


# ===========================================================================
# Mock backend
# ===========================================================================

POLICY_RE = re.compile(r"\bP-?(\d{4})\b", re.I)

BENEFIT_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"part b deduct", re.I), "part_b_deductible"),
    (re.compile(r"part a deduct", re.I), "part_a_deductible"),
    (re.compile(r"excess charge", re.I), "part_b_excess_charges"),
    (re.compile(r"foreign|travel|abroad|overseas", re.I), "foreign_travel_emergency"),
    (re.compile(r"drug|prescription|part d", re.I), "prescription_drugs"),
    (re.compile(r"dental|vision|hearing", re.I), "routine_dental_vision_hearing"),
    (re.compile(r"skilled nursing|snf", re.I), "skilled_nursing_coinsurance"),
    (re.compile(r"coinsurance", re.I), "part_b_coinsurance"),
]

PERIOD_KEYWORDS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"medigap.*(open|guarantee)|guaranteed issue|underwrit", re.I), "MEDIGAP_OE"),
    (re.compile(r"turning 65|initial enrol|\biep\b|first eligible", re.I), "IEP"),
    (re.compile(r"moved|lost (my |her |his |their )?(employer|job|coverage)"
                r"|\bsep\b|life event|special enrol", re.I), "SEP"),
    (re.compile(r"\bma[- ]?oep\b|january|march 31"
                r"|(switch|change|leave|drop)\b[^.?]{0,40}\badvantage\b", re.I), "MA_OEP"),
    (re.compile(r"\baep\b|annual enrol|october|oct 15|december", re.I), "AEP"),
]

ANNUAL_RE = re.compile(r"annual|per year|a year|yearly|12 month", re.I)

# Hints used only by the mock when it is standing in for the router's classifier.
REGULATED_ADVICE_HINT = re.compile(
    r"\bshould\b|\brecommend\b|\bbest (plan|option) for\b|\bbetter off\b"
    r"|\badvise\b|\bwhat would you do\b", re.I)
PERIOD_HINT = re.compile(
    r"\benrol|\bwhen can\b|\bwindow\b|\beligib|\bswitch\b|\bsign up\b"
    r"|\baep\b|\boep\b|\biep\b|\bsep\b|\bunderwrit", re.I)
COVERAGE_HINT = re.compile(
    r"\bcover|\bdeduct|\bpremium|\bbenefit|\bcopay|\bcoinsurance|\bplan\b"
    r"|\bpolicy\b|\bP-?\d{4}\b|\bpart [abd]\b", re.I)


class MockClient:
    """
    A rule-based stand-in that produces the same tool-calling *behaviour* as a
    real model on this domain, deterministically.

    It is deliberately not clever. It reads the conversation, decides which tool
    would obviously be needed next, and emits it. When there is nothing left to
    call, it composes an answer from whatever the tools returned. That is enough
    to exercise every branch of the orchestrator.
    """

    def __init__(self, model: str = "mock-model"):
        self.model = model
        self.backend = "mock"

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _tool_results(messages: list[dict]) -> dict[str, list[dict]]:
        """Everything the tools have returned so far, keyed by tool name."""
        out: dict[str, list[dict]] = {}
        pending: dict[str, str] = {}
        for m in messages:
            content = m.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                if btype == "tool_use":
                    bid = b["id"] if isinstance(b, dict) else b.id
                    bname = b["name"] if isinstance(b, dict) else b.name
                    pending[bid] = bname
                elif btype == "tool_result":
                    name = pending.get(b.get("tool_use_id"), "?")
                    try:
                        payload = json.loads(b.get("content", "{}"))
                    except (ValueError, TypeError):
                        payload = {}
                    out.setdefault(name, []).append(payload)
        return out

    @staticmethod
    def _user_text(messages: list[dict]) -> str:
        for m in messages:
            if m["role"] == "user" and isinstance(m.get("content"), str):
                return m["content"]
        return ""

    # -- the model -------------------------------------------------------
    def create(self, system: str, messages: list[dict], tools: list[dict],
               max_tokens: int = 1500, temperature: float = 0.0) -> Response:
        time.sleep(0.05)                       # so the UI feels real
        available = {t["name"] for t in tools}
        question = self._user_text(messages)

        # --- the router's classification call (no tools, tiny max_tokens) ---
        if not tools and "route messages" in system.lower():
            if REGULATED_ADVICE_HINT.search(question):
                label = "escalation"
            elif PERIOD_HINT.search(question):
                label = "enrollment"
            elif COVERAGE_HINT.search(question):
                label = "coverage"
            else:
                label = "escalation"           # fail safe: uncertain -> human
            return Response(stop_reason="end_turn",
                            content=[Block(type="text", text=label)],
                            usage=Usage(120, 3), model=self.model)

        results = self._tool_results(messages)
        step = len(results.get("lookup_policy", [])) + len(results.get("check_coverage", [])) \
            + len(results.get("enrollment_window", [])) + len(results.get("calculate", []))

        def tool(name: str, args: dict, thinking: str = "") -> Response:
            return Response(
                stop_reason="tool_use",
                content=[Block(type="text", text=thinking),
                         Block(type="tool_use", id=f"toolu_mock_{step:02d}",
                               name=name, input=args)],
                usage=Usage(420 + step * 180, 60),
                model=self.model,
            )

        def answer(text: str) -> Response:
            return Response(
                stop_reason="end_turn",
                content=[Block(type="text", text=text)],
                usage=Usage(500 + step * 200, max(40, len(text) // 4)),
                model=self.model,
            )

        # 0. Escalation agent — its only tool is the hand-off
        if "escalate_to_licensed_agent" in available:
            if not results.get("escalate_to_licensed_agent"):
                return tool("escalate_to_licensed_agent", {
                    "reason": "regulated_advice",
                    "summary": question[:200],
                }, "This needs a licensed human, not an automated answer.")
            outcome = results["escalate_to_licensed_agent"][0]
            if "error" in outcome:
                return answer(
                    "That's a recommendation for a specific person, which has to "
                    "come from a licensed agent rather than from me — it depends on "
                    "her doctors, her prescriptions, her budget and her health "
                    "history. I've requested a hand-off to the licensed agent desk "
                    "and flagged it for approval. In the meantime I can tell you "
                    "what each plan covers, if that helps you frame the conversation."
                )
            return answer(
                f"That's a recommendation for a specific person, which has to come "
                f"from a licensed agent. I've passed it to the "
                f"{outcome.get('queue', 'licensed-agent')} queue with a summary — "
                f"someone will pick it up within {outcome.get('sla_minutes', 30)} "
                f"minutes. I can still tell you what each plan covers if that's useful."
            )

        # 1. Policy mentioned but not yet looked up
        match = POLICY_RE.search(question)
        if match and "lookup_policy" in available and not results.get("lookup_policy"):
            return tool("lookup_policy", {"policy_id": f"P-{match.group(1)}"},
                        "The user gave a policy number, so I need the record first.")

        policy = next((r for r in results.get("lookup_policy", []) if "error" not in r), None)

        # 1b. Lookup failed — try the plausible correction once, then report
        failed = [r for r in results.get("lookup_policy", []) if "error" in r]
        if failed and not policy and "lookup_policy" in available:
            if len(failed) == 1:
                return tool("lookup_policy", {"policy_id": "P-1002"},
                            "That ID isn't on file. I'll try the nearest known policy.")
            return answer(
                f"I couldn't find policy {failed[0].get('policy_id')} in the policy "
                f"admin system, and the nearest match didn't fit either. Please "
                f"confirm the policy number and I'll look again."
            )

        # 2. Coverage question with a known plan
        benefit = next((b for rx, b in BENEFIT_KEYWORDS if rx.search(question)), None)
        if benefit and "check_coverage" in available and not results.get("check_coverage"):
            plan = policy["plan"] if policy else "Medicare Supplement Plan G"
            return tool("check_coverage", {"plan": plan, "benefit": benefit},
                        f"Checking {benefit.replace('_', ' ')} on {plan}.")

        # 3. Enrollment question
        period = next((p for rx, p in PERIOD_KEYWORDS if rx.search(question)), None)
        if period and "enrollment_window" in available and not results.get("enrollment_window"):
            return tool("enrollment_window", {"period": period},
                        f"Looking up the {period} rules.")

        # 4. Annual premium arithmetic
        if (policy and ANNUAL_RE.search(question) and "calculate" in available
                and not results.get("calculate")):
            return tool("calculate", {"expression": f"{policy['premium_monthly']} * 12"},
                        "Using the calculator rather than doing this in my head.")

        # 5. Compose the answer from what came back
        parts: list[str] = []

        # If an earlier lookup failed, say so before reporting what did work —
        # otherwise the answer silently substitutes a different member's record,
        # which is the worst possible failure mode on a support desk.
        if failed and policy:
            parts.append(
                f"I couldn't find policy {failed[0].get('policy_id')} — it isn't in "
                f"the policy admin system. The closest record on file is "
                f"{policy.get('policy_id')}; please confirm which one you meant."
            )

        if policy:
            parts.append(
                f"Policy {policy.get('policy_id', '')} — {policy['holder']} holds a "
                f"{policy['plan']} in {policy['state']}, currently "
                f"{policy['status'].upper()} (effective {policy['effective_date']}) "
                f"[{policy.get('source', 'policy-admin')}]."
            )

        for cov in results.get("check_coverage", []):
            if "error" in cov:
                parts.append(
                    f"I don't have a rule on file for "
                    f"{cov.get('benefit', 'that benefit')} under "
                    f"{cov.get('plan', 'that plan')}, so I can't confirm it from my sources."
                )
            else:
                parts.append(f"{cov['coverage']} [{cov['source']}]")

        for enr in results.get("enrollment_window", []):
            if "error" not in enr:
                parts.append(f"{enr['name']}: {enr['window']} {enr['allows']} [{enr['source']}]")

        for calc in results.get("calculate", []):
            if "error" not in calc:
                parts.append(
                    f"That works out to ${calc['result']:,.2f} per year "
                    f"({calc['expression']})."
                )

        if not parts:
            return answer(
                "I don't have that in my sources. To answer it I'd need the plan's "
                "Summary of Benefits, or a policy number I can look up. Could you "
                "give me the policy ID?"
            )

        return answer(" ".join(parts))


# ===========================================================================
def get_client(force_mock: bool = False):
    """Real client when a key is present, mock otherwise. No other code changes."""
    if force_mock or not os.getenv("ANTHROPIC_API_KEY"):
        return MockClient()
    try:
        return ClaudeClient()
    except Exception:                                        # noqa: BLE001
        return MockClient()
