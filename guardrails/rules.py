"""
Guardrails: deterministic checks around the model.

Prompting is guidance. A guardrail is enforcement. A system prompt is a request;
a code path is a guarantee. You need both, and you never rely on the prompt for
anything a compliance officer would need proof of.

Three layers:
    INPUT   before the model sees it   — reject, or route elsewhere
    TOOL    before a tool executes     — authorise, validate, cap
    OUTPUT  before the user sees it    — redact, block, or flag

Everything here is regex or a comparison: microseconds, no hallucination, unit
testable. An LLM-based classifier goes on top of this, never instead of it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Verdict:
    allowed: bool
    rule: str = ""
    reason: str = ""
    rewritten: str | None = None
    severity: str = "block"          # block | redact | flag

    def __str__(self) -> str:
        if self.allowed and not self.rule:
            return "ALLOW"
        return f"{'ALLOW' if self.allowed else 'BLOCK'} [{self.rule}] {self.reason}"


# ===========================================================================
# LAYER 1 — input
# ===========================================================================

MAX_INPUT_CHARS = 6_000

# Attempts to override the system prompt. Matters most for text you did NOT
# write — uploaded documents, retrieved chunks, email bodies. Treat all of it as
# untrusted input.
INJECTION = re.compile(
    r"ignore (all )?(previous|prior|above) (instructions|prompts)"
    r"|disregard your (system prompt|instructions|rules)"
    r"|you are now (in )?(developer|debug|dan|god) mode"
    r"|(reveal|print|repeat) (your|the) (system prompt|instructions|rules)",
    re.I,
)

# Individualised advice a licensed distributor's tool must not give. This is a
# regulatory boundary, not a tone preference — so it is enforced in code and
# routed to a human, not merely discouraged in the prompt.
_SUBJECT = r"(i|we|he|she|they|my client|my mother|my father|this member|the member)"

REGULATED_ADVICE = re.compile(
    # "should my client drop…", "should she switch…", "should I buy…"
    rf"\bshould {_SUBJECT} (enroll|enrol|switch|drop|buy|cancel|choose|keep|stay|move|take)\b"
    rf"|\bwhich plan (should|would be best for) {_SUBJECT}\b"
    rf"|\bwhat('s| is) the best plan for {_SUBJECT}\b"
    rf"|\bwould {_SUBJECT} be better off\b"
    rf"|\bdo you recommend\b"
    rf"|\bis (this|it) covered for (my|her|his|their) "
    rf"(cancer|diabetes|heart|condition|surgery|treatment|medication)\b"
    rf"|\bwill medicare pay for (my|her|his|their)\b"
    rf"|\bwhat should {_SUBJECT} do about (my|her|his|their)? ?(health|coverage|treatment)\b",
    re.I,
)

# Not an insurance support question.
OUT_OF_SCOPE = re.compile(
    r"\b(write|generate)\s+(me\s+)?(a\s+)?(poem|song|essay|story|python|javascript|code|script)\b"
    r"|\bwho should i vote for\b"
    r"|\byour opinion on (abortion|immigration|politics|religion)\b",
    re.I,
)


def check_input(text: str) -> Verdict:
    if len(text) > MAX_INPUT_CHARS:
        return Verdict(False, "input_too_long",
                       f"{len(text)} characters exceeds the {MAX_INPUT_CHARS} limit")
    if INJECTION.search(text):
        return Verdict(False, "prompt_injection",
                       "input attempts to override system instructions")
    if REGULATED_ADVICE.search(text):
        return Verdict(False, "regulated_advice",
                       "asks for individualised enrollment or coverage advice; "
                       "must route to a licensed agent")
    if OUT_OF_SCOPE.search(text):
        return Verdict(False, "out_of_scope",
                       "not an insurance support question")
    return Verdict(True)


# ===========================================================================
# LAYER 2 — tool calls
# ===========================================================================

ARG_VALIDATORS: dict[str, callable] = {
    "lookup_policy": lambda a: (
        None if re.fullmatch(r"P-\d{4}", str(a.get("policy_id", "")).upper())
        else f"policy_id {a.get('policy_id')!r} is not in the form P-nnnn"
    ),
    "calculate": lambda a: (
        None if re.fullmatch(r"[0-9\s+\-*/().]+", str(a.get("expression", "")))
        else "expression contains characters outside the allow-list"
    ),
}


def check_tool_call(name: str, args: dict, call_counts: dict[str, int],
                    tools_by_name: dict, human_approved: bool = False) -> Verdict:
    tool = tools_by_name.get(name)
    if tool is None:
        return Verdict(False, "unknown_tool", f"{name} is not in the allow-list")

    if call_counts.get(name, 0) >= tool.max_calls_per_run:
        return Verdict(False, "rate_limit",
                       f"{name} already called {tool.max_calls_per_run} times this run")

    validator = ARG_VALIDATORS.get(name)
    if validator:
        problem = validator(args)
        if problem:
            return Verdict(False, "invalid_arguments", problem)

    if tool.requires_approval and not human_approved:
        return Verdict(False, "needs_approval",
                       f"{name} has downstream effects; a human must approve")

    return Verdict(True)


# ===========================================================================
# LAYER 3 — output
# ===========================================================================

PII_OUT: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"(?<!\d)(?:\(\d{3}\)\s?|\d{3}[-.])\d{3}[-.]?\d{4}(?!\d)"), "[PHONE REDACTED]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"), "[EMAIL REDACTED]"),
]

# Language CMS marketing rules prohibit. These are compliance incidents, not
# style problems, so they are blocked outright rather than rewritten.
FORBIDDEN_CLAIMS = re.compile(
    r"\bguaranteed (approval|acceptance|savings)\b"
    r"|\byou will (definitely|certainly) (save|qualify)\b"
    r"|\bthis is the best plan for you\b"
    r"|\b100% covered\b"
    r"|\bfree medicare\b"
    r"|\bno cost to you ever\b",
    re.I,
)

CLINICAL_LANGUAGE = re.compile(r"\b(diagnos|treatment|prescrib|symptom)\w*\b", re.I)

DISCLAIMER = ("\n\nThis is general plan information, not medical, legal or tax "
              "advice. For a recommendation, please speak with a licensed agent.")


def check_output(text: str, tools_called: list[str]) -> Verdict:
    if FORBIDDEN_CLAIMS.search(text):
        return Verdict(False, "forbidden_claim",
                       "contains a marketing claim that violates CMS guidance")

    redacted, hits = text, 0
    for pattern, replacement in PII_OUT:
        redacted, n = pattern.subn(replacement, redacted)
        hits += n
    if hits:
        return Verdict(True, "pii_redacted", f"{hits} identifier(s) removed",
                       rewritten=redacted, severity="redact")

    if CLINICAL_LANGUAGE.search(text) and "licensed" not in text.lower():
        return Verdict(True, "missing_disclaimer",
                       "clinical language without a referral disclaimer",
                       rewritten=text + DISCLAIMER, severity="flag")

    # A long, confident answer produced without calling a single tool is the
    # shape of a hallucination. Flag it for review rather than blocking — some
    # legitimate answers are conversational.
    if not tools_called and len(text) > 400:
        return Verdict(True, "unsourced_long_answer",
                       "long answer produced with no tool calls", severity="flag")

    return Verdict(True)
