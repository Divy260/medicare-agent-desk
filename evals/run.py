"""
The eval runner and the regression gate.

An LLM feature has no compiler and no type system. Nothing tells you that a
prompt edit broke three behaviours you were not thinking about. This suite is
the substitute, and it only works if it runs in CI and BLOCKS THE MERGE.

Run:
    python -m evals.run                          # run and print a report
    python -m evals.run --save baseline.json     # record a baseline
    python -m evals.run --gate baseline.json     # exit 1 on any regression

Scorers, cheapest and most reliable first — an LLM judge goes on top of these,
never instead of them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field

from evals.golden_set import GOLDEN_SET, Case
from orchestrator import handle

REFUSAL_MARKERS = [
    "i don't have that in my sources",
    "i do not have that",
    "not in my sources",
    "couldn't find",
    "can't help with that",
]


# ===========================================================================
# Scorers
# ===========================================================================

def score_contains(answer: str, case: Case) -> dict:
    """Deterministic substring checks. Free, exact, no false confidence."""
    low = answer.lower()
    missing = [s for s in case.must_contain if s.lower() not in low]
    present = [s for s in case.must_not_contain if s.lower() in low]
    return {"scorer": "contains", "passed": not missing and not present,
            "missing": missing, "forbidden_present": present}


def score_refusal(answer: str, case: Case) -> dict:
    """Refused when it should have — and, just as important, did NOT otherwise."""
    refused = any(m in answer.lower() for m in REFUSAL_MARKERS)
    return {"scorer": "refusal",
            "passed": (refused == case.expected_refusal) if case.expected_refusal else True,
            "refused": refused, "expected": case.expected_refusal}


def score_routing(result, case: Case) -> dict:
    if case.expected_agent is None:
        return {"scorer": "routing", "passed": True, "skipped": True}
    return {"scorer": "routing", "passed": result.agent == case.expected_agent,
            "routed_to": result.agent, "expected": case.expected_agent}


def score_trajectory(result, case: Case) -> dict:
    """
    Did the agent take the right PATH, not just reach the right answer?

    An agent that produced the correct premium without calling lookup_policy did
    not succeed — it guessed from parametric memory and happened to be right.
    This is the scorer that catches that, and it is the one most people skip.
    """
    if not case.expected_tools:
        return {"scorer": "trajectory", "passed": True, "skipped": True}
    called = result.trace.tools_called()
    missing = [t for t in case.expected_tools if t not in called]
    return {"scorer": "trajectory", "passed": not missing,
            "called": called, "missing": missing}


def score_guardrail(result, case: Case) -> dict:
    if case.expected_guardrail is None:
        return {"scorer": "guardrail", "passed": True, "skipped": True}
    fired = result.trace.guardrails_triggered()
    return {"scorer": "guardrail", "passed": case.expected_guardrail in fired,
            "fired": fired, "expected": case.expected_guardrail}


SCORERS = [
    lambda r, c: score_contains(r.answer, c),
    lambda r, c: score_refusal(r.answer, c),
    score_routing,
    score_trajectory,
    score_guardrail,
]


# ===========================================================================
# Runner
# ===========================================================================

@dataclass
class Result:
    case_id: str
    category: str
    weight: float
    passed: bool
    scores: list[dict]
    latency_ms: float
    cost_usd: float
    answer: str
    agent: str


def run_suite(cases: list[Case] | None = None) -> list[Result]:
    cases = cases or GOLDEN_SET
    out: list[Result] = []
    for case in cases:
        t0 = time.perf_counter()
        result = handle(case.question)
        latency = (time.perf_counter() - t0) * 1000
        scores = [s(result, case) for s in SCORERS]
        out.append(Result(
            case_id=case.id, category=case.category, weight=case.weight,
            passed=all(s["passed"] for s in scores), scores=scores,
            latency_ms=latency, cost_usd=result.trace.cost_usd,
            answer=result.answer, agent=result.agent,
        ))
    return out


def summarise(results: list[Result]) -> dict:
    total_w = sum(r.weight for r in results)
    passed_w = sum(r.weight for r in results if r.passed)
    by_cat: dict[str, list[Result]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    return {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 4),
        "weighted_pass_rate": round(passed_w / total_w, 4),
        "by_category": {c: round(sum(1 for r in rs if r.passed) / len(rs), 3)
                        for c, rs in by_cat.items()},
        "p50_latency_ms": round(statistics.median(r.latency_ms for r in results), 1),
        "p95_latency_ms": round(sorted(r.latency_ms for r in results)[
            max(0, int(len(results) * 0.95) - 1)], 1),
        "total_cost_usd": round(sum(r.cost_usd for r in results), 5),
        "failures": [r.case_id for r in results if not r.passed],
    }


# ===========================================================================
# The gate
# ===========================================================================

PROTECTED = {"safety", "refusal"}


def compare(baseline: dict, candidate: dict) -> dict:
    """
    Three rules, and the third is the one that matters most:

      1. Fail on any NEW failing case, not merely on an aggregate drop.
      2. Fail on a drop in weighted pass rate.
      3. Fail on a regression in a PROTECTED category regardless of the overall
         number — an overall improvement must never be allowed to hide a safety
         regression.
    """
    delta = candidate["weighted_pass_rate"] - baseline["weighted_pass_rate"]
    new_failures = sorted(set(candidate["failures"]) - set(baseline["failures"]))
    fixed = sorted(set(baseline["failures"]) - set(candidate["failures"]))

    protected_regressions = [
        cat for cat in PROTECTED
        if candidate["by_category"].get(cat, 1.0) < baseline["by_category"].get(cat, 0.0)
    ]

    gate = "PASS"
    reasons = []
    if new_failures:
        gate, _ = "FAIL", reasons.append(f"new failing cases: {new_failures}")
    if delta < 0:
        gate, _ = "FAIL", reasons.append(f"weighted pass rate fell by {abs(delta):.4f}")
    if protected_regressions:
        gate, _ = "FAIL", reasons.append(
            f"regression in protected categories: {protected_regressions}")

    return {"delta_weighted_pass_rate": round(delta, 4),
            "new_failures": new_failures, "newly_fixed": fixed,
            "protected_regressions": protected_regressions,
            "gate": gate, "reasons": reasons}


# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="PATH", help="write this run's summary as a baseline")
    ap.add_argument("--gate", metavar="PATH", help="compare against a baseline and exit 1 on regression")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    results = run_suite()
    summary = summarise(results)

    print("=" * 78)
    print("MEDICARE AGENT DESK — EVAL SUITE")
    print("=" * 78)
    for r in results:
        mark = "PASS" if r.passed else "FAIL"
        print(f"  [{mark}] {r.case_id:<8} {r.category:<10} w={r.weight:<4} "
              f"-> {r.agent:<11} {r.latency_ms:>7.0f}ms")
        if not r.passed or args.verbose:
            for s in r.scores:
                if not s["passed"]:
                    detail = {k: v for k, v in s.items()
                              if k not in ("scorer", "passed") and v}
                    print(f"          failed {s['scorer']}: {json.dumps(detail)}")
            if not r.passed:
                print(f"          answer: {r.answer[:150]}")

    print("\n" + json.dumps(summary, indent=2))

    if args.save:
        with open(args.save, "w") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\nbaseline written to {args.save}")

    if args.gate:
        try:
            with open(args.gate) as fh:
                baseline = json.load(fh)
        except FileNotFoundError:
            print(f"\nno baseline at {args.gate} — treating this run as the first")
            return 0
        verdict = compare(baseline, summary)
        print("\n" + "=" * 78)
        print("REGRESSION GATE")
        print("=" * 78)
        print(json.dumps(verdict, indent=2))
        return 0 if verdict["gate"] == "PASS" else 1

    return 0 if not summary["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
