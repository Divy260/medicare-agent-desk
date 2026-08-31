"""
Prompt hygiene checks, run in CI.

Prompts are code. They have no compiler, no type checker and no linter, so the
few properties that CAN be checked mechanically are worth checking on every
commit — because the alternative is finding out from the eval suite, or from a
user.

Three rules, each with a specific failure behind it:

  1. EVERY PROMPT IS VERSIONED. `PROMPT_VERSION` must be set and every agent
     must carry it. Without a version, a behaviour change cannot be attributed
     to a prompt edit, and an eval result cannot be tied to what produced it.

  2. TOOL DESCRIPTIONS ARE SUBSTANTIAL. The description is the only thing
     telling the model when to reach for a tool. Vague descriptions cause more
     wrong-tool selection than any other single factor, and "Looks up a policy"
     is not a specification. 80 characters is a crude floor, but a crude floor
     that fails loudly beats a convention nobody enforces.

  3. NO HARD-CODED DOLLAR AMOUNTS IN THE ENROLLMENT CALENDAR. Deductibles,
     premiums and IRMAA brackets change every calendar year. The knowledge base
     models structure, not figures, so that a stale number is not merely
     unlikely but unrepresentable. A confidently quoted stale figure is a
     compliance incident, not a typo.

Run:  python -m tools.prompt_hygiene
"""

from __future__ import annotations

import re
import sys

from agents.roster import AGENTS, PROMPT_VERSION
from data.knowledge import ENROLLMENT_PERIODS
from tools.registry import TOOLS

MIN_DESCRIPTION_CHARS = 80
VERSION_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.[a-z]$")
DOLLARS_RE = re.compile(r"\$\s?\d")


def check() -> list[str]:
    problems: list[str] = []

    # ---- 1. versioning ---------------------------------------------------
    if not VERSION_RE.match(PROMPT_VERSION):
        problems.append(
            f"PROMPT_VERSION {PROMPT_VERSION!r} is not in the form YYYY-MM-DD.a")

    for name, agent in AGENTS.items():
        if agent.prompt_version != PROMPT_VERSION:
            problems.append(
                f"agent {name!r} is pinned to prompt_version "
                f"{agent.prompt_version!r}, not {PROMPT_VERSION!r}")
        if not agent.system.strip():
            problems.append(f"agent {name!r} has an empty system prompt")
        if not agent.tool_names:
            problems.append(f"agent {name!r} has no tools")

    # ---- 2. tool descriptions -------------------------------------------
    for tool in TOOLS:
        if len(tool.description) < MIN_DESCRIPTION_CHARS:
            problems.append(
                f"tool {tool.name!r} description is {len(tool.description)} chars; "
                f"minimum is {MIN_DESCRIPTION_CHARS}")
        for prop, spec in tool.input_schema.get("properties", {}).items():
            if "description" not in spec and "enum" not in spec:
                problems.append(
                    f"tool {tool.name!r} argument {prop!r} has neither a "
                    f"description nor an enum — the model is guessing")

    # ---- 3. no dollar figures in the enrollment calendar -----------------
    for code, entry in ENROLLMENT_PERIODS.items():
        for field, value in entry.items():
            if DOLLARS_RE.search(str(value)):
                problems.append(
                    f"enrollment period {code}.{field} contains a dollar amount; "
                    f"these change annually and must not be hard-coded")

    return problems


def main() -> int:
    problems = check()
    print("=" * 78)
    print("PROMPT HYGIENE")
    print("=" * 78)
    print(f"  prompt version : {PROMPT_VERSION}")
    print(f"  agents         : {len(AGENTS)}")
    print(f"  tools          : {len(TOOLS)}")
    print(f"  min description: {MIN_DESCRIPTION_CHARS} chars")

    if not problems:
        print("\nOK — all checks passed.")
        return 0

    print(f"\nFAILED — {len(problems)} problem(s):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
