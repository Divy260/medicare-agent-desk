"""
Stand-in for the systems a real deployment would call.

In production these are REST calls to a policy admin system, a benefits knowledge
base, and a quoting engine. Keeping them behind plain Python functions means the
agent code is identical either way — only `tools/` changes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Policy admin system (stand-in)
# ---------------------------------------------------------------------------
# NOTE the `member_phone` field. It exists here because a real policy record has
# it — and it is stripped in tools/registry.py before the model ever sees the
# record. Data minimisation is a property of the tool layer, not the prompt.

POLICIES: dict[str, dict] = {
    "P-1001": {
        "holder": "A. Rivera", "plan": "Medicare Supplement Plan G",
        "status": "active", "premium_monthly": 148.20, "state": "TX",
        "effective_date": "2025-01-01", "member_phone": "512-555-0142",
        "agent_of_record": "AGT-4471",
    },
    "P-1002": {
        "holder": "J. Okafor", "plan": "Medicare Advantage HMO",
        "status": "lapsed", "premium_monthly": 0.00, "state": "FL",
        "effective_date": "2024-06-01", "member_phone": "305-555-0199",
        "agent_of_record": "AGT-2210",
    },
    "P-1003": {
        "holder": "M. Chen", "plan": "Medicare Supplement Plan N",
        "status": "active", "premium_monthly": 112.75, "state": "TX",
        "effective_date": "2026-01-01", "member_phone": "512-555-0177",
        "agent_of_record": "AGT-4471",
    },
}

# ---------------------------------------------------------------------------
# Benefits knowledge base (stand-in)
# ---------------------------------------------------------------------------
# Medigap plans are standardised by federal law — Plan G from one carrier covers
# exactly what Plan G from another covers. That standardisation is what makes a
# lookup table a legitimate model of the domain rather than a simplification.

PLAN_RULES: dict[str, dict[str, str]] = {
    "Medicare Supplement Plan G": {
        "part_a_deductible": "Covered in full.",
        "part_b_deductible": (
            "NOT covered. The member pays the annual Part B deductible out of "
            "pocket each calendar year before Plan G begins paying Part B "
            "coinsurance. This is the only benefit difference between Plan G and "
            "Plan F."
        ),
        "part_b_coinsurance": "Covered in full after the Part B deductible is met.",
        "part_b_excess_charges": "Covered in full.",
        "skilled_nursing_coinsurance": "Covered.",
        "foreign_travel_emergency": (
            "Covered at 80% of billed charges after a $250 annual deductible, up "
            "to a $50,000 lifetime maximum, for care needed during the first 60 "
            "days of a trip outside the United States."
        ),
        "prescription_drugs": (
            "NOT included. A member wanting drug coverage must enrol separately "
            "in a standalone Part D plan."
        ),
        "routine_dental_vision_hearing": "NOT covered.",
    },
    "Medicare Supplement Plan N": {
        "part_a_deductible": "Covered in full.",
        "part_b_deductible": "NOT covered. The member pays it annually.",
        "part_b_coinsurance": (
            "Covered, except for copays of up to $20 for some office visits and "
            "up to $50 for emergency room visits that do not result in admission."
        ),
        "part_b_excess_charges": "NOT covered.",
        "skilled_nursing_coinsurance": "Covered.",
        "foreign_travel_emergency": (
            "Covered at 80% after a $250 annual deductible, up to a $50,000 "
            "lifetime maximum."
        ),
        "prescription_drugs": "NOT included. Requires a standalone Part D plan.",
        "routine_dental_vision_hearing": "NOT covered.",
    },
    "Medicare Advantage HMO": {
        "part_a_deductible": (
            "Not applicable. Medicare Advantage plans use their own cost-sharing "
            "schedule rather than Original Medicare's deductibles."
        ),
        "part_b_deductible": "Not applicable — see the plan's Summary of Benefits.",
        "part_b_coinsurance": "Not applicable — the plan sets its own copays.",
        "prescription_drugs": "Usually included. Confirm on the Summary of Benefits.",
        "routine_dental_vision_hearing": (
            "Commonly included as supplemental benefits, but varies by plan. "
            "Confirm on the Summary of Benefits."
        ),
    },
}

# ---------------------------------------------------------------------------
# Enrollment calendar
# ---------------------------------------------------------------------------
# Dates are structural (they recur annually and are set by CMS). Dollar amounts
# are deliberately absent everywhere in this file — they change every calendar
# year, and a confidently quoted stale figure is a compliance incident.

ENROLLMENT_PERIODS: dict[str, dict[str, str]] = {
    "IEP": {
        "name": "Initial Enrollment Period",
        "window": "The 7 months around the 65th birthday: 3 months before the "
                  "birthday month, the birthday month, and 3 months after.",
        "allows": "First enrollment in Parts A and B without a late enrollment penalty.",
    },
    "AEP": {
        "name": "Annual Enrollment Period",
        "window": "October 15 through December 7 each year.",
        "allows": "Join, switch, or drop a Medicare Advantage plan or a Part D "
                  "prescription drug plan. Changes take effect January 1.",
    },
    "MA_OEP": {
        "name": "Medicare Advantage Open Enrollment Period",
        "window": "January 1 through March 31 each year.",
        "allows": "A member already enrolled in a Medicare Advantage plan may "
                  "switch to a different MA plan, or return to Original Medicare "
                  "and add a Part D plan. Only one change is permitted.",
    },
    "MEDIGAP_OE": {
        "name": "Medigap Open Enrollment",
        "window": "The 6 months beginning the month the beneficiary is both 65 or "
                  "older and enrolled in Part B.",
        "allows": "Guaranteed issue — a carrier cannot deny a Medigap application "
                  "or charge more because of health history. Outside this window "
                  "most states permit medical underwriting.",
    },
    "SEP": {
        "name": "Special Enrollment Period",
        "window": "Triggered by a qualifying life event; length varies by event.",
        "allows": "Enrollment changes after moving out of a plan's service area, "
                  "losing employer coverage, or a plan leaving the market.",
    },
}
