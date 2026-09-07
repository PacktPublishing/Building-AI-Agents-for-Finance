"""
common.py — Shared models, helpers, and sample data for Chapter 9 labs.

The three labs share the same data contracts (ClaimRecord, PolicyRecord,
FraudArgument, etc.) and the same sample claim used in the chapter text,
so each notebook can focus on the *architecture* it demonstrates.
"""
from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ClaimType(str, Enum):
    AUTO = "auto"
    PROPERTY = "property"
    HEALTH = "health"
    LIABILITY = "liability"


class ClaimStatus(str, Enum):
    RECEIVED = "received"
    PARSING = "parsing"
    COVERAGE_CHECK = "coverage_check"
    FRAUD_SCREENING = "fraud_screening"
    ASSESSMENT = "assessment"
    DECISION = "decision"
    APPROVED = "approved"
    DENIED = "denied"
    ESCALATED = "escalated"
    COMPLIANCE_REVIEW = "compliance_review"


class FraudDetermination(str, Enum):
    STRONG_FRAUD_INDICATORS = "strong_fraud_indicators"
    LIKELY_FRAUD = "likely_fraud"
    INCONCLUSIVE = "inconclusive"
    LIKELY_LEGITIMATE = "likely_legitimate"


# ---------------------------------------------------------------------------
# Pydantic models — typed contracts between agents
# ---------------------------------------------------------------------------

class ClaimRecord(BaseModel):
    """Core claim data — the shared state object for the claims pipeline."""
    claim_id: str = ""
    policyholder_name: str = ""
    policy_number: str = ""
    claim_type: ClaimType = ClaimType.AUTO
    date_of_loss: date = Field(default_factory=date.today)
    description: str = ""
    status: ClaimStatus = ClaimStatus.RECEIVED

    parsed_documents: list[dict] = []
    damage_description: str = ""

    coverage_verified: bool | None = None
    applicable_limit: float | None = None
    deductible: float | None = None
    exclusions_found: list[str] = []
    coverage_reasoning: str = ""
    coverage_reason_codes: list[str] = Field(default_factory=list)
    coverage_evidence_complete: bool = False

    fraud_score: int | None = None
    red_flags: list[str] = []
    fraud_recommendation: str = ""

    estimated_payout: float | None = None
    assessment_breakdown: str = ""

    decision: str = ""
    decision_reasoning: str = ""
    decision_reason_codes: list[str] = Field(default_factory=list)
    compliance_issues: list[str] = Field(default_factory=list)

    audit_log: list[dict] = []


class PolicyRecord(BaseModel):
    policy_number: str
    policyholder_name: str
    inception_date: date
    expiration_date: date
    claim_type: ClaimType
    coverage_limit: float
    deductible: float
    coverage_sections: str = ""
    exclusions: list[str] = []


class FraudArgument(BaseModel):
    """Structured argument for the fraud-investigation debate."""
    position: str  # "fraud" or "legitimate"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = []
    reasoning: str = ""
    weaknesses: list[str] = []


class ClaimDecision(BaseModel):
    decision: str  # "APPROVED" / "DENIED" / "ESCALATED"
    payout_amount: float = 0.0
    reasoning: str = ""
    regulatory_rules_applied: list[str] = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_claim_id() -> str:
    short = uuid.uuid4().hex[:8].upper()
    return f"CLM-{date.today().year}-{short}"


def log_audit(state: dict, agent_name: str, action: str, detail: str = "") -> None:
    if "claim" not in state:
        return
    claim = state["claim"]
    claim.setdefault("audit_log", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent_name,
        "action": action,
        "detail": detail[:500],
    })


def days_between(d1: str, d2: str) -> int:
    """Signed days from d1 to d2 (positive when d2 is later).

    Accepts ISO dates or datetimes; only the date part is compared.
    """
    fmt = "%Y-%m-%d"
    try:
        return (datetime.strptime(d2[:10], fmt)
                - datetime.strptime(d1[:10], fmt)).days
    except ValueError as exc:
        raise ValueError(
            f"Expected ISO dates (YYYY-MM-DD), got {d1!r}, {d2!r}"
        ) from exc


# ---------------------------------------------------------------------------
# Sample data — used by Lab 1 (claims pipeline) and Lab 2 (fraud debate)
# ---------------------------------------------------------------------------

SAMPLE_CLAIM = {
    "name": "Sarah Chen",
    "policy_number": "POL-2024-AUTO-78432",
    "date_of_loss": "2026-04-08",
    "description": (
        "Rear-ended a stopped delivery truck on I-95 in "
        "heavy rain at approximately 2:47 AM. Front bumper "
        "cracked, headlight broken. Airbags did not deploy. "
        "Experiencing back pain."
    ),
}

SAMPLE_POLICY = {
    "policy_number": "POL-2024-AUTO-78432",
    "policyholder_name": "Sarah Chen",
    "inception_date": "2025-01-15",
    "expiration_date": "2026-07-15",
    "claim_type": "auto",
    "coverage_limit": 50000,
    "deductible": 500,
    "coverage_sections": "collision coverage",
    "exclusions": ["racing", "intentional damage"],
}

SAMPLE_DOCUMENTS = [
    {
        "type": "claim_form",
        "fields": {
            "claimant": "Sarah Chen",
            "date": "2026-04-08",
            "location": "I-95 Northbound, mile marker 42",
            "description": "Rear-end collision with delivery truck",
        },
    },
    {
        "type": "photo",
        "description": (
            "Front bumper cracked with visible impact marks. "
            "Right headlight assembly broken. Paint transfer "
            "from the delivery truck's rear step visible."
        ),
    },
    {
        "type": "photo",
        "description": (
            "Close-up of bumper damage showing crack extending "
            "approximately 18 inches. Bumper partially detached "
            "on right side."
        ),
    },
    {
        "type": "police_report",
        "summary": (
            "Report #2026-04-08-5521. Two-vehicle rear-end "
            "collision. Driver (Chen) struck stopped delivery "
            "vehicle. Weather: rain. Road conditions: wet. "
            "No citations issued. No injuries reported at scene."
        ),
    },
]


# A suspicious-looking claim — used by Lab 2 to drive the fraud debate.
SUSPICIOUS_CLAIM_SUMMARY = (
    "Claimant: J. Doe. Policy POL-2026-AUTO-99887 inception 2026-04-15. "
    "Date of loss: 2026-04-20 (5 days after inception). Description: "
    "Single-vehicle accident, total loss claimed. Vehicle reportedly stolen "
    "and burned in a remote area at night. No witnesses, no security footage. "
    "Prior claims: 3 in the past 14 months. Estimated payout would be 94% of "
    "the policy limit. Police report indicates no signs of forced entry, "
    "ignition intact, fire originated inside the passenger compartment."
)
