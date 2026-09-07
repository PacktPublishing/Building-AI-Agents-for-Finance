"""
models.py
=========
Pydantic data models for insurance claims, policies, and decisions.

These schemas enforce contracts between agents — each agent knows
exactly what it receives and what it must produce.
"""

from datetime import date
from enum import Enum
from pydantic import BaseModel, Field


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


class ClaimRecord(BaseModel):
    """Core claim data model — shared state for the pipeline."""
    claim_id: str = ""
    policyholder_name: str = ""
    policy_number: str = ""
    claim_type: ClaimType = ClaimType.AUTO
    date_of_loss: date = Field(default_factory=date.today)
    description: str = ""
    status: ClaimStatus = ClaimStatus.RECEIVED

    # Populated by Document Parser
    parsed_documents: list[dict] = []
    damage_description: str = ""

    # Populated by Coverage Agent
    coverage_verified: bool | None = None
    applicable_limit: float | None = None
    deductible: float | None = None
    exclusions_found: list[str] = []
    coverage_reasoning: str = ""
    coverage_reason_codes: list[str] = Field(default_factory=list)
    coverage_evidence_complete: bool = False

    # Populated by Fraud Screener
    fraud_score: int | None = None
    red_flags: list[str] = []
    fraud_recommendation: str = ""

    # Populated by Assessment Agent
    estimated_payout: float | None = None
    assessment_breakdown: str = ""

    # Populated by Decision Agent
    decision: str = ""
    decision_reasoning: str = ""
    decision_reason_codes: list[str] = Field(default_factory=list)
    compliance_issues: list[str] = Field(default_factory=list)

    # Audit trail
    audit_log: list[dict] = []


class PolicyRecord(BaseModel):
    """Insurance policy data."""
    policy_number: str
    policyholder_name: str
    inception_date: date
    expiration_date: date
    claim_type: ClaimType
    coverage_limit: float
    deductible: float
    coverage_sections: str = ""
    exclusions: list[str] = []


class CoverageResult(BaseModel):
    """Output of the Coverage Verification Agent."""
    is_covered: bool
    applicable_section: str = ""
    coverage_limit: float = 0.0
    deductible: float = 0.0
    exclusions_found: list[str] = []
    reasoning: str = ""


class FraudScore(BaseModel):
    """Output of the Fraud Screening Agent."""
    score: int = Field(ge=0, le=100)
    red_flags: list[str] = []
    recommendation: str = "proceed"


class FraudArgument(BaseModel):
    """Structured argument for fraud investigation debate."""
    position: str  # "fraud" or "legitimate"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = []
    reasoning: str = ""
    weaknesses: list[str] = []


class ClaimDecision(BaseModel):
    """Final claim decision."""
    decision: str  # "APPROVED", "DENIED", "ESCALATED"
    payout_amount: float = 0.0
    reasoning: str = ""
    regulatory_rules_applied: list[str] = []
