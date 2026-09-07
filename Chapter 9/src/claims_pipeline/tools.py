"""
tools.py
========
Tool functions for the claims processing pipeline agents.

Each agent has a dedicated tool that reads from and writes to the
shared Context object — the shared-state communication pattern
from Chapter 7, Section 4.1.
"""

from llama_index.core.workflow import Context

from ..common import generate_claim_id, log_audit, days_between
from ..models import ClaimType


def classify_claim_type(description: str) -> ClaimType:
    """Classify claim type from description (simplified)."""
    desc = description.lower()
    if any(w in desc for w in ["vehicle", "car", "truck", "collision",
                                "accident", "bumper", "fender"]):
        return ClaimType.AUTO
    elif any(w in desc for w in ["house", "roof", "flood", "fire",
                                  "property", "building"]):
        return ClaimType.PROPERTY
    elif any(w in desc for w in ["hospital", "medical", "surgery",
                                  "doctor", "injury"]):
        return ClaimType.HEALTH
    return ClaimType.LIABILITY


async def process_fnol(ctx: Context, claim_data: dict) -> str:
    """Process First Notice of Loss and create claim record.

    Used by: FNOLAgent (root agent).
    Writes: claim record to shared state.
    """
    claim_id = generate_claim_id()
    claim_type = classify_claim_type(claim_data.get("description", ""))

    claim = {
        "claim_id": claim_id,
        "policyholder_name": claim_data.get("name", ""),
        "policy_number": claim_data.get("policy_number", ""),
        "claim_type": claim_type.value,
        "date_of_loss": claim_data.get("date_of_loss", ""),
        "description": claim_data.get("description", ""),
        "status": "received",
        "parsed_documents": [],
        "audit_log": [],
    }

    state = await ctx.store.get("state")
    state["claim"] = claim
    log_audit(state, "FNOLAgent", "Claim created",
              f"ID: {claim_id}, Type: {claim_type.value}")
    await ctx.store.set("state", state)
    return f"FNOL processed. Claim {claim_id}. Type: {claim_type.value}."


async def parse_documents(ctx: Context) -> str:
    """Parse submitted documents and extract structured data.

    Used by: DocumentParser.
    Reads: submitted_documents from state.
    Writes: parsed_documents, damage_description to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    documents = state.get("submitted_documents", [])

    parsed = []
    damage_desc_parts = []

    for doc in documents:
        doc_type = doc.get("type", "unknown")
        if doc_type == "photo":
            # In production: call vision model
            desc = doc.get("description",
                           "Vehicle damage visible in photo.")
            parsed.append({
                "type": "damage_photo",
                "damage_description": desc,
            })
            damage_desc_parts.append(desc)
        elif doc_type == "claim_form":
            parsed.append({
                "type": "claim_form",
                "fields": doc.get("fields", {}),
            })
        elif doc_type == "police_report":
            parsed.append({
                "type": "police_report",
                "summary": doc.get("summary", ""),
            })

    claim["parsed_documents"] = parsed
    claim["damage_description"] = " ".join(damage_desc_parts)
    claim["status"] = "parsing"
    log_audit(state, "DocumentParser",
              f"Parsed {len(parsed)} documents",
              f"Types: {[d['type'] for d in parsed]}")
    await ctx.store.set("state", state)
    return f"Parsed {len(parsed)} documents."


def coverage_evidence(claim: dict, policy: dict) -> tuple[list[str], bool]:
    """Return supported denial codes and whether basic lab evidence is complete.

    Exclusion matching is a simplified lab rule, not a legal coverage assessment.
    """
    from datetime import date

    try:
        loss = date.fromisoformat(claim["date_of_loss"])
        start = date.fromisoformat(policy["inception_date"])
        end = date.fromisoformat(policy["expiration_date"])
    except (KeyError, TypeError, ValueError):
        return [], False
    if start > end:
        return [], False
    codes = []
    if not start <= loss <= end:
        codes.append("OUTSIDE_POLICY_PERIOD")
    exclusions = policy.get("exclusions")
    if not isinstance(exclusions, list) or not all(isinstance(x, str) and x.strip() for x in exclusions):
        return codes, False
    description = claim.get("description", "")
    if not isinstance(description, str) or not description.strip():
        return codes, False
    if any(x.lower() in description.lower() for x in exclusions):
        codes.append("POLICY_EXCLUSION")
    return codes, True


async def verify_coverage(ctx: Context) -> str:
    """Check policy coverage for this claim.

    Used by: CoverageAgent.
    Reads: claim, policy from state.
    Writes: coverage fields to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    policy = state.get("policy", {})

    # Check basic coverage
    reason_codes, evidence_complete = coverage_evidence(claim, policy)
    is_covered = evidence_complete and not reason_codes
    exclusions = []
    reasoning_parts = []
    if not evidence_complete:
        reasoning_parts.append("Coverage evidence is incomplete or invalid; human review required.")

    # Check policy period
    dol = claim.get("date_of_loss", "")
    inception = policy.get("inception_date", "")
    expiration = policy.get("expiration_date", "")

    if "OUTSIDE_POLICY_PERIOD" in reason_codes:
        reasoning_parts.append(
            f"Date of loss {dol} outside policy period "
            f"({inception} to {expiration})."
        )

    # Check exclusions
    policy_exclusions = policy.get("exclusions", [])
    description = claim.get("description", "")
    for exclusion in (policy_exclusions if isinstance(policy_exclusions, list) else []):
        if not isinstance(exclusion, str) or not exclusion.strip():
            continue
        if isinstance(description, str) and exclusion.lower() in description.lower():
            exclusions.append(exclusion)

    if exclusions:
        reasoning_parts.append(
            f"Exclusions found: {exclusions}"
        )

    if is_covered and not exclusions:
        reasoning_parts.append(
            f"Event covered under {policy.get('coverage_sections', 'comprehensive')} "
            f"coverage. Limit: ${policy.get('coverage_limit', 0):,.0f}. "
            f"Deductible: ${policy.get('deductible', 0):,.0f}."
        )

    claim["coverage_verified"] = is_covered and not exclusions
    claim["applicable_limit"] = policy.get("coverage_limit", 0)
    claim["deductible"] = policy.get("deductible", 0)
    claim["exclusions_found"] = exclusions
    claim["coverage_reasoning"] = " ".join(reasoning_parts)
    claim["coverage_reason_codes"] = reason_codes
    claim["coverage_evidence_complete"] = evidence_complete
    claim["status"] = "coverage_check"

    log_audit(state, "CoverageAgent",
              f"Coverage: {'verified' if claim['coverage_verified'] else 'denied'}",
              claim["coverage_reasoning"])
    await ctx.store.set("state", state)
    return f"Coverage {'verified' if claim['coverage_verified'] else 'not verified'}."


async def screen_for_fraud(ctx: Context) -> str:
    """Screen claim against fraud indicators.

    Used by: FraudScreener.
    Reads: claim, policy, claims_history from state.
    Writes: fraud_score, red_flags, fraud_recommendation to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    policy = state.get("policy", {})

    red_flags = []
    score = 0

    # Red flag 1: Claim within 30 days of policy inception
    dol = claim.get("date_of_loss", "")
    inception = policy.get("inception_date", "")
    if dol and inception:
        days = days_between(inception, dol)
        if days < 0:
            red_flags.append(
                f"Date of loss predates policy inception by {-days} days"
            )
            score += 25
        elif days < 30:
            red_flags.append(
                f"Claim filed {days} days after policy inception"
            )
            score += 25

    # Red flag 2: Multiple prior claims
    prior_claims = state.get("claims_history", [])
    if len(prior_claims) >= 3:
        red_flags.append(
            f"{len(prior_claims)} prior claims in 12 months"
        )
        score += 20

    # Red flag 3: Late reporting
    # (Amount-vs-limit checks belong AFTER assessment; screening
    # runs before the payout exists, so it is not checked here.)
    if dol:
        report_delay = days_between(
            dol, state.get("fnol_date", dol)
        )
        if report_delay < 0:
            red_flags.append(
                f"Claim reported {-report_delay} days before the "
                "recorded date of loss"
            )
            score += 15
        elif report_delay > 30:
            red_flags.append(
                f"Claim reported {report_delay} days after loss"
            )
            score += 15

    score = min(score, 100)
    recommendation = "investigate" if score > 40 else "proceed"

    claim["fraud_score"] = score
    claim["red_flags"] = red_flags
    claim["fraud_recommendation"] = recommendation
    claim["status"] = "fraud_screening"

    log_audit(state, "FraudScreener",
              f"Score: {score}/100. Recommendation: {recommendation}",
              f"Red flags: {red_flags}")
    await ctx.store.set("state", state)
    return f"Fraud screening complete. Score: {score}/100."


async def assess_damage(ctx: Context) -> str:
    """Estimate claim value from parsed damage data.

    Used by: AssessmentAgent.
    Reads: damage_description, deductible, applicable_limit from claim.
    Writes: estimated_payout, assessment_breakdown to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]

    # In production: use LLM with damage photos and repair estimates
    # For the lab: use a simplified estimation
    damage_desc = claim.get("damage_description", "")
    deductible = claim.get("deductible", 500)
    limit = claim.get("applicable_limit", 50000)

    # Simplified damage estimation based on keywords
    base_estimate = 2000  # default
    if "bumper" in damage_desc.lower():
        base_estimate = 3500
    if ("tail light" in damage_desc.lower()
            or "headlight" in damage_desc.lower()):
        base_estimate += 800
    if "airbag" in damage_desc.lower():
        base_estimate += 5000
    if "total" in damage_desc.lower():
        base_estimate = limit * 0.8

    # Apply deductible and cap at limit
    payout = max(0, min(base_estimate - deductible, limit))

    breakdown = (
        f"Repair estimate: ${base_estimate:,.0f}. "
        f"Deductible applied: ${deductible:,.0f}. "
        f"Coverage limit: ${limit:,.0f}. "
        f"Net payout: ${payout:,.0f}."
    )

    claim["estimated_payout"] = payout
    claim["assessment_breakdown"] = breakdown
    claim["status"] = "assessment"

    log_audit(state, "AssessmentAgent",
              f"Estimated payout: ${payout:,.0f}",
              breakdown)
    await ctx.store.set("state", state)
    return f"Assessment complete. Estimated payout: ${payout:,.0f}."


async def make_decision(ctx: Context) -> str:
    """Make final claim decision with reasoning.

    Used by: DecisionAgent.
    Reads: coverage, fraud, assessment data from claim.
    Writes: decision, decision_reasoning to claim.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]

    # Decision logic
    evidence_codes, evidence_complete = coverage_evidence(claim, state.get("policy", {}))
    reason_codes = []
    if not claim.get("coverage_verified", False):
        if evidence_complete and evidence_codes:
            decision = "DENIED"
            reason_codes = evidence_codes
            reasoning = (
                f"Claim denied on recorded coverage evidence: {evidence_codes}. "
                f"{claim.get('coverage_reasoning', '')}"
            )
        else:
            decision = "ESCALATED"
            reason_codes = ["INSUFFICIENT_COVERAGE_EVIDENCE"]
            reasoning = "Coverage could not be established from the supplied evidence; refer for human review."
    elif claim.get("fraud_recommendation") == "investigate":
        decision = "ESCALATED"
        reason_codes = ["FRAUD_REVIEW_REQUIRED"]
        reasoning = (
            f"Claim escalated for fraud investigation. "
            f"Fraud score: {claim.get('fraud_score', 0)}/100. "
            f"Red flags: {claim.get('red_flags', [])}."
        )
    else:
        decision = "APPROVED"
        reason_codes = ["COVERAGE_VERIFIED"]
        reasoning = (
            f"Claim approved. Coverage verified under "
            f"{claim.get('coverage_reasoning', 'policy')}. "
            f"Fraud score: {claim.get('fraud_score', 0)}/100 (low risk). "
            f"Estimated payout: ${claim.get('estimated_payout', 0):,.0f}."
        )

    claim["decision"] = decision
    claim["decision_reasoning"] = reasoning
    claim["decision_reason_codes"] = reason_codes
    claim["status"] = decision.lower()

    log_audit(state, "DecisionAgent",
              f"Decision: {decision}", reasoning)
    await ctx.store.set("state", state)
    return f"Decision: {decision}."


async def validate_compliance(ctx: Context) -> str:
    """Apply the lab's evidence checks; not a legal compliance certification.

    Used by: ComplianceAgent (guardrail — has veto power).
    Reads: decision, reasoning from claim.
    Writes: status update, compliance issues if found.
    """
    state = await ctx.store.get("state")
    claim = state["claim"]
    issues = []

    # Check 1: Denial reason validity
    if claim.get("decision") == "DENIED":
        evidence_codes, complete = coverage_evidence(claim, state.get("policy", {}))
        codes = claim.get("decision_reason_codes", [])
        if (not complete or not evidence_codes or not isinstance(codes, list)
                or not all(isinstance(code, str) for code in codes)
                or set(codes) != set(evidence_codes)):
            issues.append("Denial requires reason codes supported by the policy and loss evidence.")
        if claim.get("coverage_verified") is not False:
            issues.append("Denial contradicts the recorded coverage result.")

    if claim.get("decision") not in {"APPROVED", "DENIED", "ESCALATED"}:
        issues.append("Unrecognized decision value.")
    if claim.get("decision") == "APPROVED":
        codes, complete = coverage_evidence(claim, state.get("policy", {}))
        if not complete or codes or claim.get("coverage_verified") is not True:
            issues.append("Approval requires complete, consistent coverage evidence.")
        if claim.get("fraud_recommendation") == "investigate":
            issues.append("Approval cannot bypass a fraud referral.")

    # Check 2: Reasoning completeness
    if len(claim.get("decision_reasoning", "")) < 50:
        issues.append(
            "Decision reasoning is missing or too short for the lab's documentation check."
        )

    # Check 3: All pipeline stages completed
    required_fields = [
        "coverage_verified", "fraud_score", "estimated_payout",
    ]
    for field in required_fields:
        if claim.get(field) is None:
            issues.append(f"Missing required field: {field}")

    if issues:
        claim["status"] = "compliance_review"
        claim["compliance_issues"] = issues
        log_audit(state, "ComplianceAgent",
                  "BLOCKED — compliance issues", str(issues))
        await ctx.store.set("state", state)
        return f"BLOCKED: {issues}"
    else:
        claim["compliance_issues"] = []
        log_audit(state, "ComplianceAgent",
                  "PASSED — lab evidence checks")
        await ctx.store.set("state", state)
        return "Decision validated against the lab's evidence checks."
