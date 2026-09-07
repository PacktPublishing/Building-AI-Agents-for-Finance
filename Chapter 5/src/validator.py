"""
validator.py
=============
Deterministic validation for the Deep Search Agent.

Validates research findings using code-based checks rather than
LLM-based verification. This follows the key insight from the
AI Alliance Deep Research Agent: deterministic validation is
faster, cheaper, and more reliable than asking an LLM to verify.

Checks include:
- Completeness: all tasks completed, all companies covered
- Numerical validity: metrics in reasonable ranges
- Consistency: no contradictory data points (e.g., operating
  margin exceeding gross margin)
"""

import math

from common import (
    ResearchPlan,
    CompanyMetrics,
    ValidationResult,
    TaskStatus,
)


def validate_research(
    plan: ResearchPlan,
    results: dict[int, str],
    required_tickers: list[str],
) -> ValidationResult:
    """Validate research findings using deterministic checks.

    Args:
        plan: The executed research plan.
        results: Mapping of task_id -> result string.
        required_tickers: Tickers that must have data in the results.

    Returns:
        ValidationResult with errors, warnings, and gaps.
    """
    errors: list[str] = []
    warnings: list[str] = []
    gaps: list[str] = []

    # ---------------------------------------------------------------
    # Check 1: Task completion
    # ---------------------------------------------------------------
    if not plan.sub_tasks:
        gaps.append("Research plan contains no tasks")
    failed_tasks = [t for t in plan.sub_tasks if t.status != TaskStatus.COMPLETED]
    for task in failed_tasks:
        gaps.append(f"Task {task.id} is not completed: {task.description}")

    completed_count = sum(
        1 for t in plan.sub_tasks if t.status == TaskStatus.COMPLETED
    )
    total_count = len(plan.sub_tasks)
    if completed_count < total_count:
        completion_pct = completed_count / total_count * 100
        if completion_pct < 50:
            errors.append(
                f"Only {completed_count}/{total_count} tasks completed "
                f"({completion_pct:.0f}%) — insufficient data for synthesis"
            )
        else:
            warnings.append(
                f"{completed_count}/{total_count} tasks completed "
                f"({completion_pct:.0f}%)"
            )

    # ---------------------------------------------------------------
    # Check 2: Only completed tasks with parsed metrics establish coverage.
    # ---------------------------------------------------------------
    found_tickers: set[str] = set()
    parsed_metrics: list[CompanyMetrics] = []
    for task in plan.sub_tasks:
        if task.status != TaskStatus.COMPLETED:
            continue
        result = results.get(task.id)
        if not isinstance(result, str) or not result.strip():
            gaps.append(f"Task {task.id} has no result")
            continue
        try:
            metrics = CompanyMetrics.model_validate_json(result)
        except (ValueError, TypeError):
            if "financial_api" in task.data_sources and not task.dependencies:
                errors.append(f"Task {task.id} did not return valid financial metrics")
            continue  # Qualitative text cannot establish financial-data coverage.
        parsed_metrics.append(metrics)
        _validate_metrics(metrics, errors, warnings)
        found_tickers.add(metrics.ticker.upper())

    for ticker in sorted(set(t.upper() for t in required_tickers) - found_tickers):
        gaps.append(f"Missing structured financial data for required ticker: {ticker}")

    # ---------------------------------------------------------------
    # Check 4: Cross-company consistency
    # ---------------------------------------------------------------
    if len(parsed_metrics) >= 2:
        _validate_cross_company(parsed_metrics, warnings)

    # ---------------------------------------------------------------
    # Determine overall validity
    # ---------------------------------------------------------------
    is_valid = len(errors) == 0 and len(gaps) == 0

    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        gaps=gaps,
    )


def _validate_metrics(
    metrics: CompanyMetrics,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate a single CompanyMetrics instance for reasonableness."""
    ticker = metrics.ticker
    invalid = False
    if not metrics.period or not metrics.period.strip():
        errors.append(f"{ticker}: missing reported fiscal period")
        invalid = True
    for field in ("revenue", "revenue_growth", "eps", "pe_ratio", "gross_margin", "operating_margin", "market_cap"):
        value = getattr(metrics, field)
        # P/E is not meaningful for zero or negative EPS. Other comparison
        # fields are required by this lab; missing is not a reported zero.
        if field == "pe_ratio" and value is None and metrics.eps <= 0:
            warnings.append(f"{ticker}: P/E unavailable for non-positive EPS")
            continue
        if value is None or not math.isfinite(value):
            errors.append(f"{ticker}: {field} must be finite")
            invalid = True
    if invalid:
        return

    # P/E ratio checks
    if metrics.pe_ratio is not None and metrics.pe_ratio < 0:
        warnings.append(
            f"{ticker}: Negative P/E ratio ({metrics.pe_ratio}) "
            f"— company may be unprofitable or EPS data incorrect"
        )
    elif metrics.pe_ratio is not None and metrics.pe_ratio > 300:
        warnings.append(
            f"{ticker}: Extremely high P/E ratio ({metrics.pe_ratio}) "
            f"— verify EPS data or check if company is pre-profit"
        )

    # Margin range checks
    if metrics.gross_margin != 0 and not (-10 <= metrics.gross_margin <= 100):
        errors.append(
            f"{ticker}: Gross margin {metrics.gross_margin}% "
            f"outside plausible range [-10, 100]"
        )

    if metrics.operating_margin != 0 and not (-100 <= metrics.operating_margin <= 100):
        errors.append(
            f"{ticker}: Operating margin {metrics.operating_margin}% "
            f"outside plausible range [-100, 100]"
        )

    # Margin consistency: operating margin should not exceed gross margin
    # (unless both are near zero / default)
    if (
        metrics.gross_margin > 5
        and metrics.operating_margin > metrics.gross_margin + 1
    ):
        errors.append(
            f"{ticker}: Operating margin ({metrics.operating_margin}%) "
            f"exceeds gross margin ({metrics.gross_margin}%) "
            f"— this is mathematically impossible"
        )

    # Revenue growth sanity
    if abs(metrics.revenue_growth) > 500:
        warnings.append(
            f"{ticker}: Revenue growth of {metrics.revenue_growth}% "
            f"is extreme — verify data accuracy"
        )

    # Revenue sanity (should be positive for established companies)
    if metrics.revenue <= 0:
        warnings.append(
            f"{ticker}: Revenue is ${metrics.revenue}M "
            f"— verify this is correct"
        )

    # Market cap sanity
    if metrics.market_cap < 0:
        errors.append(f"{ticker}: Negative market cap (${metrics.market_cap}B)")


def _validate_cross_company(
    metrics_list: list[CompanyMetrics],
    warnings: list[str],
) -> None:
    """Cross-validate metrics across multiple companies."""
    # Check that all companies report the same fiscal period
    periods = {m.period for m in metrics_list}
    if len(periods) > 1:
        tickers_periods = ", ".join(
            f"{m.ticker}={m.period}" for m in metrics_list
        )
        warnings.append(
            f"Fiscal period mismatch across companies: {tickers_periods}. "
            f"Comparisons may not be apples-to-apples."
        )


def format_validation(result: ValidationResult) -> str:
    """Format a ValidationResult for display."""
    lines = []

    if result.is_valid:
        lines.append("[PASS] Validation passed")
    else:
        lines.append("[FAIL] Validation failed")

    if result.errors:
        lines.append("\n  Errors:")
        for e in result.errors:
            lines.append(f"    [X] {e}")

    if result.warnings:
        lines.append("\n  Warnings:")
        for w in result.warnings:
            lines.append(f"    [!] {w}")

    if result.gaps:
        lines.append("\n  Gaps:")
        for g in result.gaps:
            lines.append(f"    [ ] {g}")

    return "\n".join(lines)
