"""
common.py — Shared models and configuration for Chapter 5 Deep Search Agent labs.

Defines the typed contracts that flow between the four components of the
Deep Search Agent — Planner, Researcher, Validator, Synthesizer.

Keeping the schema definitions here means each lab notebook can focus on
the *behaviour* it is demonstrating rather than redefining models.
"""
from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model configuration — PydanticAI provider strings.
# Override these from a notebook to swap models per phase.
# ---------------------------------------------------------------------------
PLANNING_MODEL = "openai:gpt-4o"
SYNTHESIS_MODEL = "openai:gpt-4o"
SUMMARIZATION_MODEL = "openai:gpt-4o-mini"

# Research loop limits
MAX_REPLAN_ATTEMPTS = 2


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, Enum):
    """Status of a research sub-task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class DataSource(str, Enum):
    """Available data sources for research tools."""
    FINANCIAL_API = "financial_api"
    SEC_FILING = "sec_filing"
    WEB_SEARCH = "web_search"


# ---------------------------------------------------------------------------
# Research plan models
# ---------------------------------------------------------------------------

class SubTask(BaseModel):
    """A single research sub-task within a plan."""
    id: int
    description: str
    data_sources: list[str] = Field(
        description="Which tools to use: 'financial_api', 'sec_filing', 'web_search'"
    )
    dependencies: list[int] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one can start",
    )
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class ResearchPlan(BaseModel):
    """A structured research plan with sub-tasks and a dependency graph."""
    question: str
    sub_tasks: list[SubTask]


# ---------------------------------------------------------------------------
# Financial data models
# ---------------------------------------------------------------------------

class CompanyMetrics(BaseModel):
    """Standardized financial metrics for a company."""
    ticker: str
    company_name: str = ""
    revenue: float = Field(description="Revenue in millions USD")
    revenue_growth: float | None = Field(
        default=None, description="Year-over-year revenue growth %; None if unavailable"
    )
    eps: float = Field(description="Earnings per share")
    pe_ratio: float | None = Field(default=None, description="Price-to-earnings ratio; None if unavailable or not meaningful")
    gross_margin: float | None = Field(default=None, description="Gross margin %")
    operating_margin: float | None = Field(default=None, description="Operating margin %")
    market_cap: float | None = Field(default=None, description="Market cap in billions USD")
    period: str | None = Field(default=None, description="Reported fiscal period; never inferred")


# ---------------------------------------------------------------------------
# Validation + report models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of validating research findings."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final structured investment research report."""
    title: str
    executive_summary: str
    company_metrics: list[CompanyMetrics] = Field(default_factory=list)
    competitive_analysis: str
    risk_factors: str
    conclusion: str
    sources: list[str] = Field(default_factory=list)
    confidence_score: float = Field(
        default=0.7, ge=0, le=1,
        description="Overall confidence in the report (0-1)",
    )


# ---------------------------------------------------------------------------
# Utilities used across labs
# ---------------------------------------------------------------------------

KNOWN_TICKERS: dict[str, str] = {
    "NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel",
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
    "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
    "JPM": "JPMorgan", "GS": "Goldman Sachs", "BAC": "Bank of America",
}


def extract_ticker(text: str) -> str | None:
    """Extract a stock ticker from a task description.

    Looks for tickers in parentheses ('NVIDIA (NVDA)') or as bare uppercase
    words matching `KNOWN_TICKERS`.
    """
    import re
    match = re.search(r"\(([A-Z]{1,5})\)", text)
    if match and match.group(1) in KNOWN_TICKERS:
        return match.group(1)
    for ticker, name in KNOWN_TICKERS.items():
        if ticker in text.upper() or name.upper() in text.upper():
            return ticker
    return None


def format_plan(plan: ResearchPlan) -> str:
    """Pretty-print a research plan for display."""
    lines = [
        f"Research Plan: {plan.question}",
        f"Sub-tasks: {len(plan.sub_tasks)}",
        "-" * 50,
    ]
    for task in plan.sub_tasks:
        deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
        sources = ", ".join(task.data_sources)
        lines.append(
            f"  [{task.id}] {task.description}\n"
            f"       Sources: {sources}{deps}"
        )
    return "\n".join(lines)


def format_validation(result: ValidationResult) -> str:
    """Pretty-print a ValidationResult."""
    lines = ["[PASS] Validation passed" if result.is_valid else "[FAIL] Validation failed"]
    if result.errors:
        lines.append("\n  Errors:")
        lines.extend(f"    [X] {e}" for e in result.errors)
    if result.warnings:
        lines.append("\n  Warnings:")
        lines.extend(f"    [!] {w}" for w in result.warnings)
    if result.gaps:
        lines.append("\n  Gaps:")
        lines.extend(f"    [ ] {g}" for g in result.gaps)
    return "\n".join(lines)
