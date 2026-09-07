"""
common.py
=========
Shared data models, configuration, and utility functions for the
Chapter 5 Deep Search Agent.

All Pydantic models that flow between components are defined here,
establishing the contracts between Planner, Researcher, Validator,
and Synthesizer.
"""

import os
from enum import Enum
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
_ = load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINANCIAL_DATASETS_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SEC_EDGAR_EMAIL = os.getenv("SEC_EDGAR_EMAIL", "researcher@example.com")

# Model configuration — PydanticAI model strings.
# Format: "provider:model-name" (e.g., "openai:gpt-4o", "anthropic:claude-sonnet-5")
PLANNING_MODEL = "openai:gpt-4o"
ROUTING_MODEL = "openai:gpt-4o-mini"
SYNTHESIS_MODEL = "openai:gpt-4o"
SUMMARIZATION_MODEL = "openai:gpt-4o-mini"

# Research loop configuration
MAX_REPLAN_ATTEMPTS = 2
MAX_CONTEXT_TOKENS = 80_000  # Soft limit for context management


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
# Research Plan Models
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
        description="IDs of tasks that must complete before this one"
    )
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class ResearchPlan(BaseModel):
    """A structured research plan with sub-tasks and dependencies."""
    question: str
    sub_tasks: list[SubTask]


# ---------------------------------------------------------------------------
# Financial Data Models
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
# Validation Models
# ---------------------------------------------------------------------------

class ValidationResult(BaseModel):
    """Result of validating research findings."""
    is_valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Report Models
# ---------------------------------------------------------------------------

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
        default=0.7,
        ge=0, le=1,
        description="Overall confidence in the report (0-1)"
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def check_api_key(key_name: str) -> bool:
    """Check if an API key is set and print its status."""
    key = os.getenv(key_name)
    status = "+" if key else "X"
    print(f"  [{status}] {key_name}: {'configured' if key else 'not configured'}")
    return bool(key)


def check_all_keys() -> bool:
    """Check all required API keys and return True if all are present."""
    print("Checking API keys...")

    # The supplied model configuration and dependency extras use OpenAI.
    llm_ok = check_api_key("OPENAI_API_KEY")

    data_keys = ["FINANCIAL_DATASETS_API_KEY", "TAVILY_API_KEY"]
    results = [check_api_key(k) for k in data_keys]

    # SEC_EDGAR_EMAIL is recommended but not strictly required
    email = os.getenv("SEC_EDGAR_EMAIL")
    status = "+" if email else "~"
    print(f"  [{status}] SEC_EDGAR_EMAIL: {email or 'Using default (rate-limited)'}")
    print()

    return llm_ok and all(results)


def persist_output(filename: str, content: str) -> str:
    """Save output to a file in the output directory."""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    with open(filepath, "w") as f:
        f.write(content)
    print(f"Saved to {filepath}")
    return filepath


def extract_ticker(text: str) -> str | None:
    """Extract a stock ticker from a task description.

    Looks for common patterns like 'NVDA', 'NVIDIA (NVDA)', etc.
    """
    import re

    # Common tickers we expect to encounter
    known_tickers = {
        "NVDA": "NVIDIA", "AMD": "AMD", "INTC": "Intel",
        "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google",
        "AMZN": "Amazon", "META": "Meta", "TSLA": "Tesla",
        "JPM": "JPMorgan", "GS": "Goldman Sachs", "BAC": "Bank of America",
    }

    # Try to find ticker in parentheses: "NVIDIA (NVDA)"
    match = re.search(r'\(([A-Z]{1,5})\)', text)
    if match and match.group(1) in known_tickers:
        return match.group(1)

    # Try to find standalone ticker
    for ticker, name in known_tickers.items():
        if ticker in text.upper() or name.upper() in text.upper():
            return ticker

    return None
