"""
deep_search_agent.py
=====================
Main orchestrator for the Deep Search Agent — Chapter 5 Lab.

This is the entry point. It runs the full deep research pipeline:

    Plan -> Execute -> Validate -> (Replan?) -> Synthesize

Built on PydanticAI for structured LLM interactions and multi-model
support, with optional cost tracking via pydantic-deep middleware.

Usage:
    python deep_search_agent.py

    # Or with a custom question:
    python deep_search_agent.py --question "Analyze Apple's growth prospects"

    # With custom tickers:
    python deep_search_agent.py --tickers AAPL,MSFT,GOOGL

Requires API keys in .env file (see README or chapter text).
"""

import asyncio
import argparse
import sys
from datetime import datetime

from common import (
    check_all_keys,
    persist_output,
    MAX_REPLAN_ATTEMPTS,
    ResearchReport,
)
from planner import create_research_plan, format_plan
from researcher import execute_plan
from validator import validate_research, format_validation
from synthesizer import synthesize_report, format_report_markdown


# ---------------------------------------------------------------------------
# Optional: cost tracking via pydantic-deep middleware
# ---------------------------------------------------------------------------

def _setup_cost_tracking() -> dict:
    """Set up cost tracking if pydantic-deep is installed.

    Returns a dict that accumulates cost info across the pipeline.
    Cost tracking is optional — the agent works without it.
    """
    costs = {"total_usd": 0.0, "total_input_tokens": 0, "total_output_tokens": 0}

    try:
        from pydantic_ai_middleware import CostTrackingMiddleware, CostInfo

        def on_cost(info: CostInfo) -> None:
            costs["total_usd"] = info.cumulative_cost_usd
            costs["total_input_tokens"] = info.cumulative_input_tokens
            costs["total_output_tokens"] = info.cumulative_output_tokens

        costs["callback"] = on_cost
        costs["available"] = True
        print("  [+] Cost tracking enabled (pydantic-deep middleware)\n")
    except ImportError:
        costs["available"] = False
        print("  [~] Cost tracking unavailable (install pydantic-deep[middleware])\n")

    return costs


# ---------------------------------------------------------------------------
# Default research configuration
# ---------------------------------------------------------------------------

DEFAULT_QUESTION = (
    "Analyze NVIDIA's financial performance and competitive position in the "
    "AI chip market. Compare with AMD and Intel on revenue growth, "
    "profitability, and key risk factors. Produce a structured investment "
    "research memo."
)

DEFAULT_TICKERS = ["NVDA", "AMD", "INTC"]


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

async def deep_search(
    question: str,
    tickers: list[str],
    verbose: bool = True,
) -> ResearchReport:
    """Run the full deep search pipeline.

    Orchestrates the Plan -> Execute -> Validate -> Synthesize loop
    with up to MAX_REPLAN_ATTEMPTS retries when validation fails.

    Args:
        question: The research question to investigate.
        tickers: List of stock tickers that should be covered.
        verbose: Whether to print progress to stdout.

    Returns:
        A structured ResearchReport.
    """
    def log(msg: str) -> None:
        if verbose:
            print(msg)

    costs = _setup_cost_tracking()

    log(f"\n{'=' * 60}")
    log(f"  DEEP SEARCH AGENT")
    log(f"{'=' * 60}")
    log(f"\n  Question: {question}")
    log(f"  Tickers:  {', '.join(tickers)}")
    log(f"{'=' * 60}\n")

    additional_context = ""

    for attempt in range(MAX_REPLAN_ATTEMPTS + 1):
        # ----------------------------------------------------------
        # Phase 1: PLAN
        # ----------------------------------------------------------
        log(f"[Phase 1] Planning (attempt {attempt + 1}/{MAX_REPLAN_ATTEMPTS + 1})...")

        try:
            plan = await create_research_plan(question, additional_context)
        except Exception as e:
            log(f"  [ERROR] Planning failed: {e}")
            if attempt < MAX_REPLAN_ATTEMPTS:
                additional_context = f"Previous planning attempt failed: {e}"
                continue
            raise

        log(f"  Generated {len(plan.sub_tasks)} sub-tasks:")
        parallel = sum(1 for t in plan.sub_tasks if not t.dependencies)
        sequential = len(plan.sub_tasks) - parallel
        log(f"  ({parallel} parallel, {sequential} sequential)\n")

        if verbose:
            print(format_plan(plan))
            print()

        # ----------------------------------------------------------
        # Phase 2: EXECUTE
        # ----------------------------------------------------------
        log("[Phase 2] Executing research plan...\n")

        results = await execute_plan(plan)

        log(f"\n  Completed {len(results)} tasks\n")

        # ----------------------------------------------------------
        # Phase 3: VALIDATE
        # ----------------------------------------------------------
        log("[Phase 3] Validating findings...\n")

        validation = validate_research(plan, results, tickers)

        if verbose:
            print(format_validation(validation))
            print()

        if validation.is_valid:
            log("  Validation passed — proceeding to synthesis\n")
            break
        elif attempt < MAX_REPLAN_ATTEMPTS:
            # Build context for replanning
            gap_info = []
            if validation.gaps:
                gap_info.extend(validation.gaps)
            if validation.errors:
                gap_info.extend(validation.errors)

            additional_context = (
                "Previous research attempt had the following issues "
                "that need to be addressed:\n"
                + "\n".join(f"- {g}" for g in gap_info)
            )
            log(f"  Replanning to address {len(gap_info)} issue(s)...\n")
        else:
            raise RuntimeError(
                "Research validation failed after all replans; synthesis blocked: "
                + "; ".join(validation.errors + validation.gaps)
            )

    # ----------------------------------------------------------
    # Phase 4: SYNTHESIZE
    # ----------------------------------------------------------
    log("[Phase 4] Synthesizing research report...\n")

    report = await synthesize_report(question, results, validation)

    # ----------------------------------------------------------
    # Output
    # ----------------------------------------------------------
    log(f"{'=' * 60}")
    log(f"  {report.title}")
    log(f"{'=' * 60}\n")

    # Print executive summary
    log(report.executive_summary)
    log(f"\n  Confidence: {report.confidence_score:.0%}")

    # Print cost summary if available
    if costs.get("available"):
        log(f"\n  Cost: ${costs['total_usd']:.4f}")
        log(f"  Tokens: {costs['total_input_tokens']:,} in / "
            f"{costs['total_output_tokens']:,} out")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Search Agent — Autonomous Financial Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python deep_search_agent.py
  python deep_search_agent.py --question "Analyze Apple's growth prospects"
  python deep_search_agent.py --tickers AAPL,MSFT,GOOGL
  python deep_search_agent.py --question "Compare JPM and GS" --tickers JPM,GS
        """,
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        default=DEFAULT_QUESTION,
        help="Research question to investigate",
    )
    parser.add_argument(
        "--tickers", "-t",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated list of tickers to cover (e.g., NVDA,AMD,INTC)",
    )
    parser.add_argument(
        "--output-format",
        choices=["markdown", "json", "both"],
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--quiet", "-Q",
        action="store_true",
        help="Suppress progress output (only print final report)",
    )
    return parser.parse_args()


async def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Check API keys
    if not check_all_keys():
        print(
            "\nMissing required API keys. Please set them in a .env file:\n"
            "  OPENAI_API_KEY=sk-...      (or ANTHROPIC_API_KEY=sk-ant-...)\n"
            "  FINANCIAL_DATASETS_API_KEY=...\n"
            "  TAVILY_API_KEY=tvly-...\n"
            "  SEC_EDGAR_EMAIL=your-email@example.com (optional)\n"
        )
        sys.exit(1)

    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    # Run deep search
    report = await deep_search(
        question=args.question,
        tickers=tickers,
        verbose=not args.quiet,
    )

    # Save outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tickers_str = "_".join(tickers)

    if args.output_format in ("markdown", "both"):
        md_content = format_report_markdown(report)
        md_file = persist_output(
            f"report_{tickers_str}_{timestamp}.md", md_content
        )
        print(f"\n  Markdown report: {md_file}")

    if args.output_format in ("json", "both"):
        json_content = report.model_dump_json(indent=2)
        json_file = persist_output(
            f"report_{tickers_str}_{timestamp}.json", json_content
        )
        print(f"  JSON report:     {json_file}")

    print(f"\n  Deep Search Agent completed successfully.\n")


if __name__ == "__main__":
    asyncio.run(main())
