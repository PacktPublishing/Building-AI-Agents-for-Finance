"""
synthesizer.py
===============
Report synthesis component for the Deep Search Agent.

Takes validated research findings and produces a structured
investment research report using PydanticAI's structured output.
The synthesizer's job is narrative quality — combining
quantitative data with qualitative analysis into a coherent,
analyst-grade research memo.
"""

from pydantic_ai import Agent
from common import (
    ResearchReport,
    ValidationResult,
    SYNTHESIS_MODEL,
)

SYNTHESIZER_SYSTEM_PROMPT = """\
You are a senior equity research analyst at a top-tier investment bank.
Your task is to produce a structured investment research memo based on
the gathered data and analysis provided below.

Format requirements:
1. **Title**: Clear, descriptive title (e.g., "NVIDIA vs AMD vs Intel:
   AI Chip Market Investment Analysis")
2. **Executive Summary**: 3-5 sentences capturing the key findings and
   investment thesis. Lead with the most important insight.
3. **Company Metrics**: Include all companies analyzed with their key
   financial metrics. The data will be structured separately.
4. **Competitive Analysis**: 2-3 paragraphs analyzing relative positioning,
   competitive advantages, and market dynamics. Use specific numbers.
5. **Risk Factors**: Key risks identified, ranked by severity and
   likelihood. Be specific — avoid generic risk statements.
6. **Conclusion**: Investment stance with supporting rationale. Be direct.
7. **Sources**: List all data sources used.

Critical rules:
- Be PRECISE with numbers. Use the exact figures from the gathered data.
- NEVER fabricate or estimate data. If a metric is missing, explicitly
  note it as "Data not available" rather than guessing.
- Reference specific sources for key claims.
- If the data contains warnings or validation issues, acknowledge them.
- Assign a confidence score (0-1) reflecting data completeness and quality.\
"""

# PydanticAI agent with structured output — replaces raw OpenAI API calls
synthesizer_agent = Agent(
    model=SYNTHESIS_MODEL,
    system_prompt=SYNTHESIZER_SYSTEM_PROMPT,
    output_type=ResearchReport,
    retries=2,
)


async def synthesize_report(
    question: str,
    results: dict[int, str],
    validation: ValidationResult,
) -> ResearchReport:
    """Synthesize validated findings into a structured research report.

    Args:
        question: The original research question.
        results: Mapping of task_id -> result string from execution.
        validation: The validation result (including any warnings).

    Returns:
        A structured ResearchReport.
    """
    # Compile all findings into a single context block
    findings_parts = []
    for task_id in sorted(results.keys()):
        findings_parts.append(
            f"=== Task {task_id} Result ===\n{results[task_id]}"
        )
    findings = "\n\n".join(findings_parts)

    # Include validation context
    context = findings
    if validation.warnings:
        context += "\n\n=== Validation Warnings ===\n"
        context += "\n".join(f"- {w}" for w in validation.warnings)

    if validation.gaps:
        context += "\n\n=== Data Gaps (acknowledged) ===\n"
        context += "\n".join(f"- {g}" for g in validation.gaps)

    # Generate the report using PydanticAI structured output
    prompt = (
        f"Research question: {question}\n\n"
        f"Gathered data and analysis:\n{context}"
    )

    result = await synthesizer_agent.run(prompt)
    return result.output


def format_report_markdown(report: ResearchReport) -> str:
    """Convert a ResearchReport to a formatted Markdown document.

    Args:
        report: The structured research report.

    Returns:
        Complete Markdown document string.
    """
    lines = [
        f"# {report.title}",
        "",
        f"**Confidence Score: {report.confidence_score:.0%}**",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
    ]

    # Company metrics table
    if report.company_metrics:
        lines.extend([
            "## Key Metrics Comparison",
            "",
            "| Metric | " + " | ".join(m.ticker for m in report.company_metrics) + " |",
            "| --- | " + " | ".join("---" for _ in report.company_metrics) + " |",
        ])

        # Revenue
        lines.append(
            "| Revenue | "
            + " | ".join(f"${m.revenue:,.0f}M" for m in report.company_metrics)
            + " |"
        )
        # Revenue Growth
        lines.append(
            "| Revenue Growth | "
            + " | ".join((f"{m.revenue_growth:+.1f}%" if m.revenue_growth is not None else "N/A") for m in report.company_metrics)
            + " |"
        )
        # EPS
        lines.append(
            "| EPS | "
            + " | ".join(f"${m.eps:.2f}" for m in report.company_metrics)
            + " |"
        )
        # P/E Ratio
        lines.append(
            "| P/E Ratio | "
            + " | ".join(
                f"{m.pe_ratio:.1f}x" if m.pe_ratio is not None and m.pe_ratio > 0 else "N/A"
                for m in report.company_metrics
            )
            + " |"
        )
        # Gross Margin
        lines.append(
            "| Gross Margin | "
            + " | ".join((f"{m.gross_margin:.1f}%" if m.gross_margin is not None else "N/A") for m in report.company_metrics)
            + " |"
        )
        # Operating Margin
        lines.append(
            "| Operating Margin | "
            + " | ".join((f"{m.operating_margin:.1f}%" if m.operating_margin is not None else "N/A") for m in report.company_metrics)
            + " |"
        )
        # Market Cap
        lines.append(
            "| Market Cap | "
            + " | ".join((f"${m.market_cap:,.0f}B" if m.market_cap is not None else "N/A") for m in report.company_metrics)
            + " |"
        )
        lines.append("")

    lines.extend([
        "## Competitive Analysis",
        "",
        report.competitive_analysis,
        "",
        "## Risk Factors",
        "",
        report.risk_factors,
        "",
        "## Conclusion",
        "",
        report.conclusion,
        "",
    ])

    if report.sources:
        lines.extend([
            "## Sources",
            "",
        ])
        for source in report.sources:
            lines.append(f"- {source}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "*Report generated by Deep Search Agent (Chapter 5 Lab)*",
    ])

    return "\n".join(lines)
