"""
researcher.py
==============
Plan execution component for the Deep Search Agent.

Executes a ResearchPlan by running sub-tasks according to their
dependency graph. Independent tasks execute in parallel using
asyncio.gather(); dependent tasks wait for their prerequisites.

Uses PydanticAI Agent for LLM-based analysis tasks.
"""

import asyncio
from pydantic_ai import Agent

from common import (
    ResearchPlan,
    SubTask,
    TaskStatus,
    CompanyMetrics,
    SYNTHESIS_MODEL,
    extract_ticker,
)
from tools.financial_data import get_financial_metrics
from tools.sec_filings import get_risk_factors
from tools.web_search import search_financial_news

# PydanticAI agent for analysis tasks — replaces raw OpenAI API calls
analyst_agent = Agent(
    model=SYNTHESIS_MODEL,
    system_prompt=(
        "You are a senior financial analyst. Analyze the provided data "
        "and produce a clear, factual response. Use specific numbers "
        "from the data. Do not fabricate data — if information is "
        "missing, state that explicitly."
    ),
    retries=1,
)


async def execute_plan(plan: ResearchPlan) -> dict[int, str]:
    """Execute a research plan, respecting task dependencies.

    Tasks with no unmet dependencies execute in parallel.
    Each wave of parallel tasks completes before the next
    wave of now-unblocked tasks begins.

    Args:
        plan: The ResearchPlan to execute.

    Returns:
        Mapping of task_id -> result string for all tasks.
    """
    results: dict[int, str] = {}
    tasks_by_id = {t.id: t for t in plan.sub_tasks}

    while any(t.status == TaskStatus.PENDING for t in plan.sub_tasks):
        # Find tasks whose dependencies are all completed
        ready = [
            t
            for t in plan.sub_tasks
            if t.status == TaskStatus.PENDING
            and all(
                tasks_by_id[dep].status == TaskStatus.COMPLETED
                for dep in t.dependencies
            )
        ]

        if not ready:
            # No tasks can proceed — either deadlock or all done
            pending = [
                t for t in plan.sub_tasks if t.status == TaskStatus.PENDING
            ]
            if pending:
                # Mark remaining as failed (unresolvable dependencies)
                for t in pending:
                    t.status = TaskStatus.FAILED
                    t.result = "Could not execute: unresolved dependencies"
                    results[t.id] = t.result
            break

        # Execute ready tasks in parallel
        async def _execute_one(task: SubTask) -> tuple[int, str]:
            task.status = TaskStatus.IN_PROGRESS
            try:
                result = await _route_and_execute(task, results)
                if not isinstance(result, str) or not result.strip():
                    raise ValueError("Task returned no evidence")
                task.status = TaskStatus.COMPLETED
                task.result = result
                return (task.id, result)
            except Exception as e:
                task.status = TaskStatus.FAILED
                error_msg = f"[FAILED] Task {task.id}: {str(e)}"
                task.result = error_msg
                return (task.id, error_msg)

        completed = await asyncio.gather(
            *[_execute_one(t) for t in ready]
        )

        for task_id, result in completed:
            results[task_id] = result
            status = tasks_by_id[task_id].status.value
            print(f"    [{status}] Task {task_id}: {tasks_by_id[task_id].description[:60]}")

    return results


async def _route_and_execute(
    task: SubTask, prior_results: dict[int, str]
) -> str:
    """Route a sub-task to the appropriate tool(s) and execute it.

    Determines which tool to call based on the task's data_sources
    field, then executes and returns the result.

    Args:
        task: The sub-task to execute.
        prior_results: Results from completed dependency tasks.

    Returns:
        Result string from executing the task.
    """
    # Build context from dependency results
    context = ""
    if task.dependencies:
        dep_results = [
            prior_results[dep_id]
            for dep_id in task.dependencies
            if dep_id in prior_results
        ]
        context = "\n\n---\n\n".join(dep_results)

    # Determine which tool to use
    sources = task.data_sources

    # Financial API — structured data
    if "financial_api" in sources and not task.dependencies:
        ticker = extract_ticker(task.description)
        if ticker:
            metrics = await get_financial_metrics(ticker)
            return metrics.model_dump_json(indent=2)

    # SEC Filing — qualitative data
    if "sec_filing" in sources and "financial_api" not in sources:
        ticker = extract_ticker(task.description)
        if ticker:
            return await get_risk_factors(ticker)

    # Web Search — news and commentary
    if "web_search" in sources and len(sources) == 1:
        return await search_financial_news(task.description)

    # Mixed sources or analysis tasks — use PydanticAI agent
    return await _llm_analyze(task.description, context)


async def _llm_analyze(description: str, context: str) -> str:
    """Use a PydanticAI agent to analyze gathered data.

    Called for tasks that require reasoning rather than tool calls,
    such as comparative analysis or intermediate synthesis.

    Args:
        description: What the task should accomplish.
        context: Prior findings to reason about.

    Returns:
        LLM-generated analysis text.
    """
    prompt = (
        f"Task: {description}\n\n"
        f"Available data:\n{context if context else 'No prior data.'}"
    )

    result = await analyst_agent.run(prompt)
    return result.output
