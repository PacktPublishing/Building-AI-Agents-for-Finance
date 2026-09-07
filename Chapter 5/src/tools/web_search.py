"""
tools/web_search.py
====================
Web search integration for fetching recent financial news.

Uses the Tavily search API with domain filtering to ensure results
come from reputable financial news sources.

API docs: https://docs.tavily.com/
"""

import httpx
from common import TAVILY_API_KEY

# Trusted financial news domains
FINANCIAL_DOMAINS = [
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "cnbc.com",
    "seekingalpha.com",
    "ft.com",
    "marketwatch.com",
    "barrons.com",
]


async def search_financial_news(
    query: str, max_results: int = 5
) -> str:
    """Search for recent financial news articles.

    Queries Tavily's search API with domain filtering to return
    only results from reputable financial news sources.

    Args:
        query: Natural language search query.
        max_results: Maximum number of articles to return (default 5).

    Returns:
        Formatted string with article titles, excerpts, and source URLs.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_domains": FINANCIAL_DOMAINS,
                    "max_results": max_results,
                },
            )
            resp.raise_for_status()

        results = [r for r in resp.json().get("results", [])
                   if isinstance(r, dict) and isinstance(r.get("content"), str)
                   and r["content"].strip() and r.get("url")]

        if not results:
            raise ValueError(f"No recent news found for: {query}")

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            content = r.get("content", "No excerpt available.")
            url = r.get("url", "")
            # Truncate content to keep it concise
            excerpt = content[:400] + "..." if len(content) > 400 else content
            formatted.append(
                f"[{i}] {title}\n"
                f"    {excerpt}\n"
                f"    Source: {url}"
            )

        header = f"Financial News Search: '{query}' ({len(results)} results)\n"
        header += "=" * 60

        return header + "\n\n" + "\n\n".join(formatted)

    except httpx.HTTPStatusError as e:
        raise ValueError(
            f"Web search error (HTTP {e.response.status_code}): "
            f"Could not search for '{query}'. "
            f"Check your TAVILY_API_KEY configuration."
        )
    except Exception as e:
        raise ValueError("Financial news retrieval failed") from e


async def search_general(query: str, max_results: int = 3) -> str:
    """General web search without domain filtering.

    Use this for broader queries that may not be covered by
    financial news sites (e.g., technical details, industry reports).

    Args:
        query: Natural language search query.
        max_results: Maximum number of results to return.

    Returns:
        Formatted search results.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                },
            )
            resp.raise_for_status()

        results = [r for r in resp.json().get("results", [])
                   if isinstance(r, dict) and isinstance(r.get("content"), str)
                   and r["content"].strip() and r.get("url")]

        if not results:
            raise ValueError(f"No results found for: {query}")

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "Untitled")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            formatted.append(f"[{i}] {title}\n    {content}...\n    Source: {url}")

        return "\n\n".join(formatted)

    except Exception as e:
        raise ValueError("Web search retrieval failed") from e
