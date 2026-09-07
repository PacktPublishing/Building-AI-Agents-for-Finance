"""
tools/sec_filings.py
=====================
SEC EDGAR integration for fetching filing content.

Uses the SEC EDGAR full-text search API (EFTS) to find and extract
content from 10-K and 10-Q filings. The EDGAR API requires a
User-Agent header with a valid email address.

API docs: https://efts.sec.gov/LATEST/
"""

import httpx
from common import SEC_EDGAR_EMAIL

EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_FULL_TEXT_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_COMPANY_URL = "https://data.sec.gov/submissions"

HEADERS = {
    "User-Agent": f"DeepSearchAgent/1.0 ({SEC_EDGAR_EMAIL})",
    "Accept": "application/json",
}


async def get_risk_factors(ticker: str) -> str:
    """Fetch risk factors from a company's latest 10-K filing.

    Searches SEC EDGAR for the most recent annual report and extracts
    a summary of the risk factors section (Item 1A).

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA').

    Returns:
        Formatted string with key risk factors from the filing.
        Raises ValueError if no usable filing excerpt can be retrieved.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Search for recent 10-K filings
            resp = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f'"{ticker}" "risk factors"',
                    "forms": "10-K",
                    "dateRange": "custom",
                    "startdt": "2024-01-01",
                },
                headers=HEADERS,
            )

            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])

                if hits:
                    # Extract filing information
                    filing = hits[0].get("_source", {})
                    file_date = filing.get("file_date", "Unknown")
                    entity = filing.get("entity_name", ticker)

                    # Get filing text snippet
                    highlight = hits[0].get("highlight", {})
                    snippets = [s for s in highlight.get("text", []) if isinstance(s, str) and s.strip()]
                    if not snippets:
                        raise ValueError(f"No filing excerpt available for {ticker}")

                    risk_text = "\n".join(snippets[:5])  # First 5 snippets

                    return (
                        f"Risk Factors for {entity} ({ticker})\n"
                        f"Source: 10-K filed {file_date}\n"
                        f"{'=' * 50}\n\n"
                        f"{risk_text}\n\n"
                        f"Note: This is an excerpt. Full risk factors section "
                        f"available in the complete 10-K filing on SEC EDGAR."
                    )

            # Fallback: provide a structured response indicating data was not found
            raise ValueError(
                f"Risk Factors for {ticker}\n"
                f"Source: SEC EDGAR (search returned no results)\n"
                f"{'=' * 50}\n\n"
                f"Could not retrieve 10-K risk factors for {ticker} from EDGAR. "
                f"This may be due to rate limiting or the filing not being "
                f"indexed yet. For production use, consider using the SEC "
                f"XBRL API or a commercial data provider for reliable access."
            )

    except Exception as e:
        raise ValueError(
            f"Risk Factors for {ticker}\n"
            f"Source: SEC EDGAR (error)\n"
            f"{'=' * 50}\n\n"
            f"Error retrieving filing data: {str(e)}\n"
            f"The SEC EDGAR API may be rate-limited or temporarily unavailable."
        )


async def get_filing_summary(ticker: str, filing_type: str = "10-K") -> str:
    """Fetch a summary of the most recent filing of a given type.

    Args:
        ticker: Stock ticker symbol.
        filing_type: SEC filing type ('10-K', '10-Q', '8-K').

    Returns:
        Summary text from the filing. Raises ValueError on retrieval failure.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                "https://efts.sec.gov/LATEST/search-index",
                params={
                    "q": f'"{ticker}"',
                    "forms": filing_type,
                    "dateRange": "custom",
                    "startdt": "2024-01-01",
                },
                headers=HEADERS,
            )

            if resp.status_code == 200:
                data = resp.json()
                hits = data.get("hits", {}).get("hits", [])

                if hits:
                    filing = hits[0].get("_source", {})
                    return (
                        f"Filing: {filing_type} for {ticker}\n"
                        f"Filed: {filing.get('file_date', 'Unknown')}\n"
                        f"Entity: {filing.get('entity_name', ticker)}\n"
                    )

        raise ValueError(f"No {filing_type} filing found for {ticker} in recent filings.")

    except Exception as e:
        raise ValueError(f"Error searching for {filing_type} filing") from e
