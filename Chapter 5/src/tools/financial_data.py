"""
tools/financial_data.py
========================
Fetches structured financial metrics from the Financial Datasets API.

Returns normalized CompanyMetrics with consistent units:
- Revenue in millions USD
- Market cap in billions USD
- Margins as percentages (0-100)
- Growth rates as percentages

API contracts checked against:
https://docs.financialdatasets.ai/api/financials/income-statements
https://docs.financialdatasets.ai/api/prices/snapshot
https://docs.financialdatasets.ai/api/financial-metrics/snapshot
"""

import math
import httpx
from common import CompanyMetrics, FINANCIAL_DATASETS_API_KEY

API_BASE = "https://api.financialdatasets.ai"


def _number(data: dict, field: str, *, required: bool = False) -> float | None:
    """Preserve missing data; reject malformed/non-finite provider numbers."""
    value = data.get(field)
    if value is None:
        if required:
            raise ValueError(f"Missing required financial field: {field}")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid financial field: {field}")
    return float(value)


def _scaled(value: float | None, factor: float, digits: int = 1) -> float | None:
    return None if value is None else round(value * factor, digits)


async def get_financial_metrics(ticker: str) -> CompanyMetrics:
    """Fetch key financial metrics for a company.

    Retrieves the most recent annual income statement and current price,
    then computes derived metrics (P/E ratio, revenue growth, margins).

    Args:
        ticker: Stock ticker symbol (e.g., 'NVDA', 'AMD', 'INTC').

    Returns:
        CompanyMetrics with normalized financial data.

    Raises:
        ValueError: If ticker not found or API returns an error.
    """
    headers = {"X-API-Key": FINANCIAL_DATASETS_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Fetch income statements (last 2 years for growth calculation)
        income_resp = await client.get(
            f"{API_BASE}/financials/income-statements",
            params={"ticker": ticker, "period": "annual", "limit": 2},
            headers=headers,
        )
        income_resp.raise_for_status()
        income_data = income_resp.json().get("income_statements", [])

        if not income_data:
            raise ValueError(f"No income statement data found for {ticker}")

        # Fetch current price snapshot
        price_resp = await client.get(
            f"{API_BASE}/prices/snapshot",
            params={"ticker": ticker},
            headers=headers,
        )
        price_resp.raise_for_status()
        snapshot = price_resp.json().get("snapshot", {})

        # The price snapshot does not supply market capitalization.
        metrics_resp = await client.get(
            f"{API_BASE}/financial-metrics/snapshot",
            params={"ticker": ticker}, headers=headers,
        )
        metrics_resp.raise_for_status()
        metrics_snapshot = metrics_resp.json().get("snapshot", {})

    current = income_data[0]
    previous = income_data[1] if len(income_data) > 1 else None

    # Extract and normalize values
    revenue_raw = _number(current, "revenue", required=True)
    revenue = revenue_raw / 1_000_000  # Convert to millions

    eps = _number(current, "earnings_per_share_diluted")
    if eps is None:
        eps = _number(current, "earnings_per_share", required=True)
    price = _number(snapshot, "price")

    # Compute derived metrics
    revenue_growth = None
    if previous:
        prev_revenue = _number(previous, "revenue")
        if prev_revenue is not None and prev_revenue > 0:
            revenue_growth = ((revenue_raw - prev_revenue) / prev_revenue) * 100

    pe_ratio = None
    if eps > 0 and price is not None and price > 0:
        pe_ratio = price / eps

    gross_profit = _number(current, "gross_profit")
    operating_income = _number(current, "operating_income")
    gross_margin = None if not revenue_raw or gross_profit is None else round(gross_profit / revenue_raw * 100, 1)
    operating_margin = None if not revenue_raw or operating_income is None else round(operating_income / revenue_raw * 100, 1)
    market_cap = _scaled(_number(metrics_snapshot, "market_cap"), 1e-9)

    company_name = current.get("company_name", ticker)
    # `period` is only a frequency (annual/quarterly), not the reporting date.
    period = current.get("report_period")

    return CompanyMetrics(
        ticker=ticker.upper(),
        company_name=company_name,
        revenue=round(revenue, 1),
        revenue_growth=_scaled(revenue_growth, 1),
        eps=round(eps, 2),
        pe_ratio=_scaled(pe_ratio, 1),
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        market_cap=market_cap,
        period=period,
    )


async def get_price_history(
    ticker: str, days: int = 30
) -> list[dict]:
    """Fetch recent price history for a stock.

    Args:
        ticker: Stock ticker symbol.
        days: Number of days of history to fetch.

    Returns:
        List of dicts with 'date', 'open', 'high', 'low', 'close', 'volume'.
    """
    headers = {"X-API-Key": FINANCIAL_DATASETS_API_KEY}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{API_BASE}/prices",
            params={"ticker": ticker, "interval": "day", "limit": days},
            headers=headers,
        )
        resp.raise_for_status()

    return resp.json().get("prices", [])
