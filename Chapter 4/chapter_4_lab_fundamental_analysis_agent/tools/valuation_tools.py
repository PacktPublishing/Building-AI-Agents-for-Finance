"""
valuation_tools.py
------------------
Tools for intrinsic value estimation: DCF model and peer multiples comparison.
"""

import json
import math
import yfinance as yf
from agents import function_tool

# Sector → list of representative peer tickers
SECTOR_PEERS: dict[str, list[str]] = {
    "Technology":           ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "SNAP"],
    "Consumer Cyclical":    ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Consumer Defensive":   ["PG", "KO", "PEP", "WMT", "COST"],
    "Financial Services":   ["JPM", "BAC", "WFC", "GS", "V"],
    "Healthcare":           ["JNJ", "UNH", "PFE", "MRK", "ABBV"],
    "Industrials":          ["GE", "HON", "CAT", "UPS", "BA"],
    "Energy":               ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities":            ["NEE", "DUK", "SO", "D", "AEP"],
    "Real Estate":          ["AMT", "PLD", "EQIX", "SPG", "O"],
    "Basic Materials":      ["LIN", "APD", "ECL", "NEM", "FCX"],
}


# Weights for the peer-based verdict (must sum to 1.0)
_PEER_WEIGHTS: dict[str, float] = {
    "forward_pe": 0.30,
    "ev_ebitda":  0.25,
    "pe":         0.20,
    "ps":         0.15,
    "pb":         0.10,
}

# Thresholds: weighted average premium (%) → verdict
# Positive premium = stock is more expensive than peers = overvalued
_PEER_VERDICT_THRESHOLDS = [
    ( 30,  float("inf"), "SIGNIFICANTLY OVERVALUED"),
    ( 15,           30,  "SLIGHTLY OVERVALUED"),
    (-15,           15,  "FAIRLY VALUED"),
    (-30,          -15,  "SLIGHTLY UNDERVALUED"),
    (float("-inf"), -30, "SIGNIFICANTLY UNDERVALUED"),
]


def _peer_verdict(subject: dict, peer_avg: dict) -> tuple[str, float]:
    """Compute a weighted average premium vs peers and map it to the 5-level scale.

    Returns (verdict, weighted_avg_premium_pct).
    Only multiples where both subject and peer_avg are available are included.
    Weights are renormalized to the available multiples so the result is always
    on the same scale regardless of how many multiples are present.
    """
    weighted_sum = 0.0
    total_weight = 0.0

    for key, weight in _PEER_WEIGHTS.items():
        s_val = subject.get(key)
        p_val = peer_avg.get(key)
        if s_val is None or p_val is None or p_val == 0:
            continue
        premium_pct = ((s_val - p_val) / abs(p_val)) * 100
        weighted_sum += premium_pct * weight
        total_weight += weight

    if total_weight == 0:
        return "FAIRLY VALUED", 0.0   # no data available — neutral fallback

    weighted_avg = round(weighted_sum / total_weight, 1)

    for low, high, label in _PEER_VERDICT_THRESHOLDS:
        if low <= weighted_avg < high:
            return label, weighted_avg

    return "FAIRLY VALUED", weighted_avg  # should never reach here


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        f = float(value)
        import math
        return None if math.isnan(f) or math.isinf(f) else round(f, 2)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Tool 1: DCF Valuation Model
# ---------------------------------------------------------------------------

@function_tool
def run_dcf_model(
    ticker: str,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
) -> str:
    """
    Run a Discounted Cash Flow (DCF) valuation model for a given stock.

    Projects free cash flow over 5 years using the provided growth rate,
    discounts back at the given WACC/discount rate, and adds a terminal value
    using the Gordon Growth Model. Returns intrinsic value per share vs
    current market price to assess over/undervaluation.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL'.
        growth_rate: Expected annual FCF growth rate for the next 5 years (e.g. 0.10 for 10%).
        discount_rate: Weighted average cost of capital / required return (e.g. 0.09 for 9%).
        terminal_growth_rate: Perpetual growth rate after year 5 (e.g. 0.03 for 3%).
    """
    try:
        stock = yf.Ticker(ticker.upper())
        info  = stock.info

        # Provider FCF is a teaching proxy, not verified unlevered FCFF.
        # Operating cash flow alone omits capital expenditure: no fallback.
        base_fcf = _safe_float(info.get("freeCashflow"))
        if base_fcf is None or base_fcf <= 0:
            return json.dumps({
                "error": "This simplified DCF requires positive, finite free cash flow.",
                "ticker": ticker.upper(),
            })

        base_fcf = float(base_fcf)
        shares = _safe_float(info.get("sharesOutstanding") or info.get("impliedSharesOutstanding"))
        if shares is None or shares <= 0:
            return json.dumps({"error": "Valid shares outstanding are required.", "ticker": ticker.upper()})

        # Validate inputs
        if (not all(math.isfinite(x) for x in (growth_rate, discount_rate, terminal_growth_rate))
                or growth_rate <= -1 or terminal_growth_rate <= -1
                or discount_rate <= 0 or discount_rate <= terminal_growth_rate):
            return json.dumps({
                "error": "Rates must be finite; growth rates must exceed -1, and discount rate must exceed zero and terminal growth.",
                "ticker": ticker.upper(),
            })

        # --- 5-Year FCF Projections ---
        projected_fcf = []
        present_values = []
        total_pv_fcf = 0.0
        for year in range(1, 6):
            fcf_year = base_fcf * ((1 + growth_rate) ** year)
            pv = fcf_year / ((1 + discount_rate) ** year)
            total_pv_fcf += pv
            projected_fcf.append(round(fcf_year / 1e6, 2))   # in millions
            present_values.append(round(pv / 1e6, 2))

        # --- Terminal Value (Gordon Growth Model) ---
        fcf_year_6     = base_fcf * ((1 + growth_rate) ** 5) * (1 + terminal_growth_rate)
        terminal_value = fcf_year_6 / (discount_rate - terminal_growth_rate)
        pv_terminal    = terminal_value / ((1 + discount_rate) ** 5)

        # --- Enterprise Value & Intrinsic Price ---
        intrinsic_ev  = total_pv_fcf + pv_terminal

        # Equity value = EV + Cash - Debt
        cash = _safe_float(info.get("totalCash"))
        debt = _safe_float(info.get("totalDebt"))
        if cash is None or debt is None or cash < 0 or debt < 0:
            return json.dumps({"error": "Finite, nonnegative cash and debt inputs are required.", "ticker": ticker.upper()})
        equity_value  = intrinsic_ev + cash - debt
        intrinsic_per_share = equity_value / shares if shares > 0 else 0

        current_price = _safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        upside = ((intrinsic_per_share - current_price) / current_price * 100) if current_price is not None and current_price > 0 else None

        result = {
            "ticker":          ticker.upper(),
            "company_name":    info.get("longName", ticker.upper()),
            "model":           "5-Year DCF + Terminal Value (Gordon Growth)",
            "limitations": "Illustrative only: provider freeCashflow is an unverified proxy for unlevered FCFF. Reconcile cash-flow definitions, reinvestment, WACC and valuation-date inputs before investment use.",
            "inputs": {
                "base_fcf_M":          round(base_fcf / 1e6, 2),
                "growth_rate_pct":     round(growth_rate * 100, 1),
                "discount_rate_pct":   round(discount_rate * 100, 1),
                "terminal_growth_pct": round(terminal_growth_rate * 100, 1),
                "shares_outstanding_M": round(shares / 1e6, 2),
            },
            "projections": {
                f"year_{i+1}": {
                    "projected_fcf_M": projected_fcf[i],
                    "present_value_M": present_values[i],
                }
                for i in range(5)
            },
            "terminal_value": {
                "terminal_value_M":     round(terminal_value / 1e6, 2),
                "pv_terminal_value_M":  round(pv_terminal / 1e6, 2),
                "tv_pct_of_total":      round(pv_terminal / intrinsic_ev * 100, 1),
            },
            "valuation": {
                "sum_pv_fcf_M":          round(total_pv_fcf / 1e6, 2),
                "intrinsic_ev_M":        round(intrinsic_ev / 1e6, 2),
                "cash_M":                round(cash / 1e6, 2),
                "debt_M":                round(debt / 1e6, 2),
                "equity_value_M":        round(equity_value / 1e6, 2),
                "intrinsic_value_per_share": round(intrinsic_per_share, 2),
                "current_price":         round(current_price, 2) if current_price is not None else None,
                "upside_downside_pct":   round(upside, 1) if upside is not None else None,
                "verdict": (
                    "INSUFFICIENT_DATA" if upside is None
                    else "SIGNIFICANTLY UNDERVALUED" if upside > 30
                    else "SLIGHTLY UNDERVALUED"  if upside > 15
                    else "FAIRLY VALUED"         if upside >= -15
                    else "SLIGHTLY OVERVALUED"   if upside >= -30
                    else "SIGNIFICANTLY OVERVALUED"
                ),
            },
        }

        return json.dumps(result, indent=2, default=str, allow_nan=False)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})


# ---------------------------------------------------------------------------
# Tool 2: Peer Multiples Comparison
# ---------------------------------------------------------------------------

@function_tool
def get_peer_multiples(ticker: str) -> str:
    """
    Compare a stock's valuation multiples against its sector peers.

    Fetches P/E, P/B, EV/EBITDA, and P/S ratios for the company and
    a set of sector peers. Returns averages, medians, and highlights
    whether the stock trades at a premium or discount to peers.

    Args:
        ticker: The stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'.
    """
    try:
        stock  = yf.Ticker(ticker.upper())
        info   = stock.info
        sector = info.get("sector")

        if not sector:
            return json.dumps({
                "error": f"Could not determine sector for {ticker.upper()} from yfinance. Cannot run peer comparison.",
                "ticker": ticker.upper(),
            })

        if sector not in SECTOR_PEERS:
            return json.dumps({
                "error": f"Sector '{sector}' is not in SECTOR_PEERS. Add it to valuation_tools.py before running peer comparison.",
                "ticker": ticker.upper(),
                "sector": sector,
                "known_sectors": list(SECTOR_PEERS.keys()),
            })

        # Resolve peer list, remove the subject ticker from peers
        raw_peers = SECTOR_PEERS[sector]
        peers = [p for p in raw_peers if p != ticker.upper()][:4]

        def get_multiples(t: str) -> dict:
            try:
                i = yf.Ticker(t).info
                return {
                    "ticker":     t,
                    "pe":         _safe_float(i.get("trailingPE")),
                    "forward_pe": _safe_float(i.get("forwardPE")),
                    "pb":         _safe_float(i.get("priceToBook")),
                    "ps":         _safe_float(i.get("priceToSalesTrailing12Months")),
                    "ev_ebitda":  _safe_float(i.get("enterpriseToEbitda")),
                    "roe_pct":    round(i.get("returnOnEquity", 0) * 100, 2) if i.get("returnOnEquity") else None,
                }
            except Exception:
                return {"ticker": t, "pe": None, "pb": None, "ps": None, "ev_ebitda": None}

        # Subject company multiples
        subject = get_multiples(ticker.upper())

        # Peer multiples
        peer_data = [get_multiples(p) for p in peers]

        # Compute peer averages (excluding None)
        def avg(key: str) -> float | None:
            vals = [p[key] for p in peer_data if p.get(key) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        peer_avg = {
            "pe":         avg("pe"),
            "forward_pe": avg("forward_pe"),
            "pb":         avg("pb"),
            "ps":         avg("ps"),
            "ev_ebitda":  avg("ev_ebitda"),
            "roe_pct":    avg("roe_pct"),
        }

        # Premium / discount vs peers
        def premium(subject_val, peer_val) -> str | None:
            if subject_val is None or peer_val is None or peer_val == 0:
                return None
            pct = round(((subject_val - peer_val) / peer_val) * 100, 1)
            return f"{'+' if pct >= 0 else ''}{pct}%"

        verdict, weighted_avg = _peer_verdict(subject, peer_avg)

        result = {
            "ticker":    ticker.upper(),
            "sector":    sector,
            "subject":   subject,
            "peers":     peer_data,
            "peer_averages": peer_avg,
            "vs_peers": {
                "pe":        premium(subject.get("pe"), peer_avg["pe"]),
                "forward_pe": premium(subject.get("forward_pe"), peer_avg["forward_pe"]),
                "pb":        premium(subject.get("pb"), peer_avg["pb"]),
                "ps":        premium(subject.get("ps"), peer_avg["ps"]),
                "ev_ebitda": premium(subject.get("ev_ebitda"), peer_avg["ev_ebitda"]),
            },
            "peer_verdict": {
                "verdict":              verdict,
                "weighted_avg_premium_pct": weighted_avg,
                "weights_used":         _PEER_WEIGHTS,
                "interpretation": (
                    f"Weighted average premium vs peers: {weighted_avg:+.1f}%. "
                    f"Verdict: {verdict}."
                ),
            },
        }

        return json.dumps(result, indent=2, default=str)

    except Exception as e:
        return json.dumps({"error": str(e), "ticker": ticker})
