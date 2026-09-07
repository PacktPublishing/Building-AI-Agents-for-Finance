"""Offline regression tests for the book's research and claims examples.

Execute the actual function bodies with local fixtures. Framework orchestration
and provider calls are deliberately not exercised by these tests.
"""
import ast
import argparse
import asyncio
import contextlib
import copy
import importlib.util
import io
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock
import httpx

ROOT = Path(__file__).resolve().parents[1]


def load_functions(text, namespace):
    nodes = [n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reviewed_functions", "exec"), namespace)
    return namespace


def load_models(path):
    spec = importlib.util.spec_from_file_location("review_models", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return vars(mod).copy()


class ResearchControls(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.ns = load_models(ROOT / "Chapter 5/src/common.py")
        self.ns.update(math=math, asyncio=asyncio, argparse=argparse)
        load_functions((ROOT / "Chapter 5/src/validator.py").read_text(), self.ns)

    def plan(self, status="completed", dependencies=None):
        return self.ns["ResearchPlan"](question="Compare NVDA", sub_tasks=[self.ns["SubTask"](id=1, description="Get NVDA financial metrics", data_sources=["financial_api"], status=status, dependencies=dependencies or [])])

    def metrics(self, **kwargs):
        return self.ns["CompanyMetrics"](ticker="NVDA", revenue=100, eps=2, **kwargs).model_dump_json()

    def test_retrieval_error_does_not_count_as_evidence(self):
        result = self.ns["validate_research"](self.plan(), {1: "Error retrieving filing data for NVDA: service unavailable"}, ["NVDA"])
        self.assertFalse(result.is_valid)
        self.assertTrue(result.gaps)

    def test_valid_metrics_are_accepted(self):
        self.assertTrue(self.ns["validate_research"](self.plan(), {1: self.metrics()}, ["NVDA"]).is_valid)

    def test_missing_pending_or_orphan_results_fail(self):
        for plan, results in [(self.plan(), {}), (self.plan("pending"), {1: self.metrics()}), (self.plan(), {2: self.metrics()})]:
            with self.subTest(results=results):
                self.assertFalse(self.ns["validate_research"](plan, results, ["NVDA"]).is_valid)

    def test_empty_plan_fails(self):
        plan = self.ns["ResearchPlan"](question="empty", sub_tasks=[])
        self.assertFalse(self.ns["validate_research"](plan, {}, []).is_valid)

    def test_nonfinite_metrics_fail(self):
        self.assertFalse(self.ns["validate_research"](self.plan(), {1: self.metrics(pe_ratio=float("inf"))}, ["NVDA"]).is_valid)

    def test_notebook_validator_has_same_failure_behavior(self):
        nb = json.loads((ROOT / "Chapter 5/chapter_5_lab_2_deep_search_full_pipeline.ipynb").read_text())
        exec("".join(nb["cells"][24]["source"]), self.ns)
        self.test_retrieval_error_does_not_count_as_evidence()
        self.test_valid_metrics_are_accepted()
        self.test_missing_pending_or_orphan_results_fail()

    async def test_executor_marks_failure_and_blocks_dependents(self):
        load_functions((ROOT / "Chapter 5/src/researcher.py").read_text(), self.ns)
        self.ns["_route_and_execute"] = AsyncMock(side_effect=ValueError("retrieval failed"))
        plan = self.plan("pending")
        plan.sub_tasks.append(self.ns["SubTask"](id=2, description="Analyze", data_sources=[], dependencies=[1]))
        with contextlib.redirect_stdout(io.StringIO()):
            await self.ns["execute_plan"](plan)
        self.assertTrue(all(t.status == self.ns["TaskStatus"].FAILED for t in plan.sub_tasks))
        self.assertEqual(self.ns["_route_and_execute"].await_count, 1)

    async def test_executor_rejects_empty_tool_result(self):
        load_functions((ROOT / "Chapter 5/src/researcher.py").read_text(), self.ns)
        self.ns["_route_and_execute"] = AsyncMock(return_value="")
        plan = self.plan("pending")
        with contextlib.redirect_stdout(io.StringIO()):
            await self.ns["execute_plan"](plan)
        self.assertEqual(plan.sub_tasks[0].status, self.ns["TaskStatus"].FAILED)

    async def test_source_orchestrator_does_not_synthesize_failed_research(self):
        load_functions((ROOT / "Chapter 5/src/deep_search_agent.py").read_text(), self.ns)
        synth = AsyncMock()
        self.ns.update(MAX_REPLAN_ATTEMPTS=0, _setup_cost_tracking=lambda: {}, create_research_plan=AsyncMock(return_value=self.plan()), execute_plan=AsyncMock(return_value={1: "NVDA error"}), synthesize_report=synth)
        with self.assertRaisesRegex(RuntimeError, "synthesis blocked"):
            await self.ns["deep_search"]("Compare", ["NVDA"], verbose=False)
        synth.assert_not_awaited()

    async def test_notebook_orchestrator_does_not_synthesize_failed_research(self):
        nb = json.loads((ROOT / "Chapter 5/chapter_5_lab_2_deep_search_full_pipeline.ipynb").read_text())
        exec("".join(nb["cells"][30]["source"]), self.ns)
        synth = AsyncMock()
        self.ns.update(MAX_REPLAN_ATTEMPTS=0, planner_agent=SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(output=self.plan()))), format_plan=lambda _: "plan", execute_plan=AsyncMock(return_value={1: "NVDA error"}), synthesize_report=synth)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "synthesis blocked"):
            await self.ns["deep_search"]("Compare", ["NVDA"])
        synth.assert_not_awaited()


class Store:
    def __init__(self, state):
        self.state = state

    async def get(self, key):
        return self.state

    async def set(self, key, value):
        self.state = value


class ClaimsControls(unittest.IsolatedAsyncioTestCase):
    def namespaces(self):
        ns = {"Context": object, "log_audit": lambda *args: None, "ClaimType": object}
        source = load_functions((ROOT / "Chapter 9/src/claims_pipeline/tools.py").read_text(), ns)
        nbns = {"Context": object, "log_audit": lambda *args: None}
        nb = json.loads((ROOT / "Chapter 9/chapter_9_lab_1_claims_pipeline.ipynb").read_text())
        for i in (14, 20, 22):
            exec("".join(nb["cells"][i]["source"]), nbns)
        return [source, nbns]

    def context(self, **claim_updates):
        claim = {"date_of_loss": "2026-06-01", "description": "Vehicle collision", "fraud_score": 0, "fraud_recommendation": "proceed", "estimated_payout": 100}
        claim.update(claim_updates)
        return SimpleNamespace(store=Store({"claim": claim, "policy": {"inception_date": "2026-01-01", "expiration_date": "2026-12-31", "exclusions": ["racing"], "coverage_sections": "collision", "coverage_limit": 1000, "deductible": 0}}))

    async def test_covered_claim_passes(self):
        for ns in self.namespaces():
            ctx = self.context()
            await ns["verify_coverage"](ctx)
            await ns["make_decision"](ctx)
            self.assertEqual(ctx.store.state["claim"]["decision"], "APPROVED")
            self.assertNotIn("BLOCKED", await ns["validate_compliance"](ctx))

    async def test_evidenced_denials_pass(self):
        for ns in self.namespaces():
            for fields, code in [({"date_of_loss": "2025-01-01"}, "OUTSIDE_POLICY_PERIOD"), ({"description": "Vehicle damaged while racing"}, "POLICY_EXCLUSION")]:
                ctx = self.context(**fields)
                await ns["verify_coverage"](ctx)
                await ns["make_decision"](ctx)
                self.assertEqual(ctx.store.state["claim"]["decision_reason_codes"], [code])
                self.assertNotIn("BLOCKED", await ns["validate_compliance"](ctx))

    async def test_empty_exclusions_cannot_justify_arbitrary_denial(self):
        for ns in self.namespaces():
            ctx = self.context(coverage_verified=False, decision="DENIED", decision_reasoning="Claim denied because customer was impolite. Exclusions: [].", decision_reason_codes=["POLICY_EXCLUSION"])
            self.assertIn("BLOCKED", await ns["validate_compliance"](ctx))
            await ns["make_decision"](ctx)
            self.assertEqual(ctx.store.state["claim"]["decision"], "ESCALATED")

    async def test_missing_policy_or_bad_dates_escalate(self):
        for ns in self.namespaces():
            for date in ("", "not-a-date"):
                ctx = self.context(date_of_loss=date)
                await ns["verify_coverage"](ctx)
                await ns["make_decision"](ctx)
                self.assertEqual(ctx.store.state["claim"]["decision"], "ESCALATED")

    async def test_unknown_reason_code_is_blocked(self):
        for ns in self.namespaces():
            ctx = self.context(date_of_loss="2025-01-01")
            await ns["verify_coverage"](ctx)
            await ns["make_decision"](ctx)
            ctx.store.state["claim"]["decision_reason_codes"] = ["CUSTOMER_IMPOLITE"]
            self.assertIn("BLOCKED", await ns["validate_compliance"](ctx))

    async def test_malformed_evidence_escalates(self):
        for ns in self.namespaces():
            for field, value in [("exclusions", True), ("exclusions", None), ("inception_date", 42)]:
                ctx = self.context()
                ctx.store.state["policy"][field] = value
                await ns["verify_coverage"](ctx)
                await ns["make_decision"](ctx)
                self.assertEqual(ctx.store.state["claim"]["decision"], "ESCALATED")

    async def test_fraud_referral_cannot_be_overridden_by_approval(self):
        for ns in self.namespaces():
            ctx = self.context(fraud_recommendation="investigate")
            await ns["verify_coverage"](ctx)
            await ns["make_decision"](ctx)
            self.assertEqual(ctx.store.state["claim"]["decision"], "ESCALATED")
            ctx.store.state["claim"]["decision"] = "APPROVED"
            self.assertIn("BLOCKED", await ns["validate_compliance"](ctx))


class RetrievalControls(unittest.IsolatedAsyncioTestCase):
    def namespace(self, filename, payload, status=200):
        transport = httpx.MockTransport(lambda req: httpx.Response(status, json=payload))
        ns = {"HEADERS": {}, "TAVILY_API_KEY": "test-only", "FINANCIAL_DOMAINS": []}
        ns["httpx"] = SimpleNamespace(AsyncClient=lambda **kw: httpx.AsyncClient(transport=transport, **kw), HTTPStatusError=httpx.HTTPStatusError)
        return load_functions((ROOT / "Chapter 5/src/tools" / filename).read_text(), ns)

    async def test_sec_empty_missing_excerpt_and_http_error_raise(self):
        for payload, status in [({"hits": {"hits": []}}, 200), ({"hits": {"hits": [{"_source": {}}]}}, 200), ({}, 429)]:
            ns = self.namespace("sec_filings.py", payload, status)
            with self.assertRaises(ValueError):
                await ns["get_risk_factors"]("NVDA")

    async def test_sec_excerpt_is_returned(self):
        ns = self.namespace("sec_filings.py", {"hits": {"hits": [{"_source": {"file_date": "2026-01-01"}, "highlight": {"text": ["Material supply-chain risk."]}}]}})
        self.assertIn("Material supply-chain risk", await ns["get_risk_factors"]("NVDA"))

    async def test_web_empty_malformed_and_http_error_raise(self):
        for payload, status in [({"results": []}, 200), ({"results": [{"content": "", "url": "https://example.com"}]}, 200), ({}, 500)]:
            for function in ("search_general", "search_financial_news"):
                ns = self.namespace("web_search.py", payload, status)
                with self.assertRaises(ValueError):
                    await ns[function]("NVDA news")

    async def test_web_evidence_is_returned(self):
        for function in ("search_general", "search_financial_news"):
            ns = self.namespace("web_search.py", {"results": [{"title": "Fixture", "content": "A sourced excerpt.", "url": "https://example.com/fixture"}]})
            self.assertIn("A sourced excerpt", await ns[function]("NVDA news"))


class DCFControls(unittest.TestCase):
    def setUp(self):
        self.info = {"freeCashflow": 100_000_000, "sharesOutstanding": 100_000_000, "totalCash": 20_000_000, "totalDebt": 10_000_000, "currentPrice": 10}
        self.ns = {"json": json, "math": math, "function_tool": lambda f: f, "yf": SimpleNamespace(Ticker=lambda _: SimpleNamespace(info=self.info))}
        load_functions((ROOT / "Chapter 4/chapter_4_lab_fundamental_analysis_agent/tools/valuation_tools.py").read_text(), self.ns)

    def value(self, growth=.05, discount=.10, terminal=.02):
        return json.loads(self.ns["run_dcf_model"]("TEST", growth, discount, terminal))

    def test_discounting_and_equity_bridge(self):
        expected = sum(100e6 * 1.05**t / 1.10**t for t in range(1, 6))
        expected += 100e6 * 1.05**5 * 1.02 / (.10-.02) / 1.10**5
        expected = (expected + 20e6 - 10e6) / 100e6
        self.assertAlmostEqual(self.value()["valuation"]["intrinsic_value_per_share"], round(expected, 2))

    def test_missing_price_is_not_overvaluation(self):
        self.info.pop("currentPrice")
        value = self.value()["valuation"]
        self.assertIsNone(value["upside_downside_pct"])
        self.assertEqual(value["verdict"], "INSUFFICIENT_DATA")

    def test_zero_upside_is_fairly_valued(self):
        # With flat FCF and zero terminal growth, EV = FCF / discount.
        self.info.update(totalCash=0, totalDebt=0, currentPrice=10)
        self.assertEqual(self.value(0, .1, 0)["valuation"]["verdict"], "FAIRLY VALUED")

    def test_operating_cash_flow_is_not_a_fcf_fallback(self):
        self.info.pop("freeCashflow")
        self.info["operatingCashflow"] = 200_000_000
        self.assertIn("error", self.value())

    def test_missing_or_nonfinite_inputs_fail(self):
        for field in ("freeCashflow", "sharesOutstanding", "totalCash", "totalDebt"):
            previous = self.info[field]
            for invalid in (None, float("nan"), float("inf")):
                self.info[field] = invalid
                self.assertIn("error", self.value())
            self.info[field] = previous
        for rates in [(0.1, .02, .02), (-1, .1, .02), (.1, .1, -1), (.1, float("nan"), .02)]:
            self.assertIn("error", self.value(*rates))


class NotebookAnnotations(unittest.TestCase):
    def test_fundamentals_annotations_use_key_and_value_types(self):
        count = 0
        for name in ("chapter_2_lab_1_retrieving_fundamental_ratios_Apple.ipynb", "chapter_2_lab_6_investment_recommendation_sequential_pattern.ipynb"):
            nb = json.loads((ROOT / "Chapter 2" / name).read_text())
            for cell in nb["cells"]:
                text = "".join(cell["source"])
                if "def get_company_fundamentals" in text:
                    self.assertNotIn("dict[str:", text)
                    self.assertIn("dict[str, pd.DataFrame]", text)
                    count += 1
        self.assertEqual(count, 3)


if __name__ == "__main__":
    unittest.main()
