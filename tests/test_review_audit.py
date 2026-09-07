"""Offline regressions for the second audit; no provider calls or API keys."""
import ast
import asyncio
import contextlib
import io
import json
import math
import re
import subprocess
from types import SimpleNamespace
import unittest

import httpx

import test_review_controls as controls
from test_review_controls import ROOT, load_functions, load_models


def named_function(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, 'name', None) == name)
    return compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec')


class AuditRegressionTests(unittest.TestCase):
    def test_news_loop_stops_at_four_attempts(self):
        self.news_loop('unsuccessful', 4)

    def test_news_loop_stops_on_first_success(self):
        self.news_loop('successful', 1)

    def news_loop(self, score, expected):
        nb = json.loads((ROOT/'Chapter 2/chapter_2_lab_5_Financial_News_agent_evaluator_optimizer_pattern.ipynb').read_text())
        source = next(''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and 'nbr_iterations' in ''.join(c['source']))
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name=='main')
        calls = []
        async def run(agent, argument):
            calls.append(agent)
            return SimpleNamespace(to_input_list=lambda: [], new_items=[], final_output=SimpleNamespace(score=score, feedback='fixture'))
        ns = dict(input=lambda _: 'fixture', Runner=SimpleNamespace(run=run),
                  web_news_searcher='search', news_evaluator='evaluate',
                  ItemHelpers=SimpleNamespace(text_message_outputs=lambda _: 'fixture'))
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'news_loop', 'exec'), ns)
        with contextlib.redirect_stdout(io.StringIO()):
            asyncio.run(ns['main']())
        self.assertEqual(calls, ['search','evaluate'] * expected)

    def test_autogen_notebook_requests_model_extra(self):
        nb = json.loads((ROOT/'Chapter 3/chapter_3_lab_4_autogen.ipynb').read_text())
        installs = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code' and '%pip install' in ''.join(c['source'])]
        self.assertTrue(any('"autogen-ext[openai]"' in s for s in installs))

    def test_score_parser_accepts_only_integer_range(self):
        ns = {'re': re}
        exec(named_function(ROOT/'Chapter 6/chapter_6_lab_5_lats/lats_helpers.py', '_parse_score'), ns)
        for score in range(1, 11):
            self.assertEqual(ns['_parse_score'](f'SCORE: {score}'), score/10)
        for invalid in ['nan', 'inf', '-inf', '100', '0', '-1', '4.5', '8 extra', '']:
            with self.subTest(invalid=invalid):
                self.assertIsNone(ns['_parse_score'](f'SCORE: {invalid}'))
        self.assertEqual(ns['_parse_score']('SCORE: <integer 1-10>\nSCORE: 3\nSCORE: 8'), .8)
        self.assertIsNone(ns['_parse_score']('No score supplied'))

    def test_financial_models_match_notebook_and_source(self):
        a = load_models(ROOT/'Chapter 5/common.py')['CompanyMetrics']
        b = load_models(ROOT/'Chapter 5/src/common.py')['CompanyMetrics']
        self.assertEqual(a.model_json_schema(), b.model_json_schema())

    def test_missing_comparison_fields_fail(self):
        r = controls.ResearchControls(); r.setUp()
        for field in ['revenue_growth','gross_margin','operating_margin','market_cap','period','pe_ratio']:
            result = r.ns['validate_research'](r.plan(), {1:r.metrics(**{field:None})}, ['NVDA'])
            self.assertFalse(result.is_valid, field)

    def test_reported_zeros_are_not_missing(self):
        r = controls.ResearchControls(); r.setUp()
        fields = dict(revenue=0, eps=0, revenue_growth=0, gross_margin=0,
                      operating_margin=0, market_cap=0, pe_ratio=None)
        self.assertTrue(r.ns['validate_research'](r.plan(), {1:r.metrics(**fields)}, ['NVDA']).is_valid)

    def test_short_minimal_metrics_do_not_pass(self):
        r = controls.ResearchControls(); r.setUp()
        result = r.ns['validate_research'](r.plan(), {1:json.dumps(dict(ticker='NVDA',revenue=0,eps=0))}, ['NVDA'])
        self.assertFalse(result.is_valid)

    def test_notebook_validator_rejects_missing_fields(self):
        r = controls.ResearchControls(); r.setUp()
        nb = json.loads((ROOT/'Chapter 5/chapter_5_lab_2_deep_search_full_pipeline.ipynb').read_text())
        exec(''.join(nb['cells'][24]['source']), r.ns)
        self.assertFalse(r.ns['validate_research'](r.plan(), {1:r.metrics(period=None)}, ['NVDA']).is_valid)

    def test_claim_full_functions_block_missing_stages(self):
        async def run():
            c = controls.ClaimsControls()
            for ns in c.namespaces():
                for field in ['fraud_score','estimated_payout']:
                    ctx = c.context(); await ns['verify_coverage'](ctx)
                    ctx.store.state['claim'].pop(field)
                    await ns['make_decision'](ctx)
                    self.assertIn('BLOCKED', await ns['validate_compliance'](ctx))
        asyncio.run(run())

    def test_change_scope_covers_workflow_and_chapter(self):
        ns = load_functions((ROOT/'Chapter 11/ci/change_scope.py').read_text(), {})
        for paths, expected in [([b'README.md'],False),([b'Chapter 11/src/a.py'],True),
                                ([b'.github/workflows/kyc.yml'],True),([b'Chapter 110/a.py'],False)]:
            self.assertEqual(ns['relevant'](paths), expected)

    def test_required_gate_fails_closed(self):
        workflow = (ROOT/'Chapter 11/ci/github_actions.yml').read_text()
        gate = workflow.split('      - name: Check evaluation outcome',1)[1].split('        run: |\n',1)[1]
        script = '\n'.join(line[10:] for line in gate.splitlines())
        for change, relevant, score, good in [
            ('success','true','success',True), ('success','false','skipped',True),
            ('failure','false','skipped',False), ('cancelled','true','success',False),
            ('success','true','failure',False), ('success','true','cancelled',False),
            ('success','true','skipped',False), ('success','','skipped',False)]:
            result = subprocess.run(['/bin/bash','-e','-c',script], env={
                'CHANGES_RESULT':change,'RELEVANT':relevant,'SCORECARD_RESULT':score}, capture_output=True)
            self.assertEqual(result.returncode==0,good,(change,relevant,score))
        self.assertNotIn('    paths:',workflow)


class FinancialRetrievalAudit(unittest.IsolatedAsyncioTestCase):
    async def retrieve(self, income, snapshot, metrics_snapshot=None):
        r = controls.ResearchControls(); r.setUp()
        payloads = {'/financials/income-statements': {'income_statements':income},
                    '/prices/snapshot': {'snapshot':snapshot},
                    '/financial-metrics/snapshot': {'snapshot':metrics_snapshot or {}}}
        transport = httpx.MockTransport(lambda req: httpx.Response(200,json=payloads[req.url.path]))
        ns = r.ns.copy()
        ns.update(API_BASE='https://fixture.invalid', FINANCIAL_DATASETS_API_KEY='fixture', math=math,
                  httpx=SimpleNamespace(AsyncClient=lambda **kw:httpx.AsyncClient(transport=transport,**kw)))
        load_functions((ROOT/'Chapter 5/src/tools/financial_data.py').read_text(),ns)
        return await ns['get_financial_metrics']('NVDA'),r

    async def test_empty_provider_objects_do_not_become_zeros(self):
        with self.assertRaisesRegex(ValueError,'Missing required'):
            await self.retrieve([{}],{})

    async def test_missing_eps_rejected(self):
        with self.assertRaisesRegex(ValueError,'earnings_per_share'):
            await self.retrieve([{'revenue':100}],{})

    async def test_missing_optional_metrics_stay_missing_and_block(self):
        metrics,r = await self.retrieve([{'revenue':100,'earnings_per_share':2}],{})
        self.assertIsNone(metrics.period)
        self.assertIsNone(metrics.market_cap)
        self.assertIsNone(metrics.revenue_growth)
        self.assertFalse(r.ns['validate_research'](r.plan(),{1:metrics.model_dump_json()},['NVDA']).is_valid)

    async def test_diluted_zero_is_not_replaced_by_basic_eps(self):
        metrics,_ = await self.retrieve([{'revenue':0,'earnings_per_share_diluted':0,'earnings_per_share':9}],{})
        self.assertEqual(metrics.eps,0)
        self.assertIsNone(metrics.pe_ratio)

    async def test_complete_provider_data_passes(self):
        metrics,r = await self.retrieve([
            dict(revenue=100e6,earnings_per_share_diluted=2,gross_profit=50e6,operating_income=20e6,report_period='2025-12-31'),
            dict(revenue=90e6)], dict(price=20), dict(market_cap=1e9))
        self.assertEqual(metrics.period,'2025-12-31')
        self.assertEqual(metrics.gross_margin,50)
        self.assertEqual(metrics.operating_margin,20)
        self.assertEqual(metrics.market_cap,1)
        self.assertTrue(r.ns['validate_research'](r.plan(),{1:metrics.model_dump_json()},['NVDA']).is_valid)

    async def test_nonfinite_and_boolean_provider_values_rejected(self):
        for value in [float('inf'), True, 'unknown']:
            # Call the numeric helper directly: standard JSON rejects infinities.
            ns={'math':math}
            exec(named_function(ROOT/'Chapter 5/src/tools/financial_data.py','_number'),ns)
            with self.assertRaises(ValueError):ns['_number']({'revenue':value},'revenue')


if __name__ == '__main__':
    unittest.main()
