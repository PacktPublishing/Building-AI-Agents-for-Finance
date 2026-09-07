# Chapter 11 — Evaluating AI Agents in Finance

Companion code for Chapter 11: a complete evaluation harness for the
KYC onboarding agent the chapter specifies as its system under test.
The harness includes a CI workflow template, a production-trace integration
stub, judge-calibration utilities, drift detection, and model risk reporting.
CI and production trace capture require explicit setup. A deterministic mock of the KYC pipeline
ships with the harness, so everything runs end to end out of the box;
an adapter hook lets you plug in a real implementation.

## Layout

```
Chapter 11/
├── README.md                     # This file
├── pyproject.toml                # Python package definition
├── src/
│   ├── models.py                 # Pydantic schemas (KYCCase, Scorecard, ...)
│   ├── common.py                 # Shared config and paths
│   ├── kyc_pipeline_adapter.py   # Adapter for Ch 9 pipeline (with mock fallback)
│   ├── eval_harness/
│   │   ├── reference_metrics.py  # Risk-tier accuracy + sanctions recall
│   │   ├── trajectory_metrics.py # Required-tool coverage
│   │   ├── llm_judge_metrics.py  # Reasoning grounding (calibrated judge)
│   │   ├── safety_metrics.py     # PII redaction + injection resistance
│   │   ├── disparate_impact.py   # Four-fifths rule + chi-square
│   │   ├── efficiency_metrics.py # Latency, cost, tokens
│   │   ├── scorecard.py          # Composer (weights, thresholds, severity)
│   │   ├── scorecard.yaml        # Production scorecard configuration
│   │   └── run.py                # CLI entry point
│   ├── calibration/
│   │   └── judge_calibration.py  # Cohen's kappa workflow for LLM-judges
│   ├── continuous/
│   │   ├── drift_detector.py     # KS + PSI + Mann-Whitney
│   │   └── trace_sampler.py      # Production trace sampling (Langfuse)
│   └── reporting/
│       ├── audit_log.py          # Append-only evaluation audit log
│       └── model_risk_report.py  # Model risk committee report generator
├── data/
│   └── kyc_eval/                 # Sample evaluation dataset (5 segments)
└── ci/
    └── github_actions.yml        # CI integration
```

## Quickstart

```bash
cd "Chapter 11"
pip install -e .
python -m src.eval_harness.run --segment all --output scorecard.json
```

The harness will load every case from `data/kyc_eval/`, run the KYC
agent on each (using the bundled deterministic mock unless a real
pipeline is installed — see below), compute every metric in
`scorecard.yaml`, and write a JSON scorecard plus a Markdown summary.
With the mock, the full run ends `ship_decision=SHIP` with every
catastrophic gate passing, so a green baseline is reproducible before
you wire in a real agent.

The exit code is `0` if every catastrophic-severity metric passes its
threshold and the weighted score is `>= 0.85`; nonzero otherwise. CI
can use this as a production-promotion gate after configuration.

To activate CI, copy `ci/github_actions.yml` into the repository-root
`.github/workflows/` directory. Its triggers cover relevant Chapter 11 changes.
Configure its evaluation status check as required in branch protection or
repository rules if failed evaluations must prevent merging. Merely including
the template in `ci/` does not activate GitHub Actions or block merges.

## With a real KYC pipeline

If you have implemented the chapter's KYC pipeline (for example, by
following the sequential multi-agent pattern from Chapters 7 and 9)
and installed it as a Python package named `chapter09.kyc_pipeline`,
`src/kyc_pipeline_adapter.py` will import it automatically. Otherwise
the adapter falls back to a deterministic mock that exercises every
metric in the harness.

## Calibrating the LLM-judge

```bash
python -c "
from src.calibration.judge_calibration import (
    load_calibration_set, inter_rater_reliability)
items = load_calibration_set()
print('inter-rater kappa:', inter_rater_reliability(items))
"
```

Recommended workflow:
1. Confirm human inter-rater kappa is `>= 0.7` (otherwise refine the rubric)
2. Run the LLM-judge on the calibration cases
3. Compute judge-vs-human kappa
4. Iterate the rubric in `src/eval_harness/llm_judge_metrics.py:GROUNDING_RUBRIC`
   until kappa hits 0.7
5. Freeze the rubric, judge model, and calibration set as a versioned
   regulatory artifact

## Drift detection

```python
import numpy as np
from src.continuous.drift_detector import detect_drift

baseline = np.array([0.92, 0.88, 0.94, 0.90, 0.91, ...])  # offline judge scores
current  = np.array([0.81, 0.79, 0.83, 0.78, 0.80, ...])  # production sample

result = detect_drift(baseline, current)
print(result.explain())
```

## Generating the model risk report

```bash
python -m src.eval_harness.run --output scorecard.json
python -m src.reporting.model_risk_report scorecard.json --out report.md
pandoc report.md -o model_risk_report.pdf
```

## Notes

- **Mock vs real KYC agent.** The system under test is the KYC
  onboarding agent specified in the chapter itself. Chapter 9 builds
  insurance workflows (claims processing, fraud investigation,
  underwriting), not KYC, so no ready-made implementation exists in
  the book's code — building one from the chapter's specification,
  following the Chapter 7/9 pipeline patterns, is the natural exercise.
  Until then the adapter uses the deterministic mock; swap the import
  in `kyc_pipeline_adapter.py` to evaluate a real implementation.
- **Sample dataset size.** Each segment ships with 5–6 representative
  cases — enough to exercise the harness end-to-end, not enough for a
  production scorecard. The chapter’s The KYC evaluation dataset
  structure section describes the ~2,050-case full dataset
  construction.
- **API keys.** Without `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` set,
  the LLM-judge metric falls back to a heuristic that counts how many
  expected tool names are mentioned in the reasoning chain. Useful for
  smoke-testing the harness; not a substitute for a calibrated judge.
