# Building AI Agents for Finance

Code repository for *Building AI Agents for Finance* (Packt, 2026).

Most chapter folders contain one or more hands-on Jupyter notebooks
named `chapter_<N>_lab_<M>_<topic>.ipynb`. Several chapters also ship
a `common.py` with shared models, helpers, and sample data that the
notebooks import; the notebooks auto-download `common.py` on first
run in Colab. Chapter 11 is the exception: its evaluation harness is a
pip-installable Python package run from the command line, with its own
README.

## Running the labs

### In Google Colab (recommended)

Each notebook is designed to run end-to-end in Colab. Open the
notebook, and when prompted for an API key, add it via the **key**
icon in Colab's left sidebar using the exact name listed in the
notebook (e.g. `OPENAI_API_KEY`). The notebooks read keys with
`google.colab.userdata.get(...)`.

### Locally

1. Copy [`.env.sample`](./.env.sample) to `.env` and fill in the keys
   you need (each section lists which labs use which key).
2. Install the per-lab packages with the `%pip install` cell at the
   top of each notebook.
3. Open the notebook in Jupyter / VS Code / your editor of choice.

The notebooks wrap the Colab import in `try / except ImportError` and
fall back to reading from the environment when running locally.

## Repository layout

### Chapter 3 setup and example types

Use a separate environment for each framework, especially CrewAI and Google
ADK. Python 3.12 was used for the Lab 7/8 checks recorded below; check each
package's Python constraints before using a different interpreter version.
Install the packages from the chosen notebook's installation cell. The
pins in `Chapter 3/bakeoff/requirements.txt` belong to the separate bake-off
scripts, not to every notebook.

| Example | Main packages (plus the notebook's supporting dependencies) | Credential |
| --- | --- | --- |
| Lab 1: LangChain | `langchain`, `langchain-openai` | `OPENAI_API_KEY` |
| Lab 2: Google ADK | `google-adk`, `google-genai` | `GOOGLE_API_KEY` |
| Lab 3: CrewAI | `crewai[tools]` | `OPENAI_API_KEY` |
| Lab 4: AutoGen | `autogen-agentchat`, `autogen-ext`, `autogen-core` | `OPENAI_API_KEY` |
| Lab 5: OpenAI Agents SDK | `openai-agents` | `OPENAI_API_KEY` |
| Lab 6: LlamaIndex | `llama-index`, `llama-index-llms-openai` | `OPENAI_API_KEY` |
| Lab 7: Anthropic API, no tools | `anthropic` | `ANTHROPIC_API_KEY` |
| Lab 8: Mistral, manual tool loop | `mistralai` | `MISTRAL_API_KEY` |
| Lab 9: PydanticAI | `pydantic-ai` | `OPENAI_API_KEY` |
| Bake-off Claude Agent SDK example | `claude-agent-sdk` | `ANTHROPIC_API_KEY` |

The notebooks also install `pydantic` and `python-dotenv`. Export the required
credential in the notebook process, or use Colab Secrets. Labs 7 and 8 now
explicitly load a local `.env` file; installing `python-dotenv` alone does not
load one. Never commit credentials.

The notebooks and bake-off scripts solve the same P/E comparison task but
are not interchangeable implementations. Lab 7 calls the Anthropic Messages
API directly, without tools or an agent loop. The actual Claude Agent SDK
workflow is `Chapter 3/bakeoff/claude_agent_sdk_agent.py`. Lab 8 uses Mistral's
API-client SDK and a manual tool loop, not a separate agent framework.

Check the model selected in each example. The bake-off's `common.py` controls
the OpenAI-backed scripts only; provider-specific scripts and notebooks have
their own settings. The Claude Agent SDK script currently leaves model
selection to the SDK. Record the actual model before comparing outputs.
Lab 7 defaults to `claude-sonnet-5`, with thinking explicitly disabled; set
`CHAPTER3_ANTHROPIC_MODEL` only to a model compatible with the shown request.
Mock finance data avoids a market-data subscription, but model calls still
require network access and available provider credits.

#### Chapter 3 correction checks (7 September 2026)

- Python 3.12.14; `anthropic==1.4.0`, `mistralai==2.9.4`,
  `pydantic==2.13.5`, `python-dotenv==1.2.3`; notebook validation with
  `nbformat==5.11.1`. Lab 7/8 installation cells pin these application packages.
- Fresh installation and `pip check` passed. Six offline regression tests
  passed, including actual SDK request/response handling with mocked HTTP,
  Lab 7 truncation/empty-response rejection, Lab 8 tool dispatch, notebook
  validation and the scoring-script reference. Run
  `python -m unittest discover -s tests -v` after installing the packages above.
- Live validation is still pending: a provider-account prerequisite prevented
  the Lab 7 run, and no test credential was available for Lab 8. Mocked responses do not establish model
  accuracy, provider availability or successful end-to-end live execution.
- These checks cover the changed Labs 7/8 and scoring reference, not all
  Chapter 3 examples or all bake-off dependency combinations.

#### Cross-chapter correction checks (7 September 2026)

The correction branch also aligns Chapter 2 notebook annotations, Chapter 4
DCF inputs and verdicts, Chapter 5 failed-research handling, Chapter 9 claims
evidence checks and referral terminology, and Chapter 11 CI setup wording.
The source and notebook versions of the affected Chapter 5/9 controls are
covered by offline fixture tests. There are 33 passing tests in total; run
`python -m unittest discover -s tests -v` with the packages listed above and
`httpx==0.28.1`. These tests execute the relevant function bodies and mocked
HTTP calls, not the complete framework workflows or live providers.

The DCF tool is educational: provider `freeCashflow` is not verified unlevered
FCFF. It no longer substitutes operating cash flow, invents missing share
counts, or treats a missing price as an overvaluation verdict. Match the cash
flow definition to the discount rate before real valuation use; see
[Damodaran's valuation framework](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/val.html).

Corrected, editable figure sources are included for
[Figure 5.11](./Chapter%205/src/diagrams/svg/figure_5_11_corrected.svg) and
[Figure 9.4](./Chapter%209/figures/figure_9_4_referral_workflow.svg).
The Chapter 9 checks are simplified evidence controls, not legal compliance
certification or a finding of fraud. Live end-to-end verification remains
pending; a passing fixture suite does not establish production readiness.

### Folder guide

```text
Chapter 1/        First-principles labs (Hanane)
Chapter 2/        Memory and tool patterns (Hanane)
Chapter 3/        Framework comparison — one lab notebook per framework,
                  plus bakeoff/ (the chapter's printed implementations with
                  captured outputs and pinned requirements), templates/
                  (decision-matrix spreadsheet and scoring script), and
                  diagrams/ (figure sources)
Chapter 5/        Deep search agent (planner + full pipeline)
Chapter 7/        Multi-agent systems and architectural styles
Chapter 9/        Insurance workflows (claims, fraud, underwriting)
Chapter 11/       KYC evaluation harness — a pip-installable package run
                  from the CLI rather than notebooks; see its own README
.env.sample       Template for local-execution API keys
```

## Provider keys at a glance

| Key | Required by |
| --- | --- |
| `OPENAI_API_KEY` | Ch 1, 2, 3, 5, 7, 9 (most labs); optional for Ch 11's LLM judge |
| `ANTHROPIC_API_KEY` | Ch 1 Lab 1, Ch 3 Lab 7; optional for Ch 11's LLM judge |
| `GOOGLE_API_KEY` | Ch 1 Lab 1, Ch 3 Lab 2 |
| `MISTRAL_API_KEY` | Ch 3 Lab 8 |
| `NEWS_API_KEY` | Ch 1 Lab 3 |
| `FINANCIAL_MODELING_PREP_API_KEY` | Ch 2 Labs 1 and 6 |
| `FINANCIAL_DATASETS_API_KEY`, `TAVILY_API_KEY`, `SEC_EDGAR_EMAIL` | Only if you swap Ch 5's stub tools for the real ones |

See [`.env.sample`](./.env.sample) for sources and per-key notes.
