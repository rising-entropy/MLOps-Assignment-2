# LLMOps Assignment 2: API Cost, Latency, and Prompt Version Management

EduAssist AI scenario: measure latency, compute token cost, and version/evaluate/promote three prompts in MLflow.

## Environment
- Python 3.10 (tested with 3.10.11), Windows 11
- Libraries: see `requirements.txt` (mlflow, openai, pandas, matplotlib, reportlab, python-dotenv, nbformat, nbconvert, ipykernel, pytest)
- Model / API / pricing: set in `.env` (default `gpt-4o-mini`, OpenAI Chat Completions API, USD per 1M tokens). Record the pricing date you verified.

## Environment variables
| Variable | Purpose |
|---|---|
| `LLM_API_KEY` (or `OPENAI_API_KEY`) | API key. Never commit it. |
| `LLM_BASE_URL` | Optional, for OpenAI-compatible providers (Groq, Gemini, Ollama) |
| `LLM_MODEL` | Model name |
| `INPUT_PRICE_PER_M`, `OUTPUT_PRICE_PER_M`, `PRICING_DATE`, `PRICING_CURRENCY` | Published pricing used for cost |

Copy `.env.example` to `.env` and fill it in (`.env` is git-ignored).

## Install (PowerShell)
```powershell
cd MLOps\llmops_assignment2
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce
```powershell
python -m pytest tests            # unit tests (no API needed)
python run_pipeline.py collect    # 45 API calls -> outputs/latency_tokens_cost.csv
python run_pipeline.py mlflow     # register prompts, log runs, set `production` alias -> outputs/prompt_evaluation_results.csv
jupyter nbconvert --to notebook --execute --inplace assignment.ipynb
jupyter nbconvert --to html --embed-images assignment.ipynb
```
Or run everything: `python run_pipeline.py all`, then `python run_pipeline.py report` after adding screenshots.

## MLflow UI
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000 and capture into `screenshots/`:
1. `mlflow_experiment.jpg` - experiment `eduassist_llmops_assignment` with 3 runs and metrics
2. `prompt_registry.jpg` - Prompts tab, `eduassist-qa-prompt` with v1, v2, v3
3. `prompt_diff.jpg` - compare v2 vs v3 in the prompt versions view
4. `production_alias.jpg` - version carrying the `production` alias
5. `metrics.jpg` - run metrics / chart comparison

Then run `python run_pipeline.py report` to build `outputs/final_report.pdf`.

## Design choices
- **Questions:** five questions from the MTech modules (definition, debugging, MLOps scenario, architecture, LLMOps economics), chosen so answer lengths differ like real support traffic. Edit `QUESTIONS` in `src/prompts.py` to change them.
- **Repeats:** every version x question is sent `REPEATS` times (default 3, set with the `REPEATS` env var), interleaved, giving 45 requests. This measures latency noise and adds p95 and standard deviation.
- **Extra measures:** tokens per second, cost per 1,000 answers, answer length, key-point count, and a business and scale view (volumes, latency-target hit rate, concurrency at peak, price sensitivity). Scale inputs are labelled assumptions in `src/config.py`.
- **Latency depends on provider load.** Re-measure at a quieter or busier time with: `REFRESH=1` in the notebook environment (or `python run_pipeline.py collect`), then `python run_pipeline.py mlflow`, re-execute the notebook (its text quotes the numbers from the earlier run, so update it by hand), retake the two MLflow screenshots that show metrics, and `python run_pipeline.py report`. Tokens, cost and JSON validity are stable across runs; latency is not.

## Offline pipeline test
`DRY_RUN=1` uses a fake client and writes to `outputs_dryrun/` and `mlflow_dryrun.db`. It only checks the plumbing; **never submit dry-run results**.

## Submission ZIP checklist
`assignment.ipynb` (+ exported HTML/PDF), `requirements.txt`, `outputs/latency_tokens_cost.csv`, `outputs/prompt_evaluation_results.csv`, MLflow screenshots, `outputs/final_report.pdf`, `README.md`. Do **not** include `.env`, `.venv`, or any key.

## Layout
```
src/config.py       env-based config and pricing
src/prompts.py      questions, v1-v3 prompts, valid_json check
src/llm_client.py   API call with latency + token usage
src/analysis.py     Task 1 and 2 logic, call collection
src/registry.py     Task 3 MLflow registration, runs, alias promotion
src/report.py       PDF report generator
```

