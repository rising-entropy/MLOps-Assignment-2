# LLMOps Assignment 2: API Cost, Latency, and Prompt Version Management

EduAssist AI scenario: measure latency, compute token cost, and version, evaluate and promote three prompts in MLflow.
Everything runs from one notebook, `assignment.ipynb`.

## Environment
- Python 3.10 (tested with 3.10.11), Windows 11
- Libraries: see `requirements.txt` (mlflow, openai, pandas, matplotlib, python-dotenv, nbformat, nbconvert, ipykernel)
- Model, API and pricing are set in `.env`; the notebook prints them at the top. Record the pricing date you verified.

## Environment variables
| Variable | Purpose |
|---|---|
| `LLM_API_KEY` (or `OPENAI_API_KEY`) | API key. Never commit it. |
| `LLM_BASE_URL` | Optional, for OpenAI-compatible providers (Groq, Gemini, Ollama) |
| `LLM_MODEL`, `LLM_API_VERSION` | Model name and API version, recorded in the notebook |
| `INPUT_PRICE_PER_M`, `OUTPUT_PRICE_PER_M`, `PRICING_DATE` | Published pricing used for cost (USD per 1M tokens) |

Copy `.env.example` to `.env` and fill it in (`.env` is git-ignored).

## Install (PowerShell)
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce
```powershell
jupyter nbconvert --to notebook --execute --inplace assignment.ipynb
jupyter nbconvert --to html --embed-images assignment.ipynb
```
The notebook makes the 45 API calls (3 repeats x 3 versions x 5 questions), writes `outputs/latency_tokens_cost.csv`, registers the prompts, logs the MLflow runs, promotes the best version with the `production` alias and writes `outputs/prompt_evaluation_results.csv` and the prompt diffs.

## MLflow UI
```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open http://127.0.0.1:5000. The screenshots in `mlflow_screens/` come from it:
1. `listing_page.png` and `eval_list_page.png` - experiment `eduassist_llmops_assignment` with its 3 runs and metrics
2. `v3_prompt_overview.png` - run metrics for `prompt_v3`
3. `qa-prompt.png` - Prompts tab, `eduassist-qa-prompt` with v1, v2, v3
4. `compare_prompts.png` - prompt diff, v3 against v2
5. `production_alias.png` - version 3 carrying the `production` alias

## Notes
- Latency depends on provider load, so figures change between runs. Tokens, cost and JSON validity are stable.
- The text in the notebook quotes the numbers from the run saved with it. After a fresh run, update that text by hand.

## Submission ZIP checklist
`assignment.ipynb` (+ exported HTML), `requirements.txt`, `outputs/latency_tokens_cost.csv`, `outputs/prompt_evaluation_results.csv`, MLflow screenshots, your report PDF, `README.md`. Do **not** include `.env`, `.venv`, or any key.
