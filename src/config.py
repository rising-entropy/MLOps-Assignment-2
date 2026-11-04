"""Central configuration. All secrets and model/pricing details come from environment variables."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # python-dotenv is optional
    pass

ROOT = Path(__file__).resolve().parents[1]

# DRY_RUN=1 uses a synthetic client so the pipeline can be tested without an API key.
# Dry-run results go to a separate folder and must never be submitted.
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
OUTPUT_DIR = ROOT / ("outputs_dryrun" if DRY_RUN else "outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Any OpenAI-compatible endpoint works (OpenAI, Groq, Gemini OpenAI endpoint, Ollama, ...).
API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL") or None
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_VERSION = os.getenv("LLM_API_VERSION", "OpenAI Chat Completions API (openai-python SDK)")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
MAX_OUTPUT_TOKENS = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "400"))

# Published pricing, per 1M tokens. Verify against the provider's pricing page and set the date.
INPUT_PRICE_PER_M = float(os.getenv("INPUT_PRICE_PER_M", "0.15"))
OUTPUT_PRICE_PER_M = float(os.getenv("OUTPUT_PRICE_PER_M", "0.60"))
PRICING_DATE = os.getenv("PRICING_DATE", "2026-11-01")
CURRENCY = os.getenv("PRICING_CURRENCY", "USD")
MONTHLY_REQUESTS = 10_000

# Each (version, question) pair is sent REPEATS times, interleaved, to measure latency noise.
REPEATS = int(os.getenv("REPEATS", "3"))

# Business and scale ASSUMPTIONS (illustrative, not measured): change them to fit your own scenario.
SLA_SECONDS = 2.0                       # interactive-chat latency target
SCALE_VOLUMES = {"Baseline (brief)": 10_000, "Exam season": 100_000, "Campus-wide": 1_000_000}
PEAK_RPS = 5.0                          # requests per second at the busiest moment
PRICE_MULTIPLIERS = [0.5, 1, 2, 4]      # what-if: model price relative to today's

# MLflow
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
if DRY_RUN:
    TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow_dryrun.db').as_posix()}"
EXPERIMENT_NAME = "eduassist_llmops_assignment"
PROMPT_NAME = "eduassist-qa-prompt"
PRODUCTION_ALIAS = "production"

RAW_CALLS_CSV = OUTPUT_DIR / "latency_tokens_cost.csv"
EVAL_CSV = OUTPUT_DIR / "prompt_evaluation_results.csv"
