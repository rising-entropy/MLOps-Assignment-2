"""Task 3: register prompts, log evaluation runs and promote the best version via the `production` alias."""
import difflib

import mlflow
import pandas as pd
from mlflow import MlflowClient

from . import config
from .analysis import pick_best, version_summary
from .prompts import PROMPTS


def setup() -> None:
    mlflow.set_tracking_uri(config.TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)


def _existing_version(template: str) -> int | None:
    """Return the registry version whose template matches exactly, so reruns do not create duplicates."""
    v = 1
    while True:
        try:
            existing = mlflow.genai.load_prompt(f"prompts:/{config.PROMPT_NAME}/{v}")
        except Exception:  # noqa: BLE001 - no such prompt/version
            return None
        if existing.template == template:
            return existing.version
        v += 1


def register_prompts() -> dict[str, int]:
    """Register v1-v3 as versions of one prompt. Returns {'v1': <mlflow version>, ...}."""
    versions = {}
    for label, p in PROMPTS.items():
        found = _existing_version(p["template"])
        if found:
            versions[label] = found
            continue
        prompt = mlflow.genai.register_prompt(
            name=config.PROMPT_NAME, template=p["template"], commit_message=p["commit"],
            tags={"label": label, "style": p["name"]},
        )
        versions[label] = prompt.version
    return versions


def write_diffs() -> list:
    paths = []
    for a, b in [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]:
        diff = difflib.unified_diff(PROMPTS[a]["template"].splitlines(), PROMPTS[b]["template"].splitlines(),
                                    fromfile=f"prompt {a}", tofile=f"prompt {b}", lineterm="")
        path = config.OUTPUT_DIR / f"prompt_diff_{a}_{b}.txt"
        path.write_text("\n".join(diff) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def log_runs(df: pd.DataFrame, versions: dict[str, int]) -> pd.DataFrame:
    """One MLflow run per prompt version with the five required metrics."""
    summary = version_summary(df)
    diffs = write_diffs()
    client = MlflowClient()
    exp_id = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME).experiment_id
    for _, row in summary.iterrows():
        label = row.prompt_version
        # Reruns replace the earlier run of the same name instead of piling up duplicates.
        for old in client.search_runs([exp_id], f"tags.mlflow.runName = 'prompt_{label}'"):
            client.delete_run(old.info.run_id)
        with mlflow.start_run(run_name=f"prompt_{label}"):
            mlflow.log_params({"prompt_name": config.PROMPT_NAME, "prompt_label": label,
                               "prompt_registry_version": versions[label], "model": config.MODEL,
                               "api_version": config.API_VERSION, "temperature": config.TEMPERATURE,
                               "input_price_per_m": config.INPUT_PRICE_PER_M,
                               "output_price_per_m": config.OUTPUT_PRICE_PER_M,
                               "pricing_date": config.PRICING_DATE, "currency": config.CURRENCY,
                               "num_questions": int(df[df.prompt_version == label].question_id.nunique()),
                               "repeats": config.REPEATS, "sla_seconds": config.SLA_SECONDS,
                               "num_requests": int((df.prompt_version == label).sum())})
            required = ["avg_latency_seconds", "avg_input_tokens", "avg_output_tokens",
                        "avg_cost_per_request", "valid_json"]
            extra = ["std_latency_seconds", "p95_latency_seconds", "sla_hit_rate", "avg_tokens_per_second",
                     "avg_answer_words", "cost_per_1k_requests", "avg_key_points"]
            mlflow.log_metrics({k: float(row[k]) for k in required})
            mlflow.log_metrics({k: float(row[k]) for k in extra if pd.notna(row[k])})
            sub = df[df.prompt_version == label]
            mlflow.log_table(sub.drop(columns=["error"]), artifact_file="responses.json")
            for d in diffs:
                mlflow.log_artifact(str(d), artifact_path="prompt_diffs")
    return summary


def promote_best(summary: pd.DataFrame, versions: dict[str, int]) -> str:
    best = pick_best(summary)
    MlflowClient().set_prompt_alias(config.PROMPT_NAME, config.PRODUCTION_ALIAS, versions[best])
    return best


def build_comparison(summary: pd.DataFrame, best: str) -> pd.DataFrame:
    out = summary.copy()  # required columns first, then the extra measures
    out["recommended"] = out.prompt_version.map(lambda v: "Yes" if v == best else "No")
    out.to_csv(config.EVAL_CSV, index=False)
    return out


def run_task3(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    setup()
    versions = register_prompts()
    summary = log_runs(df, versions)
    best = promote_best(summary, versions)
    return build_comparison(summary, best), best
