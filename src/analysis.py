"""Task 1 (latency), Task 2 (tokens and cost), the business/scale view and the call-collection loop."""
import json

import numpy as np
import pandas as pd

from . import config
from .llm_client import call_llm
from .prompts import PROMPTS, QUESTIONS, is_valid_json, render


def input_cost(tokens: float) -> float:
    return tokens / 1_000_000 * config.INPUT_PRICE_PER_M


def output_cost(tokens: float) -> float:
    return tokens / 1_000_000 * config.OUTPUT_PRICE_PER_M


def _key_points(text: str):
    try:
        return len(json.loads(text)["key_points"])
    except (TypeError, ValueError, KeyError):
        return np.nan


def collect_calls() -> pd.DataFrame:
    """Send every version x question REPEATS times and save the request log.
    Repeats are interleaved (repeat -> version -> question) so slow periods hit all versions alike."""
    rows = []
    for rep in range(1, config.REPEATS + 1):
        for version, p in PROMPTS.items():
            for qid, question in QUESTIONS.items():
                r = call_llm(render(p["template"], question))
                ic, oc = input_cost(r["input_tokens"]), output_cost(r["output_tokens"])
                rows.append(dict(
                    repeat=rep, question_id=qid, prompt_version=version, model=r["model"],
                    start_time=r["start_time"], end_time=r["end_time"],
                    latency_seconds=round(r["latency_seconds"], 4), status=r["status"],
                    input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
                    total_tokens=r["total_tokens"], input_cost=ic, output_cost=oc, total_cost=ic + oc,
                    tokens_per_second=r["output_tokens"] / r["latency_seconds"] if r["latency_seconds"] else 0.0,
                    answer_words=len(r["response_text"].split()),
                    key_points=_key_points(r["response_text"]),
                    valid_json=is_valid_json(r["response_text"]), response_text=r["response_text"],
                    error=r["error"],
                ))
                print(f"rep{rep} {version} {qid}: {r['status']} {r['latency_seconds']:.2f}s "
                      f"in={r['input_tokens']} out={r['output_tokens']}")
    df = pd.DataFrame(rows)
    df.to_csv(config.RAW_CALLS_CSV, index=False)
    return df


def load_calls() -> pd.DataFrame:
    return pd.read_csv(config.RAW_CALLS_CSV)


# ---- Task 1 ---------------------------------------------------------------
def latency_table(df: pd.DataFrame, version: str = "v1", repeat: int = 1) -> pd.DataFrame:
    cols = ["question_id", "prompt_version", "model", "start_time", "end_time", "latency_seconds", "status"]
    return df[(df.prompt_version == version) & (df.repeat == repeat)][cols].reset_index(drop=True)


def latency_summary(series: pd.Series) -> pd.Series:
    return pd.Series({"mean": series.mean(), "median": series.median(), "min": series.min(),
                      "max": series.max(), "std": series.std(ddof=1)}, name="latency_seconds")


def slowest_request(df: pd.DataFrame) -> pd.Series:
    return df.loc[df.latency_seconds.idxmax()]


def p95(series: pd.Series) -> float:
    return float(np.percentile(series, 95))


# ---- Task 2 ---------------------------------------------------------------
def cost_table(df: pd.DataFrame, version: str | None = None, repeat: int = 1) -> pd.DataFrame:
    d = df[df.repeat == repeat]
    d = d if version is None else d[d.prompt_version == version]
    return d[["question_id", "prompt_version", "input_tokens", "output_tokens", "total_tokens",
              "input_cost", "output_cost", "total_cost"]].reset_index(drop=True)


def monthly_projection(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("prompt_version").agg(avg_cost_per_request=("total_cost", "mean"))
    g["projected_monthly_cost"] = g.avg_cost_per_request * config.MONTHLY_REQUESTS
    return g.reset_index()


# ---- Task 3 summary and business view --------------------------------------
def version_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("prompt_version").agg(
        avg_latency_seconds=("latency_seconds", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_cost_per_request=("total_cost", "mean"),
        valid_json=("valid_json", "mean"),
        std_latency_seconds=("latency_seconds", "std"),
        p95_latency_seconds=("latency_seconds", p95),
        sla_hit_rate=("latency_seconds", lambda s: float((s <= config.SLA_SECONDS).mean())),
        avg_tokens_per_second=("tokens_per_second", "mean"),
        avg_answer_words=("answer_words", "mean"),
        avg_key_points=("key_points", "mean"),
    ).reset_index()
    g["cost_per_1k_requests"] = g.avg_cost_per_request * 1000
    return g


def pick_best(summary: pd.DataFrame) -> str:
    """Highest JSON validity first (the product requires strict JSON), then lowest cost, then latency."""
    ranked = summary.sort_values(["valid_json", "avg_cost_per_request", "avg_latency_seconds"],
                                 ascending=[False, True, True])
    return ranked.iloc[0].prompt_version


def scale_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Monthly cost for each volume scenario; also concurrency needed at the assumed peak rate (Little's law)."""
    s = summary.set_index("prompt_version")
    out = pd.DataFrame({name: s.avg_cost_per_request * n for name, n in config.SCALE_VOLUMES.items()})
    out["concurrent_requests_at_peak"] = s.avg_latency_seconds * config.PEAK_RPS
    return out
