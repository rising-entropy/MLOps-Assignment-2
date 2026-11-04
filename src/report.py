"""Builds the final PDF report (500-700 words of prose) from the actual results.
Every number is computed from the saved logs; claims that depend on the data are asserted, not assumed."""
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from . import analysis, config
from .prompts import QUESTION_META

SCREENSHOTS = [
    ("mlflow_experiment.jpg", "MLflow experiment: runs and metrics"),
    ("metrics.jpg", "Run metrics (prompt_v3)"),
    ("prompt_registry.jpg", "Prompt Registry: v1, v2, v3"),
    ("prompt_diff.jpg", "Prompt diff (v2 vs v3)"),
    ("production_alias.jpg", "`production` alias on the promoted version"),
]


def _table(df: pd.DataFrame) -> Table:
    data = [list(df.columns)] + df.astype(str).values.tolist()
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("FONTSIZE", (0, 0), (-1, -1), 7),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.grey)]))
    return t


def _fmt(d: dict, fmt: str) -> str:
    return ", ".join(f"{k}: {fmt.format(v)}" for k, v in d.items())


def build() -> str:
    df = analysis.load_calls()
    summ = pd.read_csv(config.EVAL_CSV)
    s = summ.set_index("prompt_version")
    best = summ.loc[summ.recommended == "Yes", "prompt_version"].iloc[0]
    others = [v for v in s.index if v != best]
    n_calls, n_q = len(df), df.question_id.nunique()

    # claims the narrative relies on: fail loudly if the data disagrees
    assert (df.status == "success").all(), "some requests failed"
    assert s.loc[best, "valid_json"] == s.valid_json.max()
    assert all(s.loc[best, "avg_cost_per_request"] < s.loc[o, "avg_cost_per_request"] for o in others)
    assert all(s.loc[best, "avg_latency_seconds"] < s.loc[o, "avg_latency_seconds"] for o in others)

    v1_rep1 = analysis.latency_table(df, "v1", 1)
    st = analysis.latency_summary(v1_rep1.latency_seconds)
    slow = analysis.slowest_request(df)
    corr = df.latency_seconds.corr(df.output_tokens)
    strength = ("closely tracks" if abs(corr) >= 0.7 else "moderately tracks" if abs(corr) >= 0.4
                else "only weakly tracks")
    n_per = n_calls // len(s)
    speed_z = {}
    for o in others:  # gap in seconds and in standard errors, a rough measure of how reliable it is
        g = s.loc[o, "avg_latency_seconds"] - s.loc[best, "avg_latency_seconds"]
        se = ((s.loc[best, "std_latency_seconds"] ** 2 + s.loc[o, "std_latency_seconds"] ** 2) / n_per) ** 0.5
        speed_z[o] = (g, g / se)
    proj = analysis.monthly_projection(df).set_index("prompt_version")
    scale = analysis.scale_table(summ)
    price_ratio = config.OUTPUT_PRICE_PER_M / config.INPUT_PRICE_PER_M
    _cost = df.groupby("prompt_version")[["output_cost", "total_cost"]].sum()
    out_share = _cost.output_cost / _cost.total_cost
    big = max(config.SCALE_VOLUMES, key=config.SCALE_VOLUMES.get)
    saving = (scale.loc["v1", big] - scale.loc[best, big])
    cut_cost = 1 - s.loc[best, "avg_cost_per_request"] / s.loc["v1", "avg_cost_per_request"]
    cut_out = 1 - s.loc[best, "avg_output_tokens"] / s.loc["v1", "avg_output_tokens"]
    cats = ", ".join(m["category"].lower() for m in QUESTION_META.values())

    sections = {
        "1. Introduction": (
            f"EduAssist AI needs an academic question-answering assistant that returns strict JSON. This report "
            f"measures three prompt versions using {config.MODEL} ({config.API_VERSION}), priced at "
            f"{config.INPUT_PRICE_PER_M} {config.CURRENCY} per million input tokens and {config.OUTPUT_PRICE_PER_M} "
            f"{config.CURRENCY} per million output tokens, as published on {config.PRICING_DATE}. The {n_q} questions come "
            f"from MTech modules and mimic a real support desk ({cats}), so answer lengths differ. Each version was "
            f"sent every question {config.REPEATS} times, interleaved, giving {n_calls} requests at temperature "
            f"{config.TEMPERATURE}; all succeeded."),
        "2. Task 1: Latency": (
            f"For the five v1 requests of the first repeat, mean latency was {st['mean']:.2f}s, median "
            f"{st['median']:.2f}s, minimum {st['min']:.2f}s, maximum {st['max']:.2f}s and standard deviation "
            f"{st['std']:.2f}s. Across all {n_calls} requests the slowest was {slow.question_id} with prompt "
            f"{slow.prompt_version} at {slow.latency_seconds:.2f}s and {int(slow.output_tokens)} output tokens. "
            f"Latency {strength} output length (r = {corr:.2f}): generation is sequential, but network and provider "
            f"load add noise. Repeating requests quantified it: mean latency with p95 was "
            + ", ".join(f"{v} {s.loc[v, 'avg_latency_seconds']:.2f}s (p95 {s.loc[v, 'p95_latency_seconds']:.2f}s)"
                        for v in s.index)
            + f". Relative to {best}, the other versions were slower by "
            + ", ".join(f"{g:.2f}s ({z:.1f} standard errors) for {o}" for o, (g, z) in speed_z.items())
            + "; gaps under about two standard errors are not reliable, so latency alone should not decide "
              "the promotion."),
        "3. Task 2: Tokens and cost": (
            f"Cost per request is input tokens / 1,000,000 x input price plus output tokens / 1,000,000 x output "
            f"price. Average cost per request was {_fmt(proj.avg_cost_per_request.to_dict(), '{:.6f}')} "
            f"{config.CURRENCY}, which projects to {_fmt(proj.projected_monthly_cost.to_dict(), '{:.2f}')} "
            f"{config.CURRENCY} for {config.MONTHLY_REQUESTS:,} requests a month. Output tokens cost {price_ratio:.1f} "
            f"times as much as input tokens and make up {out_share.min():.0%} to {out_share.max():.0%} of spend, so "
            f"the most effective saving is to shorten responses: {best} wrote {cut_out:.0%} fewer output tokens than "
            f"v1 and cost {cut_cost:.0%} less. Capping max tokens, caching the stable prompt prefix and routing "
            f"simple questions to a smaller model are further options."),
        "4. Business and scale view": (
            f"The brief fixes {config.MONTHLY_REQUESTS:,} requests a month, but a real service sees other volumes. "
            f"At the {big.lower()} volume of "
            f"{config.SCALE_VOLUMES[big]:,} requests a month, {best} costs {scale.loc[best, big]:,.0f} "
            f"{config.CURRENCY} against {scale.loc['v1', big]:,.0f} for v1, saving {saving:,.0f} a month. Against an "
            f"assumed {config.SLA_SECONDS:.0f}s interactive target, the share of requests within it was "
            + _fmt({v: s.loc[v, 'sla_hit_rate'] for v in s.index}, "{:.0%}")
            + f". At an assumed peak of {config.PEAK_RPS:g} requests per second, the service needs about "
            + _fmt({v: scale.loc[v, 'concurrent_requests_at_peak'] for v in s.index}, "{:.1f}")
            + " requests in flight. Volumes, target and peak rate are illustrative assumptions, not measurements."),
        "5. Task 3: Prompt versioning and promotion": (
            f"The prompts were registered as versions of {config.PROMPT_NAME} in the MLflow Prompt Registry with "
            f"commit messages, evaluated on the same questions, and logged in {config.EXPERIMENT_NAME}. Version "
            f"{best} was promoted by setting the {config.PRODUCTION_ALIAS} alias. Promotion justification: the "
            f"product contract requires strict JSON, so validity decides. Valid JSON rates were "
            f"{_fmt(s.valid_json.to_dict(), '{:.0%}')}. Version {best} also had the lowest average latency "
            f"({s.loc[best, 'avg_latency_seconds']:.2f}s), the fewest output tokens "
            f"({s.loc[best, 'avg_output_tokens']:.0f}) and the lowest cost "
            f"({s.loc[best, 'avg_cost_per_request']:.6f} {config.CURRENCY} per request). It sends more input "
            f"tokens because the schema is included, but input is the cheap side. A free-text answer would need a "
            f"second model call or fragile parsing to become JSON. Answer correctness was not scored automatically; "
            f"the confidence and key_points fields support human review. The alias lets the application load the "
            f"production prompt without code changes and roll back instantly."),
        "6. Conclusion": (
            "Treating prompts as versioned artifacts, with measured latency, tokens and cost, gives a reproducible "
            "basis for promotion. Limitations: five questions is a small sample, the scale figures rest on stated "
            "assumptions, and quality scoring on a larger set should confirm the choice before wide rollout."),
    }
    words = sum(len(re.findall(r"\S+", t)) for t in sections.values())
    print(f"Report prose word count: {words} (target 500-700)")
    assert 500 <= words <= 700, f"report prose is {words} words"

    styles = getSampleStyleSheet()
    out = config.OUTPUT_DIR / "final_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2 * cm,
                            bottomMargin=2 * cm)
    story = [Paragraph("LLMOps Assignment 2: API Cost, Latency and Prompt Version Management", styles["Title"]),
             Paragraph("Course: MLOps | EduAssist AI scenario", styles["Normal"]), Spacer(1, 12)]
    for head, body in sections.items():
        story += [Paragraph(head, styles["Heading2"]), Paragraph(body, styles["BodyText"]), Spacer(1, 6)]
        if head.startswith("2."):
            story += [_table(v1_rep1.drop(columns=["model"]).round(3)), Spacer(1, 6)]
        if head.startswith("3."):
            story += [_table(analysis.cost_table(df, "v1").round(6)), Spacer(1, 6)]
        if head.startswith("4."):
            sc = scale.round(1).reset_index().rename(columns={"prompt_version": "version"})
            sc.columns = [c if c in ("version", "concurrent_requests_at_peak") else f"{c} (USD/month)"
                          for c in sc.columns]
            story += [_table(sc), Spacer(1, 6)]
        if head.startswith("5."):
            story += [_table(summ[["prompt_version", "avg_latency_seconds", "avg_input_tokens", "avg_output_tokens",
                                   "avg_cost_per_request", "valid_json", "recommended"]].round(6)), Spacer(1, 6)]
    for fname, caption in SCREENSHOTS:
        path = config.ROOT / "screenshots" / fname
        if path.exists():
            story += [KeepTogether([Paragraph(caption, styles["Heading4"]),
                                    Image(str(path), width=16 * cm, height=9 * cm, kind="proportional")]),
                      Spacer(1, 8)]
    doc.build(story)
    return str(out)
