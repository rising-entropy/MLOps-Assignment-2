"""Generates assignment.ipynb (run `python build_notebook.py`, then execute the notebook).

Layout and tone follow Assignment 1: header table, pre-requisites, one short purpose line per step,
self-contained code, then 'Result' / 'Observation' bullets that quote the real numbers.

Every number in the markdown is computed here from outputs/latency_tokens_cost.csv, and claims that depend on
the data are asserted, so the narrative cannot drift from the results. Rebuild after any new API run.
"""
import sys
from importlib.metadata import version as pkg_version

import nbformat as nbf
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src import analysis, config  # noqa: E402
from src.prompts import QUESTION_META, QUESTIONS  # noqa: E402

assert not config.DRY_RUN, "build the submission notebook from real results, not a dry run"

# ----------------------------------------------------------------------------- numbers for the narrative
df = analysis.load_calls()
assert (df.status == "success").all(), "failed requests in the log"
S = analysis.version_summary(df).set_index("prompt_version")
best = analysis.pick_best(S.reset_index())
others = [v for v in S.index if v != best]
R = config.REPEATS
n_q, n_calls = df.question_id.nunique(), len(df)
cur, ccy = config.INPUT_PRICE_PER_M, config.CURRENCY

v1r1 = df[(df.prompt_version == "v1") & (df.repeat == 1)]
lat = v1r1.latency_seconds
slow = v1r1.loc[lat.idxmax()]
longest = v1r1.loc[v1r1.output_tokens.idxmax()]
corr = df.latency_seconds.corr(df.output_tokens)
strength = "closely tracks" if abs(corr) >= 0.7 else "moderately tracks" if abs(corr) >= 0.4 else "only weakly tracks"
spread = df.groupby(["prompt_version", "question_id"]).latency_seconds.agg(["min", "max"])
spread["ratio"] = spread["max"] / spread["min"]
worst = spread.ratio.idxmax()
price_ratio = config.OUTPUT_PRICE_PER_M / config.INPUT_PRICE_PER_M
_cost = df.groupby("prompt_version")[["output_cost", "total_cost"]].sum()
out_share = _cost.output_cost / _cost.total_cost
proj = analysis.monthly_projection(df).set_index("prompt_version")
scale = analysis.scale_table(S.reset_index())
mid = "Exam season"
big = max(config.SCALE_VOLUMES, key=config.SCALE_VOLUMES.get)
cut_out = 1 - S.loc[best, "avg_output_tokens"] / S.loc["v1", "avg_output_tokens"]
cut_cost = 1 - S.loc[best, "avg_cost_per_request"] / S.loc["v1", "avg_cost_per_request"]
breakeven = S.loc["v1", "avg_cost_per_request"] / S.loc[best, "avg_cost_per_request"]

# claims the text below relies on
assert S.loc[best, "valid_json"] == S.valid_json.max() == 1.0
assert all(S.loc[best, "avg_cost_per_request"] < S.loc[o, "avg_cost_per_request"] for o in others)
assert all(S.loc[best, "avg_latency_seconds"] < S.loc[o, "avg_latency_seconds"] for o in others)
assert all(S.loc[best, "avg_output_tokens"] < S.loc[o, "avg_output_tokens"] for o in others)
assert S.loc[best, "avg_input_tokens"] > S.loc[others, "avg_input_tokens"].max()


rep_means = df.groupby("repeat").latency_seconds.mean()
n_per = n_calls // 3
# how strong is each speed difference? gap divided by its standard error (a rough z score)
speed_z = {}
for o in others:
    g = S.loc[o, "avg_latency_seconds"] - S.loc[best, "avg_latency_seconds"]
    se = ((S.loc[best, "std_latency_seconds"] ** 2 + S.loc[o, "std_latency_seconds"] ** 2) / n_per) ** 0.5
    speed_z[o] = (g, g / se)
reliable = [o for o, (g, z) in speed_z.items() if z >= 2]
unsure = [o for o in others if o not in reliable]
detail = "; ".join(f"{o}: {g:.2f} s slower, {z:.1f} standard errors" for o, (g, z) in speed_z.items())
if not unsure:
    speed_claim = (f"Every gap is at least two standard errors ({detail}), so the speed ordering is unlikely to be "
                   f"noise.")
else:
    verdict = {o: ("suggestive but not conclusive" if speed_z[o][1] >= 1 else "within the noise")
               for o in unsure}
    verdict.update({o: "convincing" for o in reliable})
    speed_claim = (f"Measured against {best}, the other versions were {detail}. As a rough rule only a gap of about "
                   f"two standard errors or more is trustworthy, so on this run the speed advantage over "
                   + ", and over ".join(f"{o} is {v}" for o, v in verdict.items()) + ".")


def lst(series, fmt):
    return ", ".join(f"{k} {fmt.format(v)}" for k, v in series.items())


nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# --------------------------------------------------------------------------- header
md('''
### MLOps AIMLCZG523
### Assignment 2

#### LLMOps: API Cost, Latency and Prompt Version Management


|Name|BITS ID|
|---|---|
|Devang Ajay Kamble | 2025AE05173 |
''')

libs = ["mlflow", "openai", "pandas", "numpy", "matplotlib", "python-dotenv"]
md(f'''
#### Pre-requisites

We would be using Python 3.10 and the following libraries:
''' + "\n".join(f"- {p}=={pkg_version(p)}" for p in libs) + '''

Secrets are never written in the notebook. The API key and model settings are read from environment variables (or a git-ignored `.env` file): `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `INPUT_PRICE_PER_M`, `OUTPUT_PRICE_PER_M`, `PRICING_DATE`. See `README.md` for setup.

Scenario: EduAssist AI needs an academic Q&A assistant that returns strict JSON. We call an LLM with three prompt versions, measure latency and token cost, look at what that means for a real service at scale, and promote the best version in MLflow.
''')

md('#### Imports and configuration')

code(f'''
import difflib
import json
import os
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from mlflow import MlflowClient
from openai import OpenAI

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path.cwd()
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
load_dotenv(ROOT / ".env")

API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL") or None
MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
API_VERSION = os.getenv("LLM_API_VERSION", "OpenAI Chat Completions API")
TEMPERATURE, MAX_TOKENS = 0.0, 1024
REPEATS = {R}  # every version x question pair is sent this many times

# published prices per 1M tokens (USD); the price page and date are recorded in the report
IN_PRICE = float(os.getenv("INPUT_PRICE_PER_M", "{config.INPUT_PRICE_PER_M}"))
OUT_PRICE = float(os.getenv("OUTPUT_PRICE_PER_M", "{config.OUTPUT_PRICE_PER_M}"))
PRICING_DATE = os.getenv("PRICING_DATE", "{config.PRICING_DATE}")
CURRENCY = "USD"
MONTHLY_REQUESTS = 10_000  # the volume fixed by the brief

# business and scale ASSUMPTIONS (illustrative, not measured): change them to fit your own scenario
SLA_SECONDS = {config.SLA_SECONDS}  # interactive-chat latency target
SCALE_VOLUMES = {config.SCALE_VOLUMES}
PEAK_RPS = {config.PEAK_RPS}  # requests per second at the busiest moment
PRICE_MULTIPLIERS = {config.PRICE_MULTIPLIERS}  # what-if: model price relative to today's

EXPERIMENT = "eduassist_llmops_assignment"
PROMPT_NAME = "eduassist-qa-prompt"
CALLS_CSV = OUT / "latency_tokens_cost.csv"
EVAL_CSV = OUT / "prompt_evaluation_results.csv"

print("Model        :", MODEL)
print("API          :", API_VERSION)
print(f"Pricing      : {{IN_PRICE}} in / {{OUT_PRICE}} out {{CURRENCY}} per 1M tokens (checked {{PRICING_DATE}})")
print("Repeats      :", REPEATS)
print("API key set  :", bool(API_KEY))
''')

# --------------------------------------------------------------------------- questions and prompts
md('#### Questions and prompt versions')

rows = "\n".join(f"|{q}|{m['category']}|{QUESTIONS[q]}|" for q, m in QUESTION_META.items())
md(f'''
The brief asks for five academic questions. Instead of generic textbook ones, these are drawn from our own MTech modules (ML, deep learning, MLOps, LLMOps) and written the way a working-professional student would ask a support desk. They differ on purpose in kind and expected answer length, because EduAssist's real traffic would, and a spread of lengths exercises the cost and latency behaviour better than five look-alike questions. Every prompt version gets the same questions, so differences come from the prompt and not from the input.

|ID|Category|Question|
|---|---|---|
{rows}

`{{{{question}}}}` below is the MLflow template variable.
''')

q_code = "QUESTIONS = {\n" + "".join(f"    {q!r}: {t!r},\n" for q, t in QUESTIONS.items()) + "}\n\n"
q_code += "QUESTION_META = {\n" + "".join(
    f"    {q!r}: dict(category={m['category']!r}),\n" for q, m in QUESTION_META.items()) + "}\n"

code(q_code + '''
PROMPT_V1 = """Answer the following student question clearly.
Question: {{question}}"""

PROMPT_V2 = """You are an academic assistant.
Answer the following student question in a clear, concise, and accurate manner.
Use simple language suitable for undergraduate students.
Question: {{question}}"""

PROMPT_V3 = """You are an academic assistant.
Answer the student question and return ONLY valid JSON.
Do not include markdown, explanations outside JSON, or code fences.
Use this exact JSON schema:
{
  "answer": "string",
  "confidence": "high | medium | low",
  "key_points": ["string"]
}
Question: {{question}}"""

PROMPTS = {
    "v1": dict(template=PROMPT_V1, style="Basic",
               commit="v1 Basic: bare instruction, no role, no format or audience guidance (baseline)."),
    "v2": dict(template=PROMPT_V2, style="Structured",
               commit="v2 Structured: adds academic-assistant role and clear/concise/simple-language constraints."),
    "v3": dict(template=PROMPT_V3, style="JSON Output",
               commit="v3 JSON Output: adds strict JSON-only output with answer/confidence/key_points schema."),
}


def render(template, question):
    return template.replace("{{question}}", question)


print(render(PROMPT_V2, QUESTIONS["Q1"]))
''')

md('''
#### JSON validity check

The `valid_json` metric is strict on purpose. The raw response must parse as JSON and match the schema exactly: keys `answer`, `confidence`, `key_points`, with `confidence` limited to high / medium / low. A response wrapped in a code fence or preceded by a sentence counts as invalid, because the prompt forbids both and a downstream parser would fail on them.
''')

code('''
def is_valid_json(text):
    # strict: raw text must parse and match the schema, no fences or extra prose
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return False
    return (isinstance(obj, dict)
            and set(obj) == {"answer", "confidence", "key_points"}
            and isinstance(obj["answer"], str)
            and obj["confidence"] in {"high", "medium", "low"}
            and isinstance(obj["key_points"], list)
            and all(isinstance(k, str) for k in obj["key_points"]))


good = '{"answer": "x", "confidence": "high", "key_points": ["a"]}'
print("plain JSON  :", is_valid_json(good))
print("code fence  :", is_valid_json("```json\\n" + good + "\\n```"))
print("prose first :", is_valid_json("Sure! " + good))
print("bad schema  :", is_valid_json('{"answer": "x"}'))
''')

md('''
Result

* Only the plain, schema-conforming JSON passes. The fenced, prose-prefixed and incomplete replies are all rejected, so `valid_json` will not be flattered by sloppy formatting.
''')

# --------------------------------------------------------------------------- task 1
md('### Task-1: Call an LLM and Measure Latency')

md('''
#### API call function

One function sends a prompt and returns everything Task 1 and Task 2 need: start and end time, latency, status, response text and token usage. Latency is timed with a monotonic clock around the API call only, so local processing is excluded. The key comes from the environment, and any API failure is recorded as `status = "error"` instead of crashing the run (it is retried with a growing delay first).
''')

code('''
def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def call_llm(prompt, retries=4):
    if not API_KEY:
        raise RuntimeError("Set LLM_API_KEY in your environment or .env file")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    for attempt in range(retries + 1):
        start, t0 = now(), time.perf_counter()
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": prompt}])
            return dict(start_time=start, end_time=now(), latency_seconds=time.perf_counter() - t0,
                        status="success", response_text=r.choices[0].message.content or "",
                        input_tokens=r.usage.prompt_tokens, output_tokens=r.usage.completion_tokens,
                        total_tokens=r.usage.total_tokens, error="")
        except Exception as exc:  # record any API failure in the log
            failure = dict(start_time=start, end_time=now(), latency_seconds=time.perf_counter() - t0,
                           status="error", response_text="", input_tokens=0, output_tokens=0,
                           total_tokens=0, error=f"{type(exc).__name__}: {exc}")
            time.sleep(2 ** attempt)
    return failure
''')

md(f'''
#### Run the requests

Each of the three prompt versions is sent each of the five questions {R} times: 5 x 3 x {R} = {n_calls} requests. The repeats are interleaved (repeat, then version, then question) so a slow period on the provider's side hits every version alike instead of penalising whichever one happened to run then.

Task 1 reports the five `v1` requests of the first repeat, as the brief asks. The repeats are used afterwards to measure how noisy latency really is, and Tasks 2 and 3 average over all of them, so every number in this notebook comes from one set of API calls. Cost, tokens per second, answer length and key-point count are added to each row.

The cell below reuses the saved log `outputs/latency_tokens_cost.csv` so that the notebook, screenshots and report stay consistent. Set the environment variable `REFRESH=1` to call the API again (a few minutes).
''')

code('''
def input_cost(tokens):
    return tokens / 1_000_000 * IN_PRICE


def output_cost(tokens):
    return tokens / 1_000_000 * OUT_PRICE


def key_point_count(text):
    try:
        return len(json.loads(text)["key_points"])
    except (TypeError, ValueError, KeyError):
        return np.nan


def collect_calls():
    rows = []
    for rep in range(1, REPEATS + 1):
        for version, p in PROMPTS.items():
            for qid, question in QUESTIONS.items():
                r = call_llm(render(p["template"], question))
                ic, oc = input_cost(r["input_tokens"]), output_cost(r["output_tokens"])
                rows.append(dict(repeat=rep, question_id=qid, prompt_version=version, model=MODEL,
                                 start_time=r["start_time"], end_time=r["end_time"],
                                 latency_seconds=round(r["latency_seconds"], 4), status=r["status"],
                                 input_tokens=r["input_tokens"], output_tokens=r["output_tokens"],
                                 total_tokens=r["total_tokens"], input_cost=ic, output_cost=oc,
                                 total_cost=ic + oc,
                                 tokens_per_second=r["output_tokens"] / r["latency_seconds"],
                                 answer_words=len(r["response_text"].split()),
                                 key_points=key_point_count(r["response_text"]),
                                 valid_json=is_valid_json(r["response_text"]),
                                 response_text=r["response_text"], error=r["error"]))
                print(f"rep{rep} {version} {qid}: {r['status']:7s} {r['latency_seconds']:.2f}s "
                      f"in={r['input_tokens']} out={r['output_tokens']}")
    df = pd.DataFrame(rows)
    df.to_csv(CALLS_CSV, index=False)
    return df


if CALLS_CSV.exists() and os.getenv("REFRESH") != "1":
    df = pd.read_csv(CALLS_CSV)
    print("Loaded saved log:", CALLS_CSV.relative_to(ROOT))
else:
    df = collect_calls()

print("Shape    :", df.shape)
print("Status   :", df["status"].value_counts().to_dict())
df.drop(columns=["response_text", "error"]).head()
''')

md(f'''
Result

* All {n_calls} requests ({R} repeats x 3 versions x 5 questions) succeeded, and the log is saved to `outputs/latency_tokens_cost.csv`.
''')

md('#### Latency table: five v1 requests')

code('''
v1 = df[(df["prompt_version"] == "v1") & (df["repeat"] == 1)]
latency_table = v1[["question_id", "prompt_version", "model", "start_time", "end_time",
                    "latency_seconds", "status"]].reset_index(drop=True)
latency_table
''')

md('#### Latency summary statistics')

code('''
lat = v1["latency_seconds"]
lat_stats = pd.Series({"mean": lat.mean(), "median": lat.median(), "min": lat.min(),
                       "max": lat.max(), "std": lat.std(ddof=1)}, name="latency_seconds").round(3)
print(lat_stats.to_string())

slow = v1.loc[lat.idxmax()]
print(f"\\nSlowest v1 request: {slow.question_id}, {slow.latency_seconds:.2f}s, {slow.output_tokens} output tokens")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.bar(v1["question_id"], v1["latency_seconds"])
ax.set(title="v1 latency per question (repeat 1)", ylabel="seconds")
plt.tight_layout()
plt.show()
''')

if slow.question_id == longest.question_id:
    why_slow = (f"{slow.question_id} was also the longest answer ({int(slow.output_tokens)} output tokens), "
                f"so its length is the likely reason.")
else:
    why_slow = (f"It was not the longest answer ({longest.question_id} wrote {int(longest.output_tokens)} tokens in "
                f"{longest.latency_seconds:.2f} s, against {int(slow.output_tokens)} tokens here), so output length does "
                f"not explain it alone: network round-trip time and provider load at that moment also vary from call to call.")
md(f'''
Observation

* **Spread:** the five v1 requests average {lat.mean():.2f} s (median {lat.median():.2f} s), range {lat.min():.2f} s to {lat.max():.2f} s, standard deviation {lat.std(ddof=1):.2f} s.
* **Slowest request:** {slow.question_id} ({QUESTION_META[slow.question_id]['category'].lower()}) at {slow.latency_seconds:.2f} s with {int(slow.output_tokens)} output tokens. {why_slow}
* Five requests are too few to say how much of this is the prompt and how much is chance, so the next step measures it with repeats.
''')

md('#### How noisy is latency?')

md(f'''
Each (version, question) pair was sent {R} times with identical input and `temperature = 0`, so any difference in latency between repeats is noise from the network and the provider, not from the prompt. The table gives mean, standard deviation and the 95th percentile (p95: the latency that 95% of requests stay under, which is what a service-level target is normally written against) for each version, plus the share of requests that met the assumed {config.SLA_SECONDS:.0f} s interactive target.
''')

code('''
def p95(s):
    return float(np.percentile(s, 95))


noise = df.groupby("prompt_version").agg(
    mean_s=("latency_seconds", "mean"), std_s=("latency_seconds", "std"),
    min_s=("latency_seconds", "min"), p95_s=("latency_seconds", p95), max_s=("latency_seconds", "max"),
    sla_hit_rate=("latency_seconds", lambda s: (s <= SLA_SECONDS).mean()))
print(noise.round(3).to_string())

same_input = df.groupby(["prompt_version", "question_id"])["latency_seconds"].agg(["min", "max"])
same_input["max_over_min"] = same_input["max"] / same_input["min"]
print("\\nLargest swing for identical input:", same_input["max_over_min"].idxmax(),
      f"{same_input['max_over_min'].max():.1f}x between fastest and slowest repeat")
print("Correlation (latency, output tokens): %.2f" % df["latency_seconds"].corr(df["output_tokens"]))
print("Mean latency by repeat (s):", df.groupby("repeat")["latency_seconds"].mean().round(2).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
df.boxplot(column="latency_seconds", by="prompt_version", ax=axes[0])
axes[0].set(title="latency by prompt version", xlabel="", ylabel="seconds")
axes[1].scatter(df["output_tokens"], df["latency_seconds"], c=df["prompt_version"].map({"v1": 0, "v2": 1, "v3": 2}))
axes[1].set(title="all requests: latency vs output tokens", xlabel="output tokens", ylabel="seconds")
plt.suptitle("")
plt.tight_layout()
plt.show()
''')

md(f'''
Observation

* **Ranking:** mean latency was {lst(S.avg_latency_seconds, '{:.2f} s')}. {best} had the lowest average, but the standard deviations ({lst(S.std_latency_seconds, '{:.2f} s')}) show how much a single request can differ from it. {speed_claim}
* **Tail:** p95 was {lst(S.p95_latency_seconds, '{:.2f} s')}. A service target should be written against p95, not the mean, because the slow requests are the ones users complain about.
* **Noise:** identical input took up to {spread.ratio.max():.1f}x longer on one repeat than another ({worst[0]} {worst[1]}). Output length {strength} latency (correlation {corr:.2f}), so provider and network conditions add a large share on top of generation time.
* **Drift:** mean latency by repeat was {', '.join(f'repeat {k} {v:.2f} s' for k, v in rep_means.items())}, so the provider's speed changed during the run itself. This is why the repeats are interleaved: a version that happened to run in a slow minute would otherwise look worse for no reason of its own.
* **Target:** the share of requests within {config.SLA_SECONDS:.0f} s was {lst(S.sla_hit_rate, '{:.0%}')}. These absolute figures describe the provider at the time of this run. They should be re-measured at the time of deployment, not read off one run.
''')

# --------------------------------------------------------------------------- task 2
md('### Task-2: Count Tokens and Calculate Cost')

md(f'''
Token counts are read from the `usage` field of each API response, not estimated. Cost uses the published per-million-token prices:

```
Input Cost   = Input Tokens  / 1,000,000 x Input Price per Million
Output Cost  = Output Tokens / 1,000,000 x Output Price per Million
Total Cost   = Input Cost + Output Cost
Monthly Cost = Average Cost per Request x 10,000
```

Pricing: Gemini 3.5 Flash-Lite, standard paid tier, {config.INPUT_PRICE_PER_M} USD per 1M input and {config.OUTPUT_PRICE_PER_M} USD per 1M output tokens, checked on {config.PRICING_DATE}. The key used here is on the free tier, so the actual bill is zero; the figures are what the same usage would cost at published rates.
''')

md('#### Tokens and cost per request')

md('''
The table shows every question under every prompt version for the first repeat, which is the "per request" view the brief asks for. The full log with all repeats is in `outputs/latency_tokens_cost.csv`.
''')

code('''
cost_table = df[df["repeat"] == 1][["question_id", "prompt_version", "input_tokens", "output_tokens",
                                    "total_tokens", "input_cost", "output_cost", "total_cost"]]
cost_table.sort_values(["prompt_version", "question_id"]).reset_index(drop=True)
''')

md('#### Monthly projection for 10,000 requests')

md(f'''
The brief's projection is the average cost per request times 10,000. The average here is taken over all {n_calls} requests, which is steadier than a single run.
''')

code('''
proj = df.groupby("prompt_version").agg(avg_input_tokens=("input_tokens", "mean"),
                                        avg_output_tokens=("output_tokens", "mean"),
                                        avg_cost_per_request=("total_cost", "mean"))
proj["projected_monthly_cost"] = proj["avg_cost_per_request"] * MONTHLY_REQUESTS
proj.round(6)
''')

md('#### Extra measures')

md('''
Four extra measures turn the same log into figures a business reader can use: output tokens per second (how fast the model generates), cost per 1,000 answers (an easier unit than fractions of a cent), the share of the bill that is output tokens, and the average answer length and number of key points.
''')

code('''
def summarize(d):
    g = d.groupby("prompt_version").agg(
        avg_latency_seconds=("latency_seconds", "mean"),
        avg_input_tokens=("input_tokens", "mean"),
        avg_output_tokens=("output_tokens", "mean"),
        avg_cost_per_request=("total_cost", "mean"),
        valid_json=("valid_json", "mean"),
        std_latency_seconds=("latency_seconds", "std"),
        p95_latency_seconds=("latency_seconds", p95),
        sla_hit_rate=("latency_seconds", lambda s: (s <= SLA_SECONDS).mean()),
        avg_tokens_per_second=("tokens_per_second", "mean"),
        avg_answer_words=("answer_words", "mean"),
        avg_key_points=("key_points", "mean"))
    g["cost_per_1k_requests"] = g["avg_cost_per_request"] * 1000
    return g.reset_index()


summary = summarize(df)
cost_sum = df.groupby("prompt_version")[["output_cost", "total_cost"]].sum()
share = cost_sum["output_cost"] / cost_sum["total_cost"]
extra = summary.set_index("prompt_version")[["avg_tokens_per_second", "cost_per_1k_requests",
                                              "avg_answer_words", "avg_key_points"]].assign(output_share_of_cost=share)
print(f"Output token price / input token price = {OUT_PRICE / IN_PRICE:.1f}x")
extra.round(4)
''')

md(f'''
Result

* v1 costs {S.loc['v1', 'avg_cost_per_request']:.6f} USD per request, which projects to {proj.loc['v1', 'projected_monthly_cost']:.2f} USD for 10,000 requests a month; {'; '.join(f"{v} projects to {proj.loc[v, 'projected_monthly_cost']:.2f} USD" for v in S.index if v != 'v1')}.
* Output tokens are {out_share.min():.0%} to {out_share.max():.0%} of the spend, so {best} is cheaper even though it sends {S.loc[best, 'avg_input_tokens'] / S.loc['v1', 'avg_input_tokens']:.1f} times the input tokens of v1: it writes only {S.loc[best, 'avg_output_tokens'] / S.loc['v1', 'avg_output_tokens']:.0%} of the output.
''')

md(f'''
#### Cost-optimization recommendation

**Shorten the responses.** Output tokens drive almost all of the cost, so controlling answer length gives the largest saving. Moving from the open-ended v1 to the constrained {best} cut output tokens by {cut_out:.0%} and cost per request by {cut_cost:.0%} ({S.loc['v1', 'avg_cost_per_request']:.6f} to {S.loc[best, 'avg_cost_per_request']:.6f} USD), about {proj.loc['v1', 'projected_monthly_cost'] - proj.loc[best, 'projected_monthly_cost']:.2f} USD a month at 10,000 requests. A `max_tokens` cap on top of this prevents a rare long answer from inflating the bill.

Further options, in order of effort:
* Cache the stable prompt prefix (role and schema) if the provider supports prompt caching. This helps {best} most, since its input is the largest.
* Route simple, short questions to a smaller or cheaper model and keep the larger one for hard questions.
''')

# --------------------------------------------------------------------------- scale and business view
md('#### Scale and business view')

vol = ", ".join(f"{k.lower()} {v:,}" for k, v in config.SCALE_VOLUMES.items())
md(f'''
The brief fixes 10,000 requests, but a decision about a real service depends on what happens at other volumes and against a latency target. The figures in this section combine measured averages with a few stated assumptions, so they are scenarios and not forecasts:

* **Volumes:** {vol} requests a month.
* **Latency target:** {config.SLA_SECONDS:.0f} s for an interactive chat answer.
* **Peak rate:** {config.PEAK_RPS:g} requests per second at the busiest moment. By Little's law, the number of requests in flight is the arrival rate times the average latency, which is roughly how many concurrent connections the service must handle.
''')

code('''
s = summary.set_index("prompt_version")
scale = pd.DataFrame({name: s["avg_cost_per_request"] * n for name, n in SCALE_VOLUMES.items()})
scale["cost_per_request"] = s["avg_cost_per_request"]
scale["concurrent_requests_at_peak"] = s["avg_latency_seconds"] * PEAK_RPS
print("Monthly cost (USD) by volume, plus cost per request and concurrency at peak")
scale.round(6)
''')

code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))

volumes = list(SCALE_VOLUMES.values())
for v in s.index:
    axes[0].plot(volumes, [s.loc[v, "avg_cost_per_request"] * n for n in volumes], marker="o", label=v)
axes[0].set(xscale="log", yscale="log", xlabel="requests per month", ylabel="monthly cost (USD)",
            title="Monthly cost by volume")
axes[0].legend()

mid = list(SCALE_VOLUMES.values())[1]
width = 0.25
for i, v in enumerate(s.index):
    axes[1].bar(np.arange(len(PRICE_MULTIPLIERS)) + i * width,
                [s.loc[v, "avg_cost_per_request"] * m * mid for m in PRICE_MULTIPLIERS], width, label=v)
axes[1].set_xticks(np.arange(len(PRICE_MULTIPLIERS)) + width)
axes[1].set_xticklabels([f"{m}x" for m in PRICE_MULTIPLIERS])
axes[1].set(xlabel="model price relative to today", ylabel="monthly cost (USD)",
            title=f"Price sensitivity at {mid:,} requests a month")
axes[1].legend()
plt.tight_layout()
plt.show()

recommended = summary.sort_values(["valid_json", "avg_cost_per_request"],
                                  ascending=[False, True])["prompt_version"].iloc[0]
breakeven = s.loc["v1", "avg_cost_per_request"] / s.loc[recommended, "avg_cost_per_request"]
print(f"{recommended} stays cheaper than v1 unless the model it runs on costs more than {breakeven:.1f}x as much per token")
''')

md(f'''
Observation

* **Scale:** cost grows in step with volume, so the gap between prompts grows with it. At {config.SCALE_VOLUMES[big]:,} requests a month ({big.lower()}) {best} costs {scale.loc[best, big]:,.0f} USD against {scale.loc['v1', big]:,.0f} USD for v1, a saving of {scale.loc['v1', big] - scale.loc[best, big]:,.0f} USD a month for answering the same questions. At {config.SCALE_VOLUMES[mid]:,} requests (exam season) the saving is {scale.loc['v1', mid] - scale.loc[best, mid]:,.0f} USD.
* **A prompt change is worth a cheaper model:** {best} runs at {cut_cost:.0%} lower cost than v1 on the same model. To lose that advantage, the model behind {best} would have to cost {breakeven:.1f} times as much per token. Prompt design is therefore a cost lever on par with switching to a model that is several times cheaper, and it is free to try.
* **Capacity:** at {config.PEAK_RPS:g} requests per second the service needs about {lst(scale.concurrent_requests_at_peak, '{:.1f}')} requests in flight. These use this run's average latency, which drifted during the run, so treat them as a shape and not a sizing. The point stands: slower and longer answers hold connections open, so output length also drives infrastructure, not just the token bill.
* **Caveat:** volumes, the target and the peak rate are assumptions. Replace them with real usage data before using these numbers to budget.
''')

# --------------------------------------------------------------------------- task 3
md('### Task-3: Prompt Versioning and MLflow Evaluation')

md('''
MLflow's Prompt Registry stores prompts centrally with versions, commit messages and aliases. The plan is:
* register v1, v2 and v3 as three versions of one prompt, `eduassist-qa-prompt`;
* log one MLflow run per version in the experiment `eduassist_llmops_assignment` with the five required metrics, plus the extra measures from above;
* compare the versions and promote the best one by pointing the `production` alias at it.
''')

md('#### Register the three prompt versions')

md('''
Registration is idempotent: if a version with an identical template already exists it is reused, so re-running the notebook does not create duplicate versions. Each version gets a commit message that says what changed.
''')

code('''
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment(EXPERIMENT)
client = MlflowClient()


def existing_version(template):
    v = 1
    while True:
        try:
            p = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{v}")
        except Exception:  # no such prompt or version yet
            return None
        if p.template == template:
            return p.version
        v += 1


versions = {}
for label, p in PROMPTS.items():
    found = existing_version(p["template"])
    if found:
        versions[label] = found
        continue
    reg = mlflow.genai.register_prompt(name=PROMPT_NAME, template=p["template"], commit_message=p["commit"],
                                       tags={"label": label, "style": p["style"]})
    versions[label] = reg.version

for label, v in versions.items():
    print(f"{label} -> registry version {v}: {PROMPTS[label]['commit']}")
''')

md('''
Result

* `eduassist-qa-prompt` now holds three versions (v1, v2, v3 as registry versions 1, 2, 3), each with its own commit message.
''')

md('#### Prompt diff')

md('''
The unified diff shows exactly what changed between versions, which is the point of versioning prompts instead of editing them in place. The diffs are also saved to `outputs/` for the submission, and the MLflow UI shows the same comparison with highlighting (screenshot further below).
''')

code('''
def prompt_diff(a, b):
    lines = difflib.unified_diff(PROMPTS[a]["template"].splitlines(), PROMPTS[b]["template"].splitlines(),
                                 fromfile=f"prompt {a}", tofile=f"prompt {b}", lineterm="")
    return "\\n".join(lines) + "\\n"


for a, b in [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]:
    (OUT / f"prompt_diff_{a}_{b}.txt").write_text(prompt_diff(a, b), encoding="utf-8")

print(prompt_diff("v2", "v3"))
''')

md('''
Observation

* v1 to v2 adds a role and style constraints ("clear, concise, accurate", "undergraduate students") without changing the task.
* v2 to v3 keeps the role and adds the output contract: JSON only, no markdown or code fences, and the exact schema.
''')

md('#### Log one MLflow run per prompt version')

md(f'''
Each run records the model, pricing and prompt version as parameters, and the five required metrics: `avg_latency_seconds`, `avg_input_tokens`, `avg_output_tokens`, `avg_cost_per_request` and `valid_json`. For v3, `valid_json` is valid responses divided by total responses; it is computed the same way for v1 and v2 so the three can be compared. The averages cover all {n_calls // 3} requests per version. The extra measures (latency standard deviation and p95, target hit rate, tokens per second, cost per 1,000 requests) are logged alongside. The per-request responses are attached to each run as a table, and a rerun replaces the earlier run of the same name instead of piling up duplicates.
''')

code('''
required = ["avg_latency_seconds", "avg_input_tokens", "avg_output_tokens", "avg_cost_per_request", "valid_json"]
optional = ["std_latency_seconds", "p95_latency_seconds", "sla_hit_rate", "avg_tokens_per_second",
            "avg_answer_words", "cost_per_1k_requests", "avg_key_points"]

exp_id = mlflow.get_experiment_by_name(EXPERIMENT).experiment_id
for _, row in summary.iterrows():
    label = row["prompt_version"]
    for old in client.search_runs([exp_id], f"tags.mlflow.runName = 'prompt_{label}'"):
        client.delete_run(old.info.run_id)
    sub = df[df["prompt_version"] == label]
    with mlflow.start_run(run_name=f"prompt_{label}"):
        mlflow.log_params({"prompt_name": PROMPT_NAME, "prompt_label": label,
                           "prompt_registry_version": versions[label], "model": MODEL,
                           "api_version": API_VERSION, "temperature": TEMPERATURE,
                           "input_price_per_m": IN_PRICE, "output_price_per_m": OUT_PRICE,
                           "pricing_date": PRICING_DATE, "currency": CURRENCY,
                           "num_questions": sub["question_id"].nunique(), "repeats": REPEATS,
                           "num_requests": len(sub), "sla_seconds": SLA_SECONDS})
        mlflow.log_metrics({k: float(row[k]) for k in required})
        mlflow.log_metrics({k: float(row[k]) for k in optional if pd.notna(row[k])})
        mlflow.log_table(sub.drop(columns=["error"]), artifact_file="responses.json")
        for a, b in [("v1", "v2"), ("v2", "v3"), ("v1", "v3")]:
            mlflow.log_artifact(str(OUT / f"prompt_diff_{a}_{b}.txt"), artifact_path="prompt_diffs")
    print(f"logged run prompt_{label}")
''')

md('#### Compare the versions and promote the best one')

md('''
Selection rule, in order: highest `valid_json` first, because EduAssist's product contract is strict JSON and an answer a parser rejects is unusable; then lowest cost per request; then lowest latency. The winner gets the `production` alias, which lets the application load the current prompt by alias without a code change and roll back by moving the alias.
''')

code('''
ranked = summary.sort_values(["valid_json", "avg_cost_per_request", "avg_latency_seconds"],
                             ascending=[False, True, True])
best = ranked.iloc[0]["prompt_version"]
client.set_prompt_alias(PROMPT_NAME, "production", versions[best])

comparison = summary.copy()
comparison["recommended"] = comparison["prompt_version"].map(lambda v: "Yes" if v == best else "No")
comparison.to_csv(EVAL_CSV, index=False)

print(f"Promoted {best} (registry version {versions[best]}) to alias 'production'")
comparison[["prompt_version"] + required + ["recommended"]].rename(columns={
    "prompt_version": "Prompt Version", "avg_latency_seconds": "Avg Latency (s)",
    "avg_input_tokens": "Avg Input Tokens", "avg_output_tokens": "Avg Output Tokens",
    "avg_cost_per_request": "Avg Cost (USD)", "valid_json": "Valid JSON Rate",
    "recommended": "Recommended?"}).round(6)
''')

code('''
fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
for ax, col, title in zip(axes, ["avg_latency_seconds", "avg_cost_per_request", "valid_json"],
                          ["Avg latency (s)", "Avg cost per request (USD)", "Valid JSON rate"]):
    ax.bar(comparison["prompt_version"], comparison[col],
           color=["tab:green" if r == "Yes" else "tab:gray" for r in comparison["recommended"]])
    ax.set_title(title)
plt.tight_layout()
plt.show()
''')

md(f'''
Result

* {best} is recommended and now carries the `production` alias; valid JSON rate is {lst(S.valid_json, '{:.0%}')}. The table is saved to `outputs/prompt_evaluation_results.csv`.
* {best} also has the lowest average latency ({S.loc[best, 'avg_latency_seconds']:.2f} s vs {' and '.join(f"{S.loc[o, 'avg_latency_seconds']:.2f} s" for o in others)}) and the lowest cost per request ({S.loc[best, 'avg_cost_per_request']:.6f} vs {' and '.join(f"{S.loc[o, 'avg_cost_per_request']:.6f}" for o in others)} USD).
''')

md('#### Check the production alias')

md('''
Load the prompt through the alias exactly as an application would, fill in a question and confirm the model's reply parses against the schema.
''')

code('''
prod = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
print("production -> version", prod.version)
print("commit message:", prod.commit_message)

reply = call_llm(render(prod.template, QUESTIONS["Q1"]))
parsed = json.loads(reply["response_text"])
print("\\nvalid against schema:", is_valid_json(reply["response_text"]))
print("confidence :", parsed["confidence"])
print("key_points :", *parsed["key_points"], sep="\\n  - ")
''')

md(f'''
Result

* `production` resolves to the promoted {best} prompt and its commit message, and a live call through the alias returned JSON that passes the schema check.
* This cell makes one extra API call, so the answer text differs slightly from the saved log.
''')

md('#### MLflow UI evidence')

md('''
Start the UI with `mlflow ui --backend-store-uri sqlite:///mlflow.db` and open http://127.0.0.1:5000. The screenshots below were taken from it.

**Experiment `eduassist_llmops_assignment`: three runs and their metrics**

![experiment runs and metrics](screenshots/mlflow_experiment.jpg)

**Run metrics for `prompt_v3`**

![run metrics](screenshots/metrics.jpg)

**Prompt Registry: versions 1, 2 and 3**

![prompt registry](screenshots/prompt_registry.jpg)

**Prompt diff, v3 against v2**

![prompt diff](screenshots/prompt_diff.jpg)

**`production` alias on version 3**

![production alias](screenshots/production_alias.jpg)
''')

md('#### Promotion justification')

md(f'''
I promoted **{best}** to the `production` alias.

**JSON validity (decisive).** EduAssist's product contract is strict JSON. {best} returned valid, schema-conforming JSON for {int(df[df.prompt_version == best].valid_json.sum())} of {int((df.prompt_version == best).sum())} requests ({S.loc[best, 'valid_json']:.0%}), while {' and '.join(others)} scored {S.loc[others, 'valid_json'].iloc[0]:.0%}{'' if S.loc[others, 'valid_json'].nunique() == 1 else ' or less'} because they answer in free-form prose and were never asked for JSON. Downstream code can parse {best} directly; the others would need a second model call or fragile text parsing.

**Latency.** {best} averages {S.loc[best, 'avg_latency_seconds']:.2f} s, against {lst(S.loc[others, 'avg_latency_seconds'], '{:.2f} s')} (p95 for {best} is {S.loc[best, 'p95_latency_seconds']:.2f} s). {speed_claim} It also writes the fewest tokens, which is the lasting reason to expect it to be quicker, since generation time grows with output length.

**Tokens.** {best} uses more input tokens ({S.loc[best, 'avg_input_tokens']:.1f} vs {lst(S.loc[others, 'avg_input_tokens'], '{:.1f}')}) because it carries the schema, but far fewer output tokens ({S.loc[best, 'avg_output_tokens']:.1f} vs {lst(S.loc[others, 'avg_output_tokens'], '{:.1f}')}). Output tokens are the expensive side ({price_ratio:.1f}x the input price), so the extra input is a good trade.

**Cost and scale.** Cost per request falls to {S.loc[best, 'avg_cost_per_request']:.6f} USD, against {lst(S.loc[others, 'avg_cost_per_request'], '{:.6f}')}, about {cut_cost:.0%} below v1. At 10,000 requests a month this projects to {proj.loc[best, 'projected_monthly_cost']:.2f} USD versus {lst(proj.loc[others, 'projected_monthly_cost'], '{:.2f}')} USD, and the gap widens linearly with volume: {scale.loc['v1', big] - scale.loc[best, big]:,.0f} USD a month at {config.SCALE_VOLUMES[big]:,} requests.

**Quality.** Answer correctness was not scored automatically in this experiment. {best} keeps the academic-assistant role and adds a confidence level and key points, which make answers easier for a human to review.

**Limitations.** Five questions is a small sample, latency depends on provider load at the time of the run, and the scale figures rest on stated assumptions. A larger evaluation set with quality scoring should confirm the choice before wide rollout. The alias makes rollback instant: pointing `production` back at v2 needs no code change.
''')

md('#### Why we built it this way')

md(f'''
| Decision | What | Why |
|---|---|---|
| Questions from our own modules | 5 questions across 5 categories | realistic mix of answer lengths, and not the generic set others will use |
| Same questions for all versions | 3 versions x 5 questions | differences come from the prompt, not the input |
| {R} interleaved repeats | {n_calls} requests in total | measures latency noise, and a slow period cannot favour one version |
| `temperature = 0` | fixed for every call | reduces run-to-run variation in length and wording |
| Token counts from `usage` | read from each API response | exact billing basis, not an estimate |
| p95 and target hit rate | beside the mean | a service is judged on its slow requests, not its average |
| Scale scenarios | 3 volumes, latency target, peak rate | a prompt decision is a cost and capacity decision at real volume |
| Strict `valid_json` | raw text must parse and match the schema | fences or prose would break a real parser |
| One prompt, three versions | `eduassist-qa-prompt` v1 to v3 | gives a real diff and one alias to move |
| Promote by alias | `production` -> best version | deploy and roll back without changing code |
| Key from environment | `.env`, git-ignored | secrets never reach the notebook or the ZIP |
''')

md('#### Reproducibility checks')

md('''
Confirm the saved log is complete and internally consistent: all requests succeeded, costs recompute exactly from the token counts and prices, and the alias points to the recommended version.
''')

code(f'''
log = pd.read_csv(CALLS_CSV)
recomputed = log["input_tokens"] / 1e6 * IN_PRICE + log["output_tokens"] / 1e6 * OUT_PRICE

print("Rows                     :", len(log), "(expected {n_calls})")
print("All requests succeeded   :", bool((log["status"] == "success").all()))
print("Costs match tokens x price:", bool((recomputed - log["total_cost"]).abs().max() < 1e-12))
print("Tokens add up            :", bool((log["input_tokens"] + log["output_tokens"] == log["total_tokens"]).all()))
print("Alias -> recommended     :", mlflow.genai.load_prompt(f"prompts:/{{PROMPT_NAME}}@production").version == versions[best])
print("Prompt versions in registry:", sorted(versions.values()))
''')

md(f'''
Result

* All {n_calls} rows succeeded, costs recompute exactly from tokens x price, and tokens add up. The `production` alias points to {best} and the registry holds exactly versions 1, 2 and 3.
* Token counts, costs and JSON validity depend on the prompt and should reproduce closely. Latency, p95 and target hit rate depend on provider load and will differ on a new run.
''')

nb.cells = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "assignment.ipynb")
print("wrote assignment.ipynb")
