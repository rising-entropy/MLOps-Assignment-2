"""Thin LLM client that returns timing, status and token usage for every request."""
import json
import time
from datetime import datetime, timezone

from . import config


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _dry_run_response(prompt: str) -> tuple[str, int, int]:
    """Synthetic response used only for pipeline testing (DRY_RUN=1)."""
    time.sleep(0.05)
    if "ONLY valid JSON" in prompt:
        text = json.dumps({"answer": "dry-run answer", "confidence": "medium", "key_points": ["a", "b"]})
    else:
        text = "dry-run answer"
    return text, max(1, len(prompt) // 4), max(1, len(text) // 4)


def call_llm(prompt: str, retries: int = 4) -> dict:
    """Send one prompt; return text, usage and latency. Latency is that of the final (successful) attempt only."""
    if config.DRY_RUN:
        start, t0 = _now(), time.perf_counter()
        text, tin, tout = _dry_run_response(prompt)
        return dict(model="dry-run-mock", start_time=start, end_time=_now(),
                    latency_seconds=time.perf_counter() - t0, status="success", response_text=text,
                    input_tokens=tin, output_tokens=tout, total_tokens=tin + tout, error="")

    if not config.API_KEY:
        raise RuntimeError("No API key found. Set LLM_API_KEY (or OPENAI_API_KEY) in your environment or .env file.")
    from openai import OpenAI

    client = OpenAI(api_key=config.API_KEY, base_url=config.BASE_URL)
    last_err = ""
    for attempt in range(retries + 1):
        start, t0 = _now(), time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=config.MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_OUTPUT_TOKENS,
            )
            latency = time.perf_counter() - t0
            usage = resp.usage
            return dict(model=resp.model or config.MODEL, start_time=start, end_time=_now(),
                        latency_seconds=latency, status="success",
                        response_text=resp.choices[0].message.content or "",
                        input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens, error="")
        except Exception as exc:  # noqa: BLE001 - record any API failure in the log
            last_err = f"{type(exc).__name__}: {exc}"
            failed = dict(model=config.MODEL, start_time=start, end_time=_now(),
                          latency_seconds=time.perf_counter() - t0, status="error", response_text="",
                          input_tokens=0, output_tokens=0, total_tokens=0, error=last_err)
            time.sleep(2 ** attempt)
    return failed
