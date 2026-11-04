"""The five evaluation questions and the three prompt versions (verbatim from the assignment)."""
import json

# Questions come from the MTech modules (ML, deep learning, MLOps, LLMOps) and mimic a real support desk:
# they differ in kind and expected answer length so latency, token and cost behaviour is not uniform.
QUESTIONS = {
    "Q1": "What does the F1 score measure, and when should it be preferred over accuracy?",
    "Q2": "A neural network's training loss keeps falling but its validation loss starts rising after epoch 12. "
          "What is happening, and how can it be fixed?",
    "Q3": "A fraud-detection model had 95% accuracy at launch but performs much worse six months later. "
          "What are data drift and concept drift, and what should the team monitor to catch them?",
    "Q4": "Why do LSTMs handle long sequences better than simple RNNs?",
    "Q5": "Why does the cost of an LLM application grow with prompt length, and how can a team keep it "
          "under control at scale?",
}

QUESTION_META = {
    "Q1": dict(category="Definition"),
    "Q2": dict(category="Debugging"),
    "Q3": dict(category="MLOps scenario"),
    "Q4": dict(category="Architecture"),
    "Q5": dict(category="LLMOps economics"),
}

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
    "v1": {"template": PROMPT_V1, "name": "Basic",
           "commit": "v1 Basic: bare instruction, no role, no format or audience guidance (baseline)."},
    "v2": {"template": PROMPT_V2, "name": "Structured",
           "commit": "v2 Structured: adds academic-assistant role and clear/concise/simple-language constraints."},
    "v3": {"template": PROMPT_V3, "name": "JSON Output",
           "commit": "v3 JSON Output: adds strict JSON-only output with answer/confidence/key_points schema."},
}


def render(template: str, question: str) -> str:
    return template.replace("{{question}}", question)


def is_valid_json(text: str) -> bool:
    """Strict check: the raw response must parse as JSON and match the required schema.
    Markdown fences or surrounding prose therefore count as invalid, as the prompt forbids them."""
    try:
        obj = json.loads(text)
    except (TypeError, ValueError):
        return False
    return (
        isinstance(obj, dict)
        and set(obj) == {"answer", "confidence", "key_points"}
        and isinstance(obj["answer"], str)
        and obj["confidence"] in {"high", "medium", "low"}
        and isinstance(obj["key_points"], list)
        and all(isinstance(k, str) for k in obj["key_points"])
    )
