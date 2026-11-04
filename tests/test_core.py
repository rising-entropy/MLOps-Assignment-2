import json

from src import config
from src.analysis import input_cost, output_cost
from src.prompts import PROMPTS, is_valid_json, render


def test_valid_json_accepts_schema():
    good = json.dumps({"answer": "x", "confidence": "high", "key_points": ["a"]})
    assert is_valid_json(good)


def test_valid_json_rejects_fences_prose_and_bad_schema():
    good = json.dumps({"answer": "x", "confidence": "high", "key_points": ["a"]})
    assert not is_valid_json("```json\n" + good + "\n```")
    assert not is_valid_json("Sure! " + good)
    assert not is_valid_json(json.dumps({"answer": "x", "confidence": "certain", "key_points": []}))
    assert not is_valid_json(json.dumps({"answer": "x", "confidence": "low"}))
    assert not is_valid_json("plain text answer")


def test_cost_formula():
    assert abs(input_cost(1_000_000) - config.INPUT_PRICE_PER_M) < 1e-12
    assert abs(output_cost(500_000) - config.OUTPUT_PRICE_PER_M / 2) < 1e-12


def test_render_replaces_placeholder():
    for p in PROMPTS.values():
        assert "{{question}}" not in render(p["template"], "Why?")
