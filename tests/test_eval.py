import json

import pytest

from laya_onnx.eval import (
    expected_calibration_error,
    format_markdown,
    load_items,
    percentile,
    run_eval,
    score_answer,
)


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 4.0
    assert percentile([], 0.5) is None


def test_ece_perfect_and_worst():
    # Confidence equals accuracy exactly -> ECE 0.
    assert expected_calibration_error([1.0, 1.0, 0.0, 0.0], [True, True, False, False]) == 0.0
    # Always fully confident, always wrong -> ECE 1.
    assert expected_calibration_error([1.0, 1.0], [False, False]) == 1.0
    # Overconfident on half the items -> 0.1.
    assert expected_calibration_error([0.9, 0.9, 0.1, 0.1], [True, True, False, False]) == pytest.approx(0.1)
    assert expected_calibration_error([], []) is None


def test_score_answer_types():
    assert score_answer("choice", {"choice": "billing"}, "billing") == (True, None)
    assert score_answer("choice", {"choice": "sales"}, "billing") == (False, None)
    assert score_answer("noul", {"noul": 0.8}, True) == (True, None)
    assert score_answer("noul", {"noul": 0.2}, True) == (False, None)
    correct, error = score_answer("score", {"score": 1.6}, 2)
    assert correct is True and error == pytest.approx(0.4)
    assert score_answer("score", {"score": 0.0}, 2)[0] is False
    assert score_answer("choice", None, "billing") == (None, None)


class StubAgent:
    def predict(self, state, questions):
        return {
            "answers": {
                "q1": {"type": "choice", "choice": "yes", "confidence": 1.0},
                "q2": {"type": "noul", "noul": 0.1, "confidence": 0.9},
            }
        }


def test_run_eval_reports_types_and_overall():
    items = [
        {
            "state": "s",
            "questions": {
                "q1": {"type": "choice", "instructions": "?", "criteria": {"yes": "y", "no": "n"}},
                "q2": {"type": "noul", "instructions": "?"},
            },
            "gold": {"q1": "yes", "q2": True},
        }
    ]
    results = run_eval(StubAgent(), items)
    assert results["overall"]["items"] == 1
    assert results["overall"]["questions"] == 2
    assert results["overall"]["accuracy"] == 0.5
    assert set(results["by_type"]) == {"choice", "noul"}
    assert results["by_type"]["choice"]["accuracy"] == 1.0
    assert results["by_type"]["noul"]["accuracy"] == 0.0
    markdown = format_markdown(results)
    assert "| choice |" in markdown and "accuracy" in markdown


def test_load_items_roundtrip(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text(json.dumps({"state": "s", "questions": {}, "gold": {}}) + "\n", encoding="utf-8")
    items = load_items(path)
    assert len(items) == 1 and items[0]["state"] == "s"
