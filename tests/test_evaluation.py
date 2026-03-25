"""Unit tests for src/evaluation.py"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
from sklearn.pipeline import Pipeline

# ── Helpers ──────────────────────────────────────────────────────────────────

EXPECTED_METRIC_KEYS = {"accuracy", "f1", "precision", "recall", "roc_auc"}


def _make_mock_pipe(y_pred, y_proba):
    """Return a mock Pipeline that produces predetermined predictions."""
    pipe = MagicMock(spec=Pipeline)
    pipe.predict.return_value = np.array(y_pred)
    pipe.predict_proba.return_value = np.column_stack([1 - np.array(y_proba), np.array(y_proba)])
    return pipe


# ── Tests ────────────────────────────────────────────────────────────────────

class TestEvaluateMetricKeys:
    """evaluate_and_save should return a dict with the expected metric keys."""

    def test_metrics_dict_contains_expected_keys(self, tmp_path):
        from src.evaluation import evaluate_and_save

        # Perfectly separable dataset: first 5 positive, last 5 negative
        y_val = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        y_proba = [0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.25, 0.2, 0.15, 0.1]

        pipe = _make_mock_pipe(y_pred, y_proba)

        with patch("src.evaluation.EVALUATIONS_DIR", tmp_path):
            metrics = evaluate_and_save(
                pipe,
                X_val=None,
                y_val=y_val,
                model_date="2005-12-31",
                eval_window_date="2006-12-31",
            )

        assert EXPECTED_METRIC_KEYS.issubset(metrics.keys()), (
            f"Missing keys: {EXPECTED_METRIC_KEYS - metrics.keys()}"
        )

    def test_metrics_dict_contains_date_metadata(self, tmp_path):
        from src.evaluation import evaluate_and_save

        y_val = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        y_proba = [0.9, 0.1, 0.8, 0.2]

        pipe = _make_mock_pipe(y_pred, y_proba)

        with patch("src.evaluation.EVALUATIONS_DIR", tmp_path):
            metrics = evaluate_and_save(
                pipe,
                X_val=None,
                y_val=y_val,
                model_date="2005-12-31",
                eval_window_date="2006-12-31",
            )

        assert metrics["model_date"] == "2005-12-31"
        assert metrics["eval_window_date"] == "2006-12-31"

    def test_metrics_values_are_numeric(self, tmp_path):
        from src.evaluation import evaluate_and_save

        y_val = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 1]
        y_proba = [0.8, 0.6, 0.3, 0.7]

        pipe = _make_mock_pipe(y_pred, y_proba)

        with patch("src.evaluation.EVALUATIONS_DIR", tmp_path):
            metrics = evaluate_and_save(
                pipe,
                X_val=None,
                y_val=y_val,
                model_date="2005-12-31",
                eval_window_date="2006-12-31",
            )

        for key in EXPECTED_METRIC_KEYS:
            assert isinstance(metrics[key], (int, float)), (
                f"Expected numeric value for '{key}', got {type(metrics[key])}"
            )

    def test_metrics_are_rounded_to_4_decimal_places(self, tmp_path):
        from src.evaluation import evaluate_and_save

        y_val = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 0, 1]
        y_proba = [0.9, 0.4, 0.2, 0.1, 0.8]

        pipe = _make_mock_pipe(y_pred, y_proba)

        with patch("src.evaluation.EVALUATIONS_DIR", tmp_path):
            metrics = evaluate_and_save(
                pipe,
                X_val=None,
                y_val=y_val,
                model_date="2005-12-31",
                eval_window_date="2006-12-31",
            )

        for key in EXPECTED_METRIC_KEYS:
            val = metrics[key]
            # Check it has at most 4 decimal places
            assert round(val, 4) == val, f"'{key}' not rounded to 4dp: {val}"


class TestTriviallyPredictableDataset:
    """Test evaluation with a perfectly-separated dataset (all same class in pred)."""

    def test_all_class_1_predictions_with_correct_labels(self, tmp_path):
        from src.evaluation import evaluate_and_save

        # All actual labels are 1, predictions are all 1 → perfect precision/recall
        y_val = [1, 1, 1, 1, 1]
        y_pred = [1, 1, 1, 1, 1]
        # For roc_auc, we need both classes; add class 0 instance with low proba
        y_val = [1, 1, 1, 1, 0]
        y_pred = [1, 1, 1, 1, 0]
        y_proba = [0.9, 0.85, 0.8, 0.75, 0.1]

        pipe = _make_mock_pipe(y_pred, y_proba)

        with patch("src.evaluation.EVALUATIONS_DIR", tmp_path):
            metrics = evaluate_and_save(
                pipe,
                X_val=None,
                y_val=y_val,
                model_date="2005-12-31",
                eval_window_date="2006-12-31",
            )

        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_metrics_file_written_to_disk(self, tmp_path):
        from src.evaluation import evaluate_and_save

        y_val = [1, 0, 1, 0]
        y_pred = [1, 0, 1, 0]
        y_proba = [0.9, 0.1, 0.8, 0.2]

        pipe = _make_mock_pipe(y_pred, y_proba)

        with patch("src.evaluation.EVALUATIONS_DIR", tmp_path):
            evaluate_and_save(
                pipe,
                X_val=None,
                y_val=y_val,
                model_date="2005-12-31",
                eval_window_date="2006-12-31",
            )

        expected_file = tmp_path / "eval_model_2005-12-31_on_2006-12-31.json"
        assert expected_file.exists(), f"Expected file {expected_file} to exist"

        with open(expected_file) as f:
            saved = json.load(f)

        assert EXPECTED_METRIC_KEYS.issubset(saved.keys())
