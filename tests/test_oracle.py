from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.stages.oracle import (
    _train_extra_trees_oracle,
    _train_rf_oracle,
    predict_sklearn_probs,
    restore_safe_oracle_model,
    serialize_safe_oracle_model,
)


class _BinaryDecisionModel:
    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x[:, 0], dtype=np.float64)


class _MultiDecisionModel:
    def decision_function(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(
            np.stack([x[:, 0], x[:, 1], x[:, 0] - x[:, 1]], axis=1),
            dtype=np.float64,
        )


class _SafeLinearModel:
    coef_ = np.asarray([[1.0, -1.0]], dtype=np.float64)
    intercept_ = np.asarray([0.25], dtype=np.float64)
    classes_ = np.asarray([0, 1], dtype=np.int64)


class OracleProbUtilsTest(unittest.TestCase):
    def test_predict_sklearn_probs_binary_decision_function(self) -> None:
        model = _BinaryDecisionModel()
        probs = predict_sklearn_probs(model, np.array([[0.0], [2.0]], dtype=np.float64))
        self.assertEqual(probs.shape, (2, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1.0e-6)

    def test_predict_sklearn_probs_multiclass_decision_function(self) -> None:
        model = _MultiDecisionModel()
        probs = predict_sklearn_probs(model, np.array([[1.0, 0.5], [0.2, 2.0]], dtype=np.float64))
        self.assertEqual(probs.shape, (2, 3))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1.0e-6)

    def test_safe_linear_oracle_roundtrip_preserves_predictions(self) -> None:
        payload = serialize_safe_oracle_model(_SafeLinearModel(), "logistic")
        restored = restore_safe_oracle_model(payload or {}, "logistic")
        x = np.asarray([[0.0, 0.0], [2.0, 0.5]], dtype=np.float64)
        np.testing.assert_array_equal(restored.predict(x), np.asarray([1, 1], dtype=np.int64))
        probs = restored.predict_proba(x)
        self.assertEqual(probs.shape, (2, 2))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), atol=1.0e-6)

    def test_train_rf_oracle_uses_single_process_sklearn_backend(self) -> None:
        x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        model = type("DummyRF", (), {"fit": lambda self, *_args, **_kwargs: self})()
        with patch("rdsynth.stages.oracle.RandomForestClassifier", return_value=model) as ctor:
            trained = _train_rf_oracle(x, y, class_weight="balanced", random_state=7)
        self.assertIs(trained, model)
        self.assertEqual(ctor.call_args.kwargs["n_jobs"], 1)

    def test_train_extra_trees_oracle_uses_single_process_sklearn_backend(self) -> None:
        x = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        model = type("DummyET", (), {"fit": lambda self, *_args, **_kwargs: self})()
        with patch("rdsynth.stages.oracle.ExtraTreesClassifier", return_value=model) as ctor:
            trained = _train_extra_trees_oracle(x, y, class_weight="balanced", random_state=11)
        self.assertIs(trained, model)
        self.assertEqual(ctor.call_args.kwargs["n_jobs"], 1)


if __name__ == "__main__":
    unittest.main()
