from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.preprocessing import PowerTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.data.nb15 import prepare_splits
from rdsynth.pipeline.preprocessing import DatasetPreprocessor
from rdsynth.utils.feature_preprocessing import (
    apply_feature_transform,
    fit_feature_transform_plan,
    invert_feature_transform,
)


class DatasetPreprocessorTest(unittest.TestCase):
    def setUp(self) -> None:
        raw = np.array(
            [
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 4.0],
                [3.0, 8.0],
                [4.0, 16.0],
                [5.0, 32.0],
            ],
            dtype=np.float64,
        )
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        pt.fit(raw)
        transformed = pt.transform(raw)
        scaler = StandardScaler().fit(transformed)
        bundle = SimpleNamespace(
            scaler=scaler,
            power_transformer=pt,
            feature_names=["f1", "f2"],
            impute_strategy="zero",
            impute_values=np.zeros(2, dtype=np.float64),
            winsorize_lower=None,
            winsorize_upper=None,
            log1p_mask=None,
            log1p_shift=None,
        )
        self.preprocessor = DatasetPreprocessor.from_bundle(bundle)
        self.raw = raw

    def test_round_trip_transform(self) -> None:
        transformed = self.preprocessor.transform(self.raw)
        recovered = self.preprocessor.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, self.raw, atol=1.0e-5)

    def test_transform_accepts_single_row(self) -> None:
        transformed = self.preprocessor.transform(self.raw[0])
        self.assertEqual(transformed.shape, (2,))
        recovered = self.preprocessor.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, self.raw[0], atol=1.0e-5)

    def test_feature_validation_rejects_wrong_width(self) -> None:
        with self.assertRaises(ValueError):
            self.preprocessor.transform(np.ones((2, 3), dtype=np.float64))

    def test_feature_mean_rejects_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            self.preprocessor.feature_mean(np.empty((0, 2), dtype=np.float64))

    def test_transform_uses_bundle_median_imputation(self) -> None:
        bundle = SimpleNamespace(
            scaler=None,
            power_transformer=None,
            feature_names=["f1", "f2"],
            impute_strategy="median",
            impute_values=np.array([10.0, 20.0], dtype=np.float64),
            winsorize_lower=None,
            winsorize_upper=None,
            log1p_mask=None,
            log1p_shift=None,
        )
        preprocessor = DatasetPreprocessor.from_bundle(bundle)
        transformed = preprocessor.transform(np.array([[np.nan, 2.0]], dtype=np.float64))
        np.testing.assert_allclose(transformed, np.array([[10.0, 2.0]], dtype=np.float64))

    def test_transform_applies_train_fitted_winsorization(self) -> None:
        bundle = SimpleNamespace(
            scaler=None,
            power_transformer=None,
            feature_names=["f1", "f2"],
            impute_strategy="zero",
            impute_values=np.zeros(2, dtype=np.float64),
            winsorize_lower=np.array([0.0, 1.0], dtype=np.float64),
            winsorize_upper=np.array([5.0, 3.0], dtype=np.float64),
            log1p_mask=None,
            log1p_shift=None,
        )
        preprocessor = DatasetPreprocessor.from_bundle(bundle)
        transformed = preprocessor.transform(np.array([[-2.0, 10.0]], dtype=np.float64))
        np.testing.assert_allclose(transformed, np.array([[0.0, 3.0]], dtype=np.float64))

    def test_transform_and_inverse_apply_log1p_consistently(self) -> None:
        bundle = SimpleNamespace(
            scaler=None,
            power_transformer=None,
            feature_names=["f1", "f2"],
            impute_strategy="zero",
            impute_values=np.zeros(2, dtype=np.float64),
            winsorize_lower=None,
            winsorize_upper=None,
            log1p_mask=np.array([True, False]),
            log1p_shift=np.array([2.0, 0.0], dtype=np.float64),
        )
        preprocessor = DatasetPreprocessor.from_bundle(bundle)
        raw = np.array([[-1.0, 3.0]], dtype=np.float64)
        transformed = preprocessor.transform(raw)
        np.testing.assert_allclose(transformed, np.array([[np.log1p(1.0), 3.0]], dtype=np.float64))
        recovered = preprocessor.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, raw, atol=1.0e-6)

    def test_prepare_splits_persists_train_fitted_preprocessing_stats(self) -> None:
        features = np.array(
            [
                [0.0, 1.0],
                [1.0, np.nan],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 1000.0],
                [5.0, 4.0],
                [6.0, 5.0],
                [7.0, 6.0],
            ],
            dtype=np.float64,
        )
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        import pandas as pd

        bundle = prepare_splits(
            pd.DataFrame(features, columns=["f1", "f2"]),
            labels,
            None,
            test_frac=0.25,
            val_frac=0.25,
            seed=7,
            scaler_type="none",
            power_transform=False,
            impute_strategy="median",
            winsorize_quantile=0.2,
        )
        self.assertIsNotNone(bundle.impute_values)
        self.assertIsNotNone(bundle.winsorize_lower)
        self.assertIsNotNone(bundle.winsorize_upper)
        self.assertEqual(bundle.impute_strategy, "median")

    def test_prepare_splits_stratifies_raw_attack_labels_when_possible(self) -> None:
        import pandas as pd

        raw_labels = np.asarray(["BENIGN"] * 12 + ["AttackA"] * 12 + ["AttackB"] * 12, dtype=object)
        labels = np.asarray([0] * 12 + [1] * 24, dtype=np.int64)
        features = pd.DataFrame(
            {
                "f1": np.arange(raw_labels.shape[0], dtype=np.float64),
                "f2": np.arange(raw_labels.shape[0], dtype=np.float64) * 2.0,
            }
        )

        bundle = prepare_splits(
            features,
            labels,
            raw_labels,
            test_frac=0.25,
            val_frac=0.25,
            seed=11,
            scaler_type="none",
            power_transform=False,
            impute_strategy="zero",
        )

        self.assertEqual(set(bundle.raw_y_test.tolist()), {"BENIGN", "AttackA", "AttackB"})
        self.assertEqual(set(bundle.raw_y_val.tolist()), {"BENIGN", "AttackA", "AttackB"})
        self.assertEqual(set(bundle.raw_y_train.tolist()), {"BENIGN", "AttackA", "AttackB"})

    def test_type_aware_transform_separates_binary_discrete_and_continuous(self) -> None:
        raw = np.array(
            [
                [80.0, 1.0, 10.0, 100.0],
                [443.0, 0.0, 15.0, 120.0],
                [8080.0, 1.0, 20.0, 150.0],
                [53.0, 0.0, 25.0, 180.0],
            ],
            dtype=np.float64,
        )
        plan = fit_feature_transform_plan(
            x_train=raw,
            feature_names=["Dst Port", "ACK Flag Count", "Flow IAT Mean", "Pkt Len Mean"],
            scaler_type="standard",
            power_transform=True,
            discrete_scaling="minmax",
        )
        self.assertEqual(plan.binary_idx.tolist(), [1])
        self.assertEqual(plan.discrete_idx.tolist(), [0])
        self.assertEqual(plan.continuous_idx.tolist(), [2, 3])
        transformed = apply_feature_transform(raw, plan)
        recovered = invert_feature_transform(transformed, plan)
        np.testing.assert_allclose(recovered, raw, atol=1.0e-5)
        np.testing.assert_allclose(transformed[:, 1], raw[:, 1], atol=1.0e-8)

    def test_prepare_splits_type_aware_bundle_round_trip(self) -> None:
        import pandas as pd

        features = pd.DataFrame(
            {
                "Dst Port": [80.0, 443.0, 8080.0, 53.0, 22.0, 25.0, 110.0, 995.0],
                "ACK Flag Count": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                "Flow IAT Mean": [10.0, 12.0, 15.0, 14.0, 11.0, 13.0, 16.0, 18.0],
                "Packet Length Mean": [100.0, 120.0, 140.0, 110.0, 115.0, 130.0, 150.0, 170.0],
            }
        )
        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        bundle = prepare_splits(
            features,
            labels,
            None,
            test_frac=0.25,
            val_frac=0.25,
            seed=5,
            scaler_type="robust",
            power_transform=True,
            impute_strategy="median",
            winsorize_quantile=0.1,
            type_aware_transform=True,
            discrete_scaling="minmax",
        )
        self.assertIsNotNone(bundle.feature_transform_plan)
        preprocessor = DatasetPreprocessor.from_bundle(bundle)
        recovered = preprocessor.inverse_transform(bundle.x_train[:2])
        self.assertEqual(recovered.shape[1], features.shape[1])

    def test_type_aware_transform_treats_low_cardinality_integer_counts_as_discrete(self) -> None:
        raw = np.array(
            [
                [8.0, 10.0],
                [20.0, 12.0],
                [32.0, 14.0],
                [32.0, 16.0],
                [20.0, 18.0],
                [8.0, 20.0],
            ],
            dtype=np.float64,
        )
        plan = fit_feature_transform_plan(
            x_train=raw,
            feature_names=["Fwd Seg Size Min", "Flow IAT Mean"],
            scaler_type="robust",
            power_transform=True,
            discrete_scaling="minmax",
        )
        self.assertIn(0, plan.discrete_idx.tolist())
        self.assertNotIn(0, plan.continuous_power_idx.tolist())
        transformed = apply_feature_transform(raw, plan)
        self.assertTrue(np.all(np.isfinite(transformed[:, 0])))
        self.assertLess(float(np.max(np.abs(transformed[:, 0]))), 2.0)
        recovered = invert_feature_transform(transformed, plan)
        np.testing.assert_allclose(recovered, raw, atol=1.0e-5)

    def test_schema_overrides_force_feature_family_assignment(self) -> None:
        raw = np.array(
            [
                [8.0, 10.0],
                [20.0, 12.0],
                [28.0, 14.0],
                [32.0, 16.0],
            ],
            dtype=np.float64,
        )
        plan = fit_feature_transform_plan(
            x_train=raw,
            feature_names=["Fwd Seg Size Min", "Flow IAT Mean"],
            scaler_type="robust",
            power_transform=True,
            discrete_scaling="minmax",
            schema_overrides={"discrete": ["Fwd Seg Size Min"]},
        )
        self.assertIn(0, plan.discrete_idx.tolist())
        self.assertNotIn(0, plan.continuous_power_idx.tolist())


if __name__ == "__main__":
    unittest.main()
