from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.data import (
    DataContext,
    _data_cache_dir,
    _data_cache_path,
    _require_fraction,
    load_data_context,
)
from rdsynth.utils.config import maybe_int


class PipelineDataTest(unittest.TestCase):
    def test_parsing_helpers_validate_values(self) -> None:
        self.assertIsNone(maybe_int("", "data.max_rows"))
        self.assertEqual(maybe_int("7", "data.max_rows"), 7)
        self.assertEqual(maybe_int("0", "data.max_rows"), 0)
        self.assertAlmostEqual(_require_fraction("0.25", "data.test_frac"), 0.25)

        with self.assertRaises(ValueError):
            maybe_int("0", "data.max_rows", positive_only=True)
        with self.assertRaises(ValueError):
            maybe_int("abc", "data.max_rows")
        with self.assertRaises(ValueError):
            _require_fraction("1.0", "data.test_frac")
        with self.assertRaises(ValueError):
            _require_fraction("bad", "data.test_frac")

    def test_data_cache_dir_prefers_explicit_cache_dir_then_project_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            explicit = _data_cache_dir({"data": {"cache_dir": str(root / "explicit")}})
            self.assertEqual(explicit, (root / "explicit").resolve())

            inferred = _data_cache_dir({"data": {}, "project": {"runtime": {"cwd": str(root)}}})
            self.assertEqual(inferred, (root / ".cache" / "rdsynth_data_context").resolve())

    def test_load_data_context_populates_and_reuses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = {
                "project": {"runtime": {"cwd": str(root)}},
                "data": {
                    "dataset": "toy",
                    "test_frac": 0.2,
                    "val_frac": 0.1,
                    "cache_dir": str(root / "cache"),
                },
            }
            features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
            labels = np.array([0, 1, 0], dtype=np.int64)
            bundle = SimpleNamespace(feature_names=["f1", "f2"], x_train=np.ones((2, 2), dtype=np.float32))

            with (
                patch(
                    "rdsynth.pipeline.data.resolve_dataset_profile",
                    return_value=SimpleNamespace(
                        csv_path="data.csv",
                        csv_dir=None,
                        csv_glob=None,
                        label_col="label",
                        label_source=None,
                        benign_labels=("benign",),
                        drop_cols=[],
                    ),
                ),
                patch("rdsynth.pipeline.data.load_csv_dataset", return_value=(features, labels)) as load_csv_dataset,
                patch("rdsynth.pipeline.data.prepare_splits", return_value=bundle) as prepare_splits,
            ):
                context = load_data_context(cfg, seed=7)

            self.assertEqual(context.features.shape, (3, 2))
            np.testing.assert_array_equal(context.labels, labels)
            self.assertIs(context.bundle, bundle)
            load_csv_dataset.assert_called_once()
            prepare_splits.assert_called_once()

            cache_path = _data_cache_path(cfg, 7)
            self.assertTrue(cache_path.exists())
            artifact_dir = Path(cfg["data"]["cache_dir"]) / f"{cache_path.stem}_artifacts"
            self.assertTrue((artifact_dir / "metadata.json").exists())
            self.assertTrue((artifact_dir / "raw_dataset.npz").exists())
            self.assertTrue((artifact_dir / "split_arrays.npz").exists())
            self.assertTrue((artifact_dir / "preprocess_state.pkl").exists())

            with (
                patch(
                    "rdsynth.pipeline.data.load_csv_dataset", side_effect=AssertionError("cache should short-circuit")
                ),
                patch("rdsynth.pipeline.data.prepare_splits", side_effect=AssertionError("cache should short-circuit")),
            ):
                cached_context = load_data_context(cfg, seed=7)
            self.assertIsInstance(cached_context, DataContext)
            self.assertEqual(list(cached_context.features.columns), ["f1", "f2"])

    def test_load_data_context_ignores_non_context_cache_and_validates_split_sum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cfg = {
                "project": {"runtime": {"cwd": str(root)}},
                "data": {
                    "dataset": "toy",
                    "test_frac": 0.6,
                    "val_frac": 0.5,
                    "cache_dir": str(root / "cache"),
                },
            }
            cache_path = _data_cache_path(cfg, 5)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as handle:
                pickle.dump({"not": "a context"}, handle)

            with self.assertRaises(ValueError):
                load_data_context(cfg, seed=5)


if __name__ == "__main__":
    unittest.main()
