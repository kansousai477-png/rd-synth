from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_inputs import load_adv_samples, resolve_adv_samples_path


class Stage3InputsTest(unittest.TestCase):
    def test_resolve_adv_samples_path_uses_stage2_default(self) -> None:
        out_dir = Path("/tmp/project_outputs")
        self.assertEqual(resolve_adv_samples_path("", out_dir), out_dir / "stage2" / "adv_samples.npz")

    def test_load_adv_samples_reads_stats_and_computes_norm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            adv_path = tmp_path / "adv_samples.npz"
            adv = np.asarray([[2.0, 6.0]], dtype=np.float32)
            mean = np.asarray([1.0, 2.0], dtype=np.float32)
            std = np.asarray([0.5, 2.0], dtype=np.float32)
            np.savez_compressed(
                adv_path,
                adv_pre=adv,
                adv_space=np.asarray("preprocessed"),
                ben_stats_mean=mean,
                ben_stats_std=std,
                feature_names=np.asarray(["a", "b"]),
            )

            loaded = load_adv_samples(
                adv_path,
                project_out_dir=tmp_path / "outputs",
                current_feature_names=["a", "b"],
                expected_feature_dim=2,
            )

        self.assertTrue(loaded.loaded)
        self.assertEqual(loaded.adv_space, "preprocessed")
        np.testing.assert_allclose(loaded.adv, adv)
        np.testing.assert_allclose(loaded.adv_norm, np.asarray([[2.0, 2.0]], dtype=np.float64))

    def test_load_adv_samples_can_recover_denorm_stats_from_stage2_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            adv_path = tmp_path / "adv_samples.npz"
            np.savez_compressed(
                adv_path,
                adv_pre=np.asarray([[1.0, 2.0]], dtype=np.float32),
                adv_space=np.asarray("benign_norm"),
                feature_names=np.asarray(["a", "b"]),
            )
            stage2_dir = tmp_path / "outputs" / "stage2"
            stage2_dir.mkdir(parents=True, exist_ok=True)
            (stage2_dir / "stage2.pt").write_bytes(b"placeholder")

            load_state = Mock(
                return_value={
                    "ben_stats": {
                        "denorm_mean": np.asarray([10.0, 20.0], dtype=np.float32),
                        "denorm_std": np.asarray([2.0, 3.0], dtype=np.float32),
                    }
                }
            )

            loaded = load_adv_samples(
                adv_path,
                project_out_dir=tmp_path / "outputs",
                current_feature_names=["a", "b"],
                expected_feature_dim=2,
                load_torch_state_fn=load_state,
            )

        np.testing.assert_allclose(loaded.adv, np.asarray([[12.0, 26.0]], dtype=np.float32))
        np.testing.assert_allclose(loaded.adv_norm, np.asarray([[1.0, 2.0]], dtype=np.float64))

    def test_load_adv_samples_rejects_feature_name_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            adv_path = tmp_path / "adv_samples.npz"
            np.savez_compressed(
                adv_path,
                adv_pre=np.asarray([[1.0, 2.0]], dtype=np.float32),
                feature_names=np.asarray(["x", "y"]),
            )
            warnings: list[str] = []

            loaded = load_adv_samples(
                adv_path,
                project_out_dir=tmp_path / "outputs",
                current_feature_names=["a", "b"],
                expected_feature_dim=2,
                warn_fn=warnings.append,
            )

        self.assertFalse(loaded.loaded)
        self.assertEqual(loaded.count, 0)
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
