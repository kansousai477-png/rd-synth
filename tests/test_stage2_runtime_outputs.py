from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage2_outputs import save_stage2_manifest, save_stage2_state
from rdsynth.pipeline.stage2_runtime import (
    Stage2DistributionMetricsResult,
    compute_stage2_distribution_metrics,
    record_sample_runtime,
    record_sample_statistics,
    update_selected_candidate_metrics,
)
from rdsynth.utils.checkpoints import load_torch_state


class _DummyModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


class _DummyDDPMBundle:
    def __init__(self) -> None:
        self.denoiser = _DummyModule()
        self.encoder = _DummyModule()
        self.decoder = _DummyModule()
        self.groups = {"g": [0, 1]}
        self.ben_stats = {"min": np.array([-1.0, -1.0]), "max": np.array([1.0, 1.0])}
        self.latent_mean = torch.zeros(2)
        self.latent_std = torch.ones(2)
        self.predict_x0 = True
        self.x0_head_tanh = True
        self.best_epoch = 1
        self.best_score = 0.5
        self.train_log = [{"epoch": 1, "loss": 0.1}]


class _DummySettings:
    save_samples = True
    save_intermediate_results = False


class Stage2RuntimeOutputsTest(unittest.TestCase):
    def test_sample_helpers_record_metrics(self) -> None:
        metrics = {}
        nan_rate, inf_rate = record_sample_statistics(
            metrics_payload=metrics,
            sample_for_stats=np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
            selected_alpha=0.2,
        )
        self.assertEqual(nan_rate, 0.0)
        self.assertEqual(inf_rate, 0.0)
        record_sample_runtime(
            metrics_payload=metrics,
            sample_count=2,
            runtime_sec=0.5,
            pull_alpha=0.1,
            pull_k=2,
            moment_alpha=0.2,
            moment_std_floor=1.0e-3,
            post_clip_norm_range=True,
        )
        update_selected_candidate_metrics(metrics, {"selection_score": 1.5})
        self.assertIn("sample_generation_samples_per_sec", metrics)
        self.assertEqual(metrics["selected_candidate_selection_score"], 1.5)

    def test_compute_distribution_metrics_updates_payload(self) -> None:
        metrics = {
            "sample_generation_time_sec": 1.0,
            "asr_surrogate": 0.5,
            "mal_benign_rate": 0.0,
            "adv_prob_malicious_mean": 0.2,
        }
        cfg = {"stage2": {"metrics_max_real": 16, "metrics_max_gen": 16}}
        x_ben = np.asarray([[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
        x_adv = np.asarray([[0.1, 0.1], [0.9, 0.9], [0.6, 0.4]], dtype=np.float32)
        x_mal = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.7, 0.3]], dtype=np.float32)
        result = compute_stage2_distribution_metrics(
            metrics_payload=metrics,
            cfg=cfg,
            feature_names=["f0", "f1"],
            seed=7,
            x_ben_norm=x_ben,
            x_adv_norm=x_adv,
            x_adv_pre=x_adv,
            x_mal_pre=x_mal,
            norm_bounds_min=np.zeros(2, dtype=np.float32),
            norm_bounds_max=np.ones(2, dtype=np.float32),
            norm_nonneg=np.zeros(2, dtype=bool),
        )
        self.assertIsInstance(result, Stage2DistributionMetricsResult)
        self.assertIn("norm_FFD", metrics)
        self.assertIsNotNone(result.metrics_norm)

    def test_save_stage2_state_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            save_stage2_state(
                stage2_mode="latent_diffusion",
                generator_backbone="ddpm",
                diffusion_bundle=_DummyDDPMBundle(),
                feature_names=["f0", "f1"],
                out_path=out_dir / "stage2.pt",
            )
            state_payload = load_torch_state(out_dir / "stage2.pt", map_location="cpu")
            save_stage2_manifest(
                out_dir=out_dir,
                config_path="configs/demo.yaml",
                oracle_name="default",
                x_train=np.zeros((4, 2), dtype=np.float32),
                x_ben=np.zeros((2, 2), dtype=np.float32),
                x_mal=np.zeros((2, 2), dtype=np.float32),
                stage2_mode="latent_diffusion",
                train_log=[{"epoch": 1}],
                metrics_payload={"pareto_path": str(out_dir / "pareto.csv")},
                settings=_DummySettings(),
                x_adv_pre=np.zeros((2, 2), dtype=np.float32),
                x_adv_norm=np.zeros((2, 2), dtype=np.float32),
                x_ben_pre=np.zeros((2, 2), dtype=np.float32),
                x_mal_pre=np.zeros((2, 2), dtype=np.float32),
            )
            self.assertTrue((out_dir / "stage2.pt").exists())
            self.assertTrue((out_dir / "stage2.pt.sha256").exists())
            self.assertTrue((out_dir / "manifest.json").exists())
            self.assertEqual(state_payload["artifact_version"], 1)

    def test_save_stage2_manifest_omits_missing_eval_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            save_stage2_manifest(
                out_dir=out_dir,
                config_path="configs/demo.yaml",
                oracle_name="default",
                x_train=np.zeros((4, 2), dtype=np.float32),
                x_ben=np.zeros((2, 2), dtype=np.float32),
                x_mal=np.zeros((2, 2), dtype=np.float32),
                stage2_mode="editor",
                train_log=None,
                metrics_payload={},
                settings=_DummySettings(),
                x_adv_pre=None,
                x_adv_norm=None,
                x_ben_pre=None,
                x_mal_pre=None,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["arrays"], {})


if __name__ == "__main__":
    unittest.main()
