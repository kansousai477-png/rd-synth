from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_ops import Stage3Settings
from rdsynth.pipeline.stage3_pcap_baselines import run_stage3_baseline_pcap_eval


class Stage3PcapBaselinesTest(unittest.TestCase):
    def _settings(self, **overrides: object) -> Stage3Settings:
        cfg = {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1.0e-3,
            "pcap_compare_baselines": True,
            "pcap_apply_n": 2,
        }
        cfg.update(overrides)
        return Stage3Settings.from_cfg(cfg, {"oracle_name": "mlp_small"})

    def test_noop_when_baselines_disabled(self) -> None:
        settings = self._settings(pcap_compare_baselines=False)
        metrics_payload = {"pcap_orig_pred_malicious": 1.0}
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_stage3_baseline_pcap_eval(
                cfg={"project": {"out_dir": str(Path(tmp_dir) / "outputs")}},
                settings=settings,
                metrics_payload=metrics_payload,
                pcap_evasion_valid=True,
                preprocessor=None,
                remap_mode="auto",
                remap_use_direct=False,
                remap_bundle=None,
                x_ben_raw=None,
                feature_names=[],
                scapy_available=False,
                protocol_auto_fix=True,
                pcap_path=None,
                pcap_features=None,
                out_dir=Path(tmp_dir),
                seed=1,
                device="cpu",
                x_train=[],
                effective_blend_fn=lambda *args, **kwargs: (0.5, {}),
                blend_fn=lambda *args, **kwargs: None,
            )
        self.assertEqual(metrics_payload, {"pcap_orig_pred_malicious": 1.0})

    def test_feature_only_baseline_uses_random_remap_control_and_native_pending_uses_shared_proxy(self) -> None:
        fake_scapy = types.ModuleType("scapy")
        fake_scapy_all = types.ModuleType("scapy.all")
        fake_scapy_all.rdpcap = lambda path: ["pkt"]
        fake_scapy_all.wrpcap = lambda path, pkts: None

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "outputs"
            stage2_dir = out_dir / "stage2"
            stage2_dir.mkdir(parents=True)
            npz_path = stage2_dir / "baseline_idsgan_lite_samples.npz"
            skipped_npz_path = stage2_dir / "baseline_gpmt_lite_samples.npz"
            import numpy as np

            np.savez(
                npz_path,
                baseline_name=np.array("idsgan_lite"),
                adv_pre=np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
                feature_names=np.array(["f1", "f2"], dtype="<U8"),
            )
            np.savez(
                skipped_npz_path,
                baseline_name=np.array("gpmt_lite"),
                adv_pre=np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
                feature_names=np.array(["f1", "f2"], dtype="<U8"),
            )

            pcap_path = root / "orig.pcap"
            pcap_path.write_bytes(b"pcap")

            adv_outputs = [root / "adv_0000.pcap", root / "adv_0001.pcap"]
            for path in adv_outputs:
                path.write_bytes(b"")

            metrics_payload = {"pcap_orig_pred_malicious": 1.0}
            pcap_features = SimpleNamespace(
                classify_pcap=lambda p: (
                    np.array([[0.0, 0.0]], dtype=np.float32),
                    "scapy",
                    {"status": "ok", "alignment": {"coverage": 1.0}},
                    np.array([0.2, 0.8], dtype=np.float32),
                    np.array([[0.3, 0.35]], dtype=np.float32),
                )
            )

            with (
                patch.dict(sys.modules, {"scapy": fake_scapy, "scapy.all": fake_scapy_all}),
                patch("rdsynth.pipeline.stage3_pcap_baselines.pcap_output_dir", return_value=root / "pcaps"),
                patch(
                    "rdsynth.pipeline.stage3_pcap_baselines.build_rule_based_modifications",
                    return_value=np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
                ),
                patch(
                    "rdsynth.pipeline.stage3_pcap_baselines.build_random_remap_modifications",
                    return_value=np.array([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32),
                ),
                patch("rdsynth.pipeline.stage3_pcap_baselines.clip_modifications", side_effect=lambda x: x),
                patch(
                    "rdsynth.pipeline.stage3_pcap_baselines.write_modified_pcaps",
                    return_value=(2, 20, 4.0, adv_outputs),
                ),
            ):
                run_stage3_baseline_pcap_eval(
                    cfg={"project": {"out_dir": str(out_dir)}},
                    settings=self._settings(),
                    metrics_payload=metrics_payload,
                    main_adv_pre=np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32),
                    pcap_evasion_valid=True,
                    preprocessor=SimpleNamespace(
                        inverse_transform=lambda x: np.asarray(x, dtype=np.float32),
                    ),
                    remap_mode="direct",
                    remap_use_direct=True,
                    remap_bundle=None,
                    x_ben_raw=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    feature_names=["f1", "f2"],
                    scapy_available=True,
                    protocol_auto_fix=True,
                    pcap_path=pcap_path,
                    pcap_features=pcap_features,
                    out_dir=root,
                    seed=1,
                    device="cpu",
                    x_train=np.ones((4, 2), dtype=np.float32),
                    effective_blend_fn=lambda *args, **kwargs: (0.5, {}),
                    blend_fn=lambda *args, **kwargs: None,
                )

            self.assertIn("baseline_direct_rule_only_pcap_written_count", metrics_payload)
            self.assertIn("baseline_idsgan_lite_pcap_written_count", metrics_payload)
            self.assertEqual(metrics_payload["baseline_direct_rule_only_pcap_written_count"], 2)
            self.assertEqual(metrics_payload["baseline_idsgan_lite_pcap_written_count"], 2)
            self.assertIn("baseline_direct_rule_only_paper_pcap_attack_success_rate", metrics_payload)
            self.assertIn("baseline_idsgan_lite_paper_pcap_attack_success_rate", metrics_payload)
            self.assertEqual(metrics_payload["baseline_gpmt_lite_pcap_eval_policy"], "shared_backend_proxy")
            self.assertTrue(metrics_payload["baseline_gpmt_lite_pcap_native_realization_pending"])
            self.assertTrue((root / "baseline_pcap_eval.csv").exists())
            self.assertFalse((root / "baseline_pcap_skipped.csv").exists())


if __name__ == "__main__":
    unittest.main()
