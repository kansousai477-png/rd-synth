from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_ops import Stage3Settings
from rdsynth.pipeline.stage3_pcap_apply import (
    build_target_metric_fn,
    prepare_pcap_search_context,
    record_pcap_apply_settings,
)


class Stage3PcapApplyTest(unittest.TestCase):
    def _settings(self, **overrides: object) -> Stage3Settings:
        cfg = {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1.0e-3,
        }
        cfg.update(overrides)
        return Stage3Settings.from_cfg(cfg, {"oracle_name": "mlp_small"})

    def test_record_pcap_apply_settings_populates_metrics(self) -> None:
        settings = self._settings(
            pcap_apply_fields=["payload_scale", "dst_port_new"],
            pcap_search_probe_topk=7,
            pcap_tcp_fixup=False,
            pcap_dst_port_policy="set",
            pcap_dst_port_allowlist=[80, 443],
        )
        metrics_payload: dict[str, object] = {}

        record_pcap_apply_settings(metrics_payload, settings=settings, protocol_auto_fix=True)

        self.assertEqual(metrics_payload["pcap_apply_fields"], ["payload_scale", "dst_port_new"])
        self.assertEqual(metrics_payload["pcap_search_probe_topk"], 7)
        self.assertEqual(metrics_payload["pcap_search_rounds"], 1)
        self.assertFalse(metrics_payload["pcap_tcp_fixup"])
        self.assertTrue(metrics_payload["pcap_protocol_auto_fix"])
        self.assertEqual(metrics_payload["pcap_dst_port_policy"], "set")
        self.assertEqual(metrics_payload["pcap_dst_port_allowlist"], [80, 443])

    def test_build_target_metric_fn_respects_alignment_mask(self) -> None:
        pcap_features = Mock()
        pcap_features.extract.return_value = (
            np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
            "mock",
            {"alignment": {"missing_features": ["b"]}},
        )
        pcap_features.classify_features.return_value = (
            np.asarray([0.1, 0.9], dtype=np.float32),
            np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32),
        )
        metric_fn = build_target_metric_fn(
            pcap_features=pcap_features,
            feature_names=["a", "b", "c"],
            orig_feat_pre_mean=np.asarray([1.0, 1.0, 1.0], dtype=np.float64),
        )

        pmal, target_l2, target_mae, response_l2, _ = metric_fn(
            Path("candidate.pcap"),
            np.asarray([2.0, 999.0, 3.0], dtype=np.float64),
        )

        self.assertAlmostEqual(pmal, 0.9)
        self.assertAlmostEqual(target_l2, np.linalg.norm(np.asarray([2.0, 3.0])))
        self.assertAlmostEqual(target_mae, 2.5)
        self.assertAlmostEqual(response_l2, np.linalg.norm(np.asarray([3.0, 4.0, 5.0])))

    def test_build_target_metric_fn_uses_field_sensitive_mask(self) -> None:
        pcap_features = Mock()
        pcap_features.extract.return_value = (
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            "mock",
            {},
        )
        pcap_features.classify_features.return_value = (
            np.asarray([0.1, 0.9], dtype=np.float32),
            np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
        )
        metric_fn = build_target_metric_fn(
            pcap_features=pcap_features,
            feature_names=["Flow IAT Mean", "Dst Port", "Packet Length Mean", "ACK Flag Count"],
            orig_feat_pre_mean=np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )

        pmal, target_l2, target_mae, response_l2, _ = metric_fn(
            Path("candidate.pcap"),
            np.asarray([7.0, 999.0, 999.0, 999.0], dtype=np.float64),
            ["mean_iat_ms"],
        )

        self.assertAlmostEqual(pmal, 0.9)
        self.assertAlmostEqual(target_l2, 3.0)
        self.assertAlmostEqual(target_mae, 3.0)
        self.assertAlmostEqual(response_l2, 10.0)

    def test_build_target_metric_fn_ignores_field_mask_when_feature_dims_mismatch(self) -> None:
        pcap_features = Mock()
        pcap_features.extract.return_value = (
            np.asarray([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            "mock",
            {},
        )
        pcap_features.classify_features.return_value = (
            np.asarray([0.1, 0.9], dtype=np.float32),
            np.asarray([[10.0, 20.0]], dtype=np.float32),
        )
        metric_fn = build_target_metric_fn(
            pcap_features=pcap_features,
            feature_names=["Flow IAT Mean", "Dst Port", "Packet Length Mean", "ACK Flag Count"],
            orig_feat_pre_mean=np.asarray([1.0, 2.0], dtype=np.float64),
        )

        pmal, target_l2, target_mae, response_l2, _ = metric_fn(
            Path("candidate.pcap"),
            np.asarray([7.0, 9.0], dtype=np.float64),
            ["mean_iat_ms"],
        )

        self.assertAlmostEqual(pmal, 0.9)
        self.assertAlmostEqual(target_l2, np.linalg.norm(np.asarray([3.0, 11.0])))
        self.assertAlmostEqual(target_mae, 7.0)
        self.assertAlmostEqual(response_l2, np.linalg.norm(np.asarray([9.0, 18.0])))

    def test_prepare_pcap_search_context_collects_search_inputs(self) -> None:
        pcap_features = Mock()
        pcap_features.extract.return_value = (
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "mock",
            {},
        )
        pcap_features.pcap_prob.return_value = (0.75, None, None, None)
        pcap_features.classify_features.return_value = (
            np.asarray([0.2, 0.8], dtype=np.float32),
            np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
        )

        with patch("rdsynth.pipeline.stage3_pcap_apply.build_remap_targets", return_value=np.asarray([[2.0, 6.0]])):
            context = prepare_pcap_search_context(
                Path("seed.pcap"),
                pcap_features=pcap_features,
                feature_names=["x", "y"],
                pcap_target_mod=None,
            )

        self.assertEqual(context.orig_pmal_for_selection, 0.75)
        np.testing.assert_allclose(context.pcap_target_mod, np.asarray([2.0, 6.0]))
        np.testing.assert_allclose(context.orig_feat_pre_mean, np.asarray([20.0, 30.0]))


if __name__ == "__main__":
    unittest.main()
