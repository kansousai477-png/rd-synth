from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_ops import (
    Stage3Settings,
    aligned_feature_diff,
    blend_modifications,
    detect_stage3_environment,
    effective_blend_alpha,
    pcap_output_dir,
    resolve_pcap_eval_model,
    select_remap_training_data,
    validate_remap_mode,
    write_modified_pcaps,
)


class Stage3HelpersTest(unittest.TestCase):
    def _settings(self) -> Stage3Settings:
        return Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
            },
            {"oracle_name": "mlp_small"},
        )

    def test_aligned_feature_diff_filters_missing_features(self) -> None:
        diff = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        out = aligned_feature_diff(
            diff,
            {"missing_features": ["b"]},
            ["a", "b", "c"],
        )
        np.testing.assert_allclose(out, np.array([1.0, 3.0]))

    def test_aligned_feature_diff_skips_mask_when_vector_dim_mismatches(self) -> None:
        diff = np.array([1.0, 2.0], dtype=np.float64)
        out = aligned_feature_diff(
            diff,
            {"missing_features": ["b"]},
            ["a", "b", "c"],
        )
        np.testing.assert_allclose(out, diff)

    def test_pcap_output_dir_uses_default_when_missing(self) -> None:
        settings = self._settings()
        out_dir = Path("/tmp/example_stage3")
        self.assertEqual(pcap_output_dir(settings, out_dir), out_dir / "pcap")

    def test_detect_stage3_environment_probes_optional_backends(self) -> None:
        with (
            patch("rdsynth.pipeline.stage3_ops.importlib.util.find_spec") as find_spec,
            patch("rdsynth.pipeline.stage3_ops._check_java_available") as check_java,
        ):
            find_spec.side_effect = [object(), None]
            check_java.return_value = False
            env = detect_stage3_environment()

        self.assertTrue(env.scapy_available)
        self.assertFalse(env.nfstream_available)
        self.assertFalse(env.cicflowmeter_available)

    def test_resolve_pcap_eval_model_prefers_oracle_when_requested(self) -> None:
        oracle = object()
        surrogate = object()

        selected = resolve_pcap_eval_model(
            ids=None,
            oracle=oracle,
            surrogate=surrogate,
            prefer_ids=False,
            prefer_oracle=True,
        )

        self.assertIs(selected.pcap_eval_model, oracle)
        self.assertEqual(selected.pcap_eval_model_name, "oracle")

    def test_resolve_pcap_eval_model_prefers_ids_when_requested(self) -> None:
        ids = object()
        oracle = object()

        selected = resolve_pcap_eval_model(
            ids=ids,
            oracle=oracle,
            surrogate=None,
            prefer_ids=True,
            prefer_oracle=True,
        )

        self.assertIs(selected.pcap_eval_model, ids)
        self.assertEqual(selected.pcap_eval_model_name, "ids")

    def test_select_remap_training_data_uses_requested_slice(self) -> None:
        x_train = np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)
        y_train = np.asarray([0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_allclose(select_remap_training_data(x_train, y_train, "benign"), np.asarray([[1.0], [3.0]]))
        np.testing.assert_allclose(
            select_remap_training_data(x_train, y_train, "malicious"), np.asarray([[2.0], [4.0]])
        )
        np.testing.assert_allclose(select_remap_training_data(x_train, y_train, "all"), x_train)

    def test_validate_remap_mode_rejects_unknown_value(self) -> None:
        with self.assertRaises(ValueError):
            validate_remap_mode("invalid")

    def test_effective_blend_alpha_enables_collapse_guard(self) -> None:
        mod_names = ["mean_iat_ms", "dst_port_new", "payload_scale"]
        learned = np.asarray([[0.1, 10.0, 0.2], [0.1, 11.0, 0.2]], dtype=np.float32)
        direct = np.asarray([[0.0, 20.0, 0.0], [1.0, 21.0, 1.0]], dtype=np.float32)

        alpha, info = effective_blend_alpha(
            learned,
            direct,
            mod_names,
            requested_alpha=0.8,
            collapse_ratio_threshold=0.5,
        )

        self.assertEqual(alpha, 0.35)
        self.assertEqual(info["blend_reason"], "collapse_guard")

    def test_blend_modifications_uses_weighted_continuous_features(self) -> None:
        mod_names = [
            "mean_iat_ms",
            "std_iat_ms",
            "pad_bytes",
            "dst_port_new",
            "flag_ratio",
            "flow_scale",
            "payload_scale",
        ]
        learned = np.asarray([[0.8, 1.0, 2.0, 10.0, 0.2, 1.5, 0.2]], dtype=np.float32)
        direct = np.asarray([[0.2, 3.0, 6.0, 30.0, 0.6, 1.0, 0.6]], dtype=np.float32)

        blended = blend_modifications(learned, direct, 0.25, mod_names)

        np.testing.assert_allclose(blended[:, 0], np.asarray([0.35], dtype=np.float32))
        np.testing.assert_allclose(blended[:, 6], np.asarray([0.5], dtype=np.float32))
        np.testing.assert_allclose(blended[:, 3], np.asarray([30.0], dtype=np.float32))

    def test_write_modified_pcaps_uses_helper_pipeline(self) -> None:
        settings = self._settings()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "pcap"
            written_paths: list[str] = []

            def fake_wrpcap(path: str, pkts: object) -> None:
                written_paths.append(path)

            with patch("rdsynth.pipeline.stage3_ops.apply_mod_using_scapy", return_value=[1, 2, 3]):
                written, packet_total, elapsed, paths = write_modified_pcaps(
                    pkts=[0],
                    mods=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                    output_dir=output_dir,
                    seed=11,
                    count=1,
                    settings=settings,
                    protocol_auto_fix=True,
                    wrpcap_fn=fake_wrpcap,
                )
            self.assertEqual(written, 1)
            self.assertEqual(packet_total, 3)
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertEqual(len(paths), 1)
            self.assertEqual(len(written_paths), 1)


if __name__ == "__main__":
    unittest.main()
