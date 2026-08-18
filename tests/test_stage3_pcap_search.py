from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys

sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_ops import Stage3Settings
from rdsynth.pipeline.stage3_pcap_search import (
    Stage3PcapSearchResult,
    _candidate_field_sets,
    _candidate_mods,
    _candidate_probe_filename,
    search_and_write_pcaps,
)
from rdsynth.stages.stage3_targets import MOD_NAMES

N_MOD = len(MOD_NAMES)


class Stage3PcapSearchTest(unittest.TestCase):
    def _settings(self) -> Stage3Settings:
        return Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": [1.0],
            },
            {"oracle_name": "mlp_small"},
        )

    def test_search_and_write_pcaps_simple_path(self) -> None:
        settings = self._settings()
        writes: list[str] = []

        def fake_wrpcap(path: str, pkts: object) -> None:
            writes.append(path)

        def fake_target_metric(path: Path, target_pre: np.ndarray | None):
            return 0.2, 1.0, 0.5, 0.3, {"alignment": {"coverage": 1.0}}

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            with patch("rdsynth.pipeline.stage3_pcap_search.apply_mod_using_scapy", return_value=[1, 2]):
                result = search_and_write_pcaps(
                    pkts=[0],
                    mods=np.zeros((1, N_MOD), dtype=np.float32),
                    adv=np.zeros((1, N_MOD), dtype=np.float32),
                    pcap_out_dir=out_dir / "pcap",
                    settings=settings,
                    seed=1,
                    protocol_auto_fix=True,
                    pcap_eval_model=None,
                    search_alphas=[1.0],
                    pcap_target_mod=None,
                    orig_pmal_for_selection=None,
                    orig_feat_pre_mean=np.zeros(N_MOD, dtype=np.float32),
                    out_dir=out_dir,
                    target_metric_fn=fake_target_metric,
                    wrpcap_fn=fake_wrpcap,
                )
        self.assertIsInstance(result, Stage3PcapSearchResult)
        self.assertTrue(result.pcap_modified)
        self.assertEqual(result.pcap_written_count, 1)
        self.assertEqual(result.pcap_packet_count, 2)
        self.assertIn("pcap_out_dir", result.metrics_payload())
        self.assertEqual(len(writes), 1)

    def test_candidate_helpers_and_search_probe_path(self) -> None:
        field_sets = _candidate_field_sets(
            ["mean_iat_ms", "std_iat_ms", "pad_bytes", "flag_ratio", "dst_port_new", "payload_scale", "flow_scale"],
            True,
        )
        self.assertTrue(any(fields == ["dst_port_new"] for fields in field_sets))
        self.assertTrue(any("pad_bytes" in fields for fields in field_sets))

        mod_row = np.arange(N_MOD, dtype=np.float32)
        with patch("rdsynth.pipeline.stage3_pcap_search.clip_modifications", side_effect=lambda x: x):
            candidates = _candidate_mods(mod_row, np.zeros_like(mod_row), [0.5, 1.0], bidirectional=False)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0][0], 0.5)

        with patch("rdsynth.pipeline.stage3_pcap_search.clip_modifications", side_effect=lambda x: x):
            bi_candidates = _candidate_mods(mod_row, np.zeros_like(mod_row), [1.0], bidirectional=True)
        self.assertEqual([row[0] for row in bi_candidates], [1.0, -0.5, 0.0])

        settings = self._settings()
        settings = Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": [0.5, 1.0],
                "pcap_apply_fields": ["pad_bytes", "payload_scale", "dst_port_new"],
                "pcap_search_field_subsets": True,
                "pcap_search_probe_topk": 2,
                "pcap_apply_n": 1,
            },
            {"oracle_name": "mlp_small"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            def fake_target_metric(path: Path, target_pre: np.ndarray | None):
                name = path.name
                if "0_5" in name:
                    return 0.4, 0.2, 0.1, 0.3, {"alignment": {"coverage": 1.0}}
                return 0.1, 0.1, 0.05, 0.6, {"alignment": {"coverage": 1.0}}

            with (
                patch(
                    "rdsynth.pipeline.stage3_pcap_search.apply_mod_using_scapy",
                    side_effect=lambda pkts, mod, **kwargs: [1, 2, 3],
                ),
                patch("rdsynth.pipeline.stage3_pcap_search.clip_modifications", side_effect=lambda x: x),
            ):
                result = search_and_write_pcaps(
                    pkts=[0],
                    mods=np.asarray([[1.0] * len(mod_row)], dtype=np.float32),
                    adv=np.asarray([[0.1] * len(mod_row)], dtype=np.float32),
                    pcap_out_dir=out_dir / "pcap",
                    settings=settings,
                    seed=1,
                    protocol_auto_fix=True,
                    pcap_eval_model=object(),
                    search_alphas=[0.5, 1.0],
                    pcap_target_mod=np.zeros(len(mod_row), dtype=np.float32),
                    orig_pmal_for_selection=None,
                    orig_feat_pre_mean=np.zeros(len(mod_row), dtype=np.float32),
                    out_dir=out_dir,
                    target_metric_fn=fake_target_metric,
                    wrpcap_fn=lambda path, pkts: None,
                )
            self.assertIsNotNone(result.pcap_search_trace_path)
            self.assertEqual(result.metrics_payload()["pcap_search_trace_path"], result.pcap_search_trace_path)
            with Path(result.pcap_search_trace_path).open("r", encoding="utf-8-sig", newline="") as handle:
                trace_rows = list(csv.DictReader(handle))

        self.assertEqual(result.pcap_written_count, 1)
        self.assertIsNotNone(result.pcap_selected_alpha_mean)
        self.assertIsNotNone(result.pcap_selected_field_sets)
        self.assertIsNotNone(result.pcap_selected_deployability_score_mean)
        self.assertIsNotNone(result.pcap_selected_response_l2_mean)
        self.assertIsNotNone(result.pcap_selected_pmal_mean)
        self.assertIsNotNone(result.pcap_selected_target_l2_mean)
        self.assertIsNotNone(result.pcap_selected_alignment_coverage_mean)
        self.assertGreaterEqual(len(trace_rows), 1)
        self.assertIn("malicious_prob", trace_rows[0])
        self.assertIn("accepted_as_best", trace_rows[0])
        self.assertTrue(any(row["malicious_prob"] for row in trace_rows))

    def test_search_prefers_relative_pmal_improvement(self) -> None:
        settings = Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": [0.5, 1.0],
                "pcap_apply_fields": ["payload_scale"],
                "pcap_search_field_subsets": True,
                "pcap_search_probe_topk": 4,
                "pcap_apply_n": 1,
                "pcap_search_bidirectional": False,
            },
            {"oracle_name": "mlp_small"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            def fake_target_metric(path: Path, target_pre: np.ndarray | None):
                if "0_5" in path.name:
                    return 0.30, 1.2, 0.7, 0.2, {"alignment": {"coverage": 1.0}}
                return 0.42, 0.4, 0.2, 1.0, {"alignment": {"coverage": 1.0}}

            with (
                patch(
                    "rdsynth.pipeline.stage3_pcap_search.apply_mod_using_scapy",
                    side_effect=lambda pkts, mod, **kwargs: [1, 2, 3],
                ),
                patch("rdsynth.pipeline.stage3_pcap_search.clip_modifications", side_effect=lambda x: x),
            ):
                result = search_and_write_pcaps(
                    pkts=[0],
                    mods=np.asarray([[1.0] * N_MOD], dtype=np.float32),
                    adv=np.asarray([[0.1] * 7], dtype=np.float32),
                    pcap_out_dir=out_dir / "pcap",
                    settings=settings,
                    seed=1,
                    protocol_auto_fix=True,
                    pcap_eval_model=object(),
                    search_alphas=[0.5, 1.0],
                    pcap_target_mod=np.zeros(N_MOD, dtype=np.float32),
                    orig_pmal_for_selection=0.65,
                    orig_feat_pre_mean=np.zeros(N_MOD, dtype=np.float32),
                    out_dir=out_dir,
                    target_metric_fn=fake_target_metric,
                    wrpcap_fn=lambda path, pkts: None,
                )

        self.assertEqual(result.pcap_selected_alphas, [0.5])

    def test_bidirectional_search_can_select_reverse_residual(self) -> None:
        settings = Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": [1.0],
                "pcap_apply_fields": ["payload_scale"],
                "pcap_search_probe_topk": 4,
                "pcap_apply_n": 1,
                "pcap_search_bidirectional": True,
            },
            {"oracle_name": "mlp_small"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            def fake_target_metric(path: Path, target_pre: np.ndarray | None):
                if "-0_5" in path.name or "-0.5" in path.name:
                    return 0.10, 1.2, 0.7, 0.8, {"alignment": {"coverage": 1.0}}
                return 0.55, 0.3, 0.1, 0.2, {"alignment": {"coverage": 1.0}}

            with (
                patch(
                    "rdsynth.pipeline.stage3_pcap_search.apply_mod_using_scapy",
                    side_effect=lambda pkts, mod, **kwargs: [1, 2, 3],
                ),
                patch("rdsynth.pipeline.stage3_pcap_search.clip_modifications", side_effect=lambda x: x),
            ):
                result = search_and_write_pcaps(
                    pkts=[0],
                    mods=np.asarray([[1.0] * N_MOD], dtype=np.float32),
                    adv=np.asarray([[0.1] * 7], dtype=np.float32),
                    pcap_out_dir=out_dir / "pcap",
                    settings=settings,
                    seed=1,
                    protocol_auto_fix=True,
                    pcap_eval_model=object(),
                    search_alphas=[1.0],
                    pcap_target_mod=np.zeros(N_MOD, dtype=np.float32),
                    orig_pmal_for_selection=0.65,
                    orig_feat_pre_mean=np.zeros(N_MOD, dtype=np.float32),
                    out_dir=out_dir,
                    target_metric_fn=fake_target_metric,
                    wrpcap_fn=lambda path, pkts: None,
                )

        self.assertEqual(result.pcap_selected_alphas, [-0.5])
        self.assertEqual(result.pcap_kept_original_count, 0)
        self.assertIsNotNone(result.pcap_selected_pmal_mean)

    def test_iterative_search_refines_selected_modification(self) -> None:
        settings = Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": [1.0],
                "pcap_apply_fields": ["payload_scale"],
                "pcap_search_probe_topk": 1,
                "pcap_apply_n": 1,
                "pcap_search_bidirectional": False,
                "pcap_search_rounds": 2,
            },
            {"oracle_name": "mlp_small"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            def fake_target_metric(path: Path, target_pre: np.ndarray | None):
                if "_r1_" in path.name:
                    return 0.01, 0.0, 0.0, 2.0, {"alignment": {"coverage": 1.0}}
                return 0.35, 0.8, 0.4, 0.3, {"alignment": {"coverage": 1.0}}

            with (
                patch(
                    "rdsynth.pipeline.stage3_pcap_search.apply_mod_using_scapy",
                    side_effect=lambda pkts, mod, **kwargs: [1, 2, 3],
                ),
                patch("rdsynth.pipeline.stage3_pcap_search.clip_modifications", side_effect=lambda x: x),
            ):
                result = search_and_write_pcaps(
                    pkts=[0],
                    mods=np.asarray([[1.0] * N_MOD], dtype=np.float32),
                    adv=np.asarray([[0.1] * 7], dtype=np.float32),
                    pcap_out_dir=out_dir / "pcap",
                    settings=settings,
                    seed=1,
                    protocol_auto_fix=True,
                    pcap_eval_model=object(),
                    search_alphas=[1.0],
                    pcap_target_mod=np.zeros(N_MOD, dtype=np.float32),
                    orig_pmal_for_selection=0.65,
                    orig_feat_pre_mean=np.zeros(N_MOD, dtype=np.float32),
                    out_dir=out_dir,
                    target_metric_fn=fake_target_metric,
                    wrpcap_fn=lambda path, pkts: None,
                )
            self.assertIsNotNone(result.pcap_search_trace_path)
            with Path(result.pcap_search_trace_path).open("r", encoding="utf-8-sig", newline="") as handle:
                trace_rows = list(csv.DictReader(handle))

        self.assertEqual(result.pcap_search_rounds_used_mean, 2.0)
        self.assertEqual(result.metrics_payload()["pcap_search_rounds_used_mean"], 2.0)
        self.assertEqual({row["round_idx"] for row in trace_rows}, {"0", "1"})
        self.assertEqual([row["malicious_prob"] for row in trace_rows], ["0.35", "0.01"])

    def test_keep_probe_uses_short_temp_root_outside_run_dir(self) -> None:
        settings = Stage3Settings.from_cfg(
            {
                "epochs": 2,
                "batch_size": 4,
                "lr": 1.0e-3,
                "pcap_search_alphas": [1.0],
                "pcap_apply_fields": ["payload_scale"],
                "pcap_apply_n": 1,
                "pcap_search_bidirectional": False,
            },
            {"oracle_name": "mlp_small"},
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / ("deep_" * 20)
            writes: list[str] = []

            def fake_wrpcap(path: str, pkts: object) -> None:
                writes.append(path)

            with patch("rdsynth.pipeline.stage3_pcap_search.apply_mod_using_scapy", return_value=[1, 2, 3]):
                search_and_write_pcaps(
                    pkts=[0],
                    mods=np.asarray([[1.0] * N_MOD], dtype=np.float32),
                    adv=np.asarray([[0.1] * 7], dtype=np.float32),
                    pcap_out_dir=out_dir / "pcap",
                    settings=settings,
                    seed=1,
                    protocol_auto_fix=True,
                    pcap_eval_model=object(),
                    search_alphas=[1.0],
                    pcap_target_mod=np.zeros(N_MOD, dtype=np.float32),
                    orig_pmal_for_selection=0.65,
                    orig_feat_pre_mean=np.zeros(N_MOD, dtype=np.float32),
                    out_dir=out_dir,
                    target_metric_fn=lambda path, target_pre, field_set=None: (
                        0.30,
                        1.2,
                        0.7,
                        0.2,
                        {"alignment": {"coverage": 1.0}},
                    ),
                    wrpcap_fn=fake_wrpcap,
                )

        self.assertGreaterEqual(len(writes), 2)
        keep_probe_path = Path(writes[0])
        self.assertEqual(keep_probe_path.name, "cand_keep_0.pcap")
        self.assertNotIn(str(out_dir), str(keep_probe_path))

    def test_candidate_probe_filename_hashes_field_set(self) -> None:
        name = _candidate_probe_filename(0, 1.0, ["mean_iat_ms", "payload_scale", "dst_port_new"])
        self.assertTrue(name.startswith("cand_0_1_0_"))
        self.assertTrue(name.endswith(".pcap"))
        self.assertLess(len(name), 40)


if __name__ == "__main__":
    unittest.main()
