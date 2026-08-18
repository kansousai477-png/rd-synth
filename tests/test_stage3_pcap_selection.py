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

from rdsynth.pipeline.stage3_ops import Stage3Settings
from rdsynth.pipeline.stage3_pcap_selection import build_pcap_selection, resolve_selected_pcap


class Stage3PcapSelectionTest(unittest.TestCase):
    def _settings(self, **overrides: object) -> Stage3Settings:
        cfg = {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1.0e-3,
        }
        cfg.update(overrides)
        return Stage3Settings.from_cfg(cfg, {"oracle_name": "mlp_small"})

    def test_build_pcap_selection_uses_existing_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pcap_path = tmp_path / "seed.pcap"
            pcap_path.write_bytes(b"pcap")
            settings = self._settings(pcap_path=str(pcap_path), pcap_scan_limit=5, pcap_scan_dir=str(tmp_path))
            metrics_payload: dict[str, object] = {}

            selection = build_pcap_selection(settings, metrics_payload)

        self.assertEqual(selection.selected_path, pcap_path)
        self.assertEqual(selection.selected_source, "config")
        self.assertEqual(metrics_payload["pcap_scan_limit"], 5)
        self.assertEqual(metrics_payload["pcap_scan_dir"], str(tmp_path))

    def test_resolve_selected_pcap_prefers_better_scanned_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_pcap = tmp_path / "config.pcap"
            scanned_pcap = tmp_path / "scanned.pcap"
            config_pcap.write_bytes(b"a")
            scanned_pcap.write_bytes(b"b")
            settings = self._settings(
                pcap_path=str(config_pcap),
                pcap_scan_dir=str(tmp_path),
                pcap_scan_limit=10,
            )
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.pcap_prob.return_value = (0.8, None, None, None)
            pcap_features.rank_pcaps.return_value = [
                {
                    "path": str(scanned_pcap),
                    "name": scanned_pcap.name,
                    "prob_malicious": 0.9,
                    "pred_label": 1,
                    "selection_score": 0.95,
                    "target_feature_l2": 1.0,
                    "target_mod_l2": 2.0,
                }
            ]
            pcap_features.score_pcap_candidate.return_value = {"selection_score": 0.5}

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_path, scanned_pcap)
        self.assertEqual(resolved.selected_source, "scan")
        self.assertEqual(metrics_payload["pcap_selected_source"], "scan")
        self.assertEqual(metrics_payload["pcap_scan_count"], 1)
        self.assertTrue(metrics_payload["pcap_evasion_valid"])

    def test_resolve_selected_pcap_ignores_skipped_scan_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            skipped_pcap = tmp_path / "too_large.pcap"
            valid_pcap = tmp_path / "valid.pcap"
            skipped_pcap.write_bytes(b"a")
            valid_pcap.write_bytes(b"b")
            settings = self._settings(pcap_scan_dir=str(tmp_path), pcap_scan_limit=2)
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.rank_pcaps.return_value = [
                {
                    "path": str(skipped_pcap),
                    "name": skipped_pcap.name,
                    "prob_malicious": None,
                    "pred_label": None,
                    "selection_score": float("-inf"),
                    "target_feature_l2": None,
                    "target_mod_l2": None,
                    "status": "skipped_too_large",
                    "skip_reason": "pcap_too_large",
                    "pcap_size_bytes": 999,
                },
                {
                    "path": str(valid_pcap),
                    "name": valid_pcap.name,
                    "prob_malicious": 0.8,
                    "pred_label": 1,
                    "selection_score": 0.8,
                    "target_feature_l2": 1.0,
                    "target_mod_l2": 2.0,
                },
            ]

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_path, valid_pcap)
        self.assertEqual(metrics_payload["pcap_scan_skipped_count"], 1)
        self.assertEqual(metrics_payload["pcap_scan_top_candidates"][0]["skip_reason"], "pcap_too_large")

    def test_resolve_selected_pcap_falls_back_to_best_candidate_when_none_meets_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            benignish_pcap = tmp_path / "benignish.pcap"
            benignish_pcap.write_bytes(b"b")
            settings = self._settings(pcap_scan_dir=str(tmp_path), pcap_scan_limit=2, pcap_scan_min_prob=0.5)
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.rank_pcaps.return_value = [
                {
                    "path": str(benignish_pcap),
                    "name": benignish_pcap.name,
                    "prob_malicious": 0.2,
                    "pred_label": 0,
                    "selection_score": 0.9,
                    "target_feature_l2": 1.0,
                    "target_mod_l2": 2.0,
                }
            ]

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertIsNotNone(resolved.selected_path)
        self.assertEqual(resolved.selected_path, benignish_pcap)

    def test_resolve_selected_pcap_falls_back_to_first_scan_candidate_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            candidate = tmp_path / "a.pcap"
            candidate.write_bytes(b"x")
            settings = self._settings(pcap_scan_dir=str(tmp_path), pcap_scan_glob="*.pcap")
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=Mock(),
                pcap_eval_model=None,
                target_pre=None,
                target_mod=None,
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_path, candidate)
        self.assertEqual(metrics_payload["pcap_selected_name"], candidate.name)

    def test_stage3_settings_default_search_weights_and_alphas_are_search_enabled(self) -> None:
        settings = self._settings()
        self.assertEqual(settings.pcap_search_alphas, [0.25, 0.5, 1.0, 1.5])
        self.assertAlmostEqual(settings.pcap_scan_pmal_weight, 0.45)
        self.assertAlmostEqual(settings.pcap_scan_target_fit_weight, 0.35)
        self.assertAlmostEqual(settings.pcap_scan_target_mod_fit_weight, 0.20)

    def test_build_pcap_selection_uses_global_attack_label_list_for_semantics(self) -> None:
        settings = self._settings(
            pcap_dataset="cic_ids2018",
            pcap_attack_label="GLOBAL",
            pcap_attack_labels=["DDOS attack-HOIC", "SQL Injection"],
        )
        metrics_payload: dict[str, object] = {}

        selection = build_pcap_selection(settings, metrics_payload)

        self.assertIn("server_scan_probe", selection.semantic_categories)
        self.assertIn("botnet_loader", selection.semantic_categories)
        self.assertIn("phishing_clickfix_ad", selection.semantic_categories)
        self.assertEqual(metrics_payload["pcap_semantic_attack_label"], "GLOBAL")
        self.assertEqual(metrics_payload["pcap_semantic_attack_labels"], ["DDOS attack-HOIC", "SQL Injection"])

    def test_resolve_selected_pcap_randomly_samples_multiple_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pcaps = []
            ranked = []
            for idx in range(5):
                candidate = tmp_path / f"cand_{idx}.pcap"
                candidate.write_bytes(b"x")
                pcaps.append(candidate)
                ranked.append(
                    {
                        "path": str(candidate),
                        "name": candidate.name,
                        "prob_malicious": 0.9 - 0.01 * idx,
                        "pred_label": 1,
                        "selection_score": 0.95 - 0.01 * idx,
                        "target_feature_l2": 1.0 + idx,
                        "target_mod_l2": 2.0 + idx,
                    }
                )
            settings = self._settings(
                pcap_scan_dir=str(tmp_path),
                pcap_scan_limit=5,
                pcap_source_selection_mode="random",
                pcap_source_sample_n=3,
                pcap_source_sample_seed=7,
            )
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.rank_pcaps.return_value = ranked
            pcap_features.pcap_prob.return_value = (0.8, None, None, None)

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_source, "scan_random")
        self.assertEqual(len(resolved.candidate_paths), 3)
        self.assertEqual(metrics_payload["pcap_source_selection_mode"], "random")
        self.assertEqual(len(metrics_payload["pcap_source_sampled_names"]), 3)

    def test_resolve_selected_pcap_random_hard_prefers_malicious_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ranked = []
            for idx, (pred_label, prob) in enumerate([(0, 0.10), (1, 0.91), (0, 0.85), (1, 0.95)]):
                candidate = tmp_path / f"cand_{idx}.pcap"
                candidate.write_bytes(b"x")
                ranked.append(
                    {
                        "path": str(candidate),
                        "name": candidate.name,
                        "prob_malicious": prob,
                        "pred_label": pred_label,
                        "selection_score": prob,
                        "target_feature_l2": 1.0 + idx,
                        "target_mod_l2": 2.0 + idx,
                    }
                )
            settings = self._settings(
                pcap_scan_dir=str(tmp_path),
                pcap_scan_limit=8,
                pcap_scan_min_prob=0.8,
                pcap_source_selection_mode="random_hard",
                pcap_source_sample_n=2,
                pcap_source_sample_seed=13,
            )
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.rank_pcaps.return_value = ranked
            pcap_features.pcap_prob.return_value = (0.92, None, None, None)

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_source, "scan_random_hard")
        self.assertEqual(len(resolved.candidate_paths), 2)
        self.assertTrue(
            all(path.name in {"cand_1.pcap", "cand_2.pcap", "cand_3.pcap"} for path in resolved.candidate_paths)
        )
        self.assertEqual(metrics_payload["pcap_source_hard_candidate_count"], 3)
        self.assertTrue(metrics_payload["pcap_source_hard_filter_applied"])

    def test_resolve_selected_pcap_all_uses_every_ranked_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ranked = []
            for idx in range(4):
                candidate = tmp_path / f"cand_{idx}.pcap"
                candidate.write_bytes(b"x")
                ranked.append(
                    {
                        "path": str(candidate),
                        "name": candidate.name,
                        "prob_malicious": 0.2 + 0.1 * idx,
                        "pred_label": idx % 2,
                        "selection_score": 0.5 + 0.01 * idx,
                        "target_feature_l2": 1.0,
                        "target_mod_l2": 2.0,
                    }
                )
            settings = self._settings(
                pcap_scan_dir=str(tmp_path),
                pcap_scan_limit=0,
                pcap_source_selection_mode="all",
                pcap_source_sample_n=0,
            )
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.rank_pcaps.return_value = ranked
            pcap_features.pcap_prob.return_value = (0.8, None, None, None)

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_source, "scan_all")
        self.assertEqual(len(resolved.candidate_paths), 4)
        self.assertEqual(len(metrics_payload["pcap_source_sampled_names"]), 4)
        pcap_features.rank_pcaps.assert_called_once()
        self.assertEqual(pcap_features.rank_pcaps.call_args.args[2], 0)

    def test_resolve_selected_pcap_top_hard_uses_ranked_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ranked = []
            for idx, (pred_label, prob) in enumerate([(1, 0.91), (0, 0.20), (1, 0.88), (1, 0.86)]):
                candidate = tmp_path / f"cand_{idx}.pcap"
                candidate.write_bytes(b"x")
                ranked.append(
                    {
                        "path": str(candidate),
                        "name": candidate.name,
                        "prob_malicious": prob,
                        "pred_label": pred_label,
                        "selection_score": 1.0 - 0.01 * idx,
                        "target_feature_l2": 1.0 + idx,
                        "target_mod_l2": 2.0 + idx,
                    }
                )
            settings = self._settings(
                pcap_scan_dir=str(tmp_path),
                pcap_scan_limit=8,
                pcap_scan_min_prob=0.8,
                pcap_source_selection_mode="top_hard",
                pcap_source_sample_n=2,
            )
            metrics_payload: dict[str, object] = {}
            selection = build_pcap_selection(settings, metrics_payload)
            pcap_features = Mock()
            pcap_features.rank_pcaps.return_value = ranked
            pcap_features.pcap_prob.return_value = (0.91, None, None, None)

            resolved = resolve_selected_pcap(
                selection,
                pcap_features=pcap_features,
                pcap_eval_model=object(),
                target_pre=np.asarray([1.0]),
                target_mod=np.asarray([2.0]),
                metrics_payload=metrics_payload,
            )

        self.assertEqual(resolved.selected_source, "scan_top_hard")
        self.assertEqual([path.name for path in resolved.candidate_paths], ["cand_0.pcap", "cand_2.pcap"])
        self.assertEqual(metrics_payload["pcap_source_sampled_names"], ["cand_0.pcap", "cand_2.pcap"])
        self.assertTrue(metrics_payload["pcap_source_hard_filter_applied"])


if __name__ == "__main__":
    unittest.main()
