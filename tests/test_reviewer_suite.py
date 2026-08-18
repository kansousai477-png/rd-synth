from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

from rdsynth.pipeline import reviewer_suite as rs


class _OracleRestoreData:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


sys.modules.setdefault(
    "eval_transfer_oracles",
    types.SimpleNamespace(
        run_transfer_oracle_eval=lambda *args, **kwargs: None,
        run_transfer_ids_eval=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "run_pipeline",
    types.SimpleNamespace(run_pipeline_config=lambda *args, **kwargs: None),
)
sys.modules.setdefault(
    "rdsynth.baselines.paper_attacks",
    types.SimpleNamespace(
        get_paper_baseline_spec=lambda *args, **kwargs: None,
        generate_paper_attack_baseline=lambda *args, **kwargs: None,
        stage3_policy_for_baseline=lambda *args, **kwargs: None,
    ),
)
sys.modules.setdefault(
    "rdsynth.utils.checkpoints",
    types.SimpleNamespace(
        OracleRestoreData=_OracleRestoreData,
        load_stage1_artifacts=lambda *args, **kwargs: None,
        load_torch_state=lambda *args, **kwargs: None,
        resolve_stage1_artifact_root=lambda cfg, oracle_name: Path(cfg["project"]["out_dir"]) / "stage1" / oracle_name,
        resolve_stage1_metrics_path=lambda *args, **kwargs: Path("stage1/metrics.json"),
    ),
)
import run_reviewer_suite as runner


class ReviewerSuiteHelpersTest(unittest.TestCase):
    def test_deep_update_and_slug_numeric_helpers(self) -> None:
        target = {"a": {"x": 1, "y": 2}, "b": 3}
        patch_payload = {"a": {"y": 9, "z": 10}, "c": 4}
        out = rs.deep_update(target, patch_payload)

        self.assertIs(out, target)
        self.assertEqual(out["a"], {"x": 1, "y": 9, "z": 10})
        self.assertEqual(rs.slugify("A/B C:D\\E"), "A_B_C_D_E")
        self.assertIsNone(rs.to_float("nan"))
        self.assertIsNone(rs.to_float(""))
        self.assertEqual(rs.to_float("1.25"), 1.25)
        self.assertEqual(rs.fmt_float("1.23456", digits=2), "1.23")
        self.assertEqual(rs.fmt_float(None), "-")

    def test_csv_json_and_index_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "rows.csv"
            json_path = tmp / "data.json"

            rows = [{"a": "1", "b": "2"}, {"b": "3", "c": "4"}]
            rs.write_csv_rows(csv_path, rows)
            loaded_rows = rs.load_csv_rows(csv_path)
            self.assertEqual(len(loaded_rows), 2)
            self.assertEqual(loaded_rows[1]["c"], "4")

            indexed = rs.load_indexed_rows(csv_path, ["a", "b"])
            self.assertIn(("1", "2"), indexed)
            self.assertEqual(rs.sorted_indexed_rows(indexed)[0]["a"], "")

            rs.upsert_row(indexed, ("x",), {"k": "v"})
            self.assertEqual(indexed[("x",)]["k"], "v")

            json_path.write_text(json.dumps({"ok": True}), encoding="utf-8")
            self.assertEqual(rs.load_json(json_path), {"ok": True})
            self.assertEqual(rs.load_json(tmp / "missing.json"), {})

    def test_resolve_python_executable_prefers_explicit_env_and_existing_venv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            venv_python = repo_root / "venv" / "Scripts" / "python.exe"
            venv_python.parent.mkdir(parents=True, exist_ok=True)
            venv_python.write_text("", encoding="utf-8")

            self.assertEqual(
                rs.resolve_python_executable(repo_root=repo_root, explicit="custom-python"), "custom-python"
            )
            with patch.dict(os.environ, {"RDSYNTH_PYTHON": "env-python"}, clear=False):
                self.assertEqual(rs.resolve_python_executable(repo_root=repo_root), "env-python")
            with patch.dict(os.environ, {"USERPROFILE": str(repo_root)}, clear=True):
                self.assertEqual(rs.resolve_python_executable(repo_root=repo_root), str(venv_python))

    def test_resolve_profile_overrides_and_workload_summary(self) -> None:
        profile = rs.resolve_profile_overrides("standard")
        self.assertEqual(profile["speed_mode"], "standard")
        self.assertIn("backbone_gan", rs.DEFAULT_ABLATION_VARIANTS)
        self.assertIn("w_o_stage1", rs.DEFAULT_ABLATION_VARIANTS)
        self.assertIn("random_remap", rs.DEFAULT_ABLATION_VARIANTS)
        with self.assertRaises(SystemExit):
            rs.resolve_profile_overrides("unknown")

        workload = rs.summarize_workload(
            selected_attacks={"nb15": ["A", "B"], "2018": ["C"]},
            seeds=[42, 43],
            stage2_baselines=["identity", "pgd"],
            ablation_variants=["full", "w_o_stage1", "random_remap"],
            transfer_oracles=["logistic_small"],
            stage2_baselines_enabled=True,
            skip_transfer=False,
        )
        self.assertEqual(workload["combinations"], 6)
        self.assertEqual(workload["main_runs"], 4)
        self.assertEqual(workload["ablation_reuses"], 4)
        self.assertEqual(workload["ablation_reruns"], 8)
        self.assertEqual(workload["stage2_baseline_runs"], 8)
        self.assertEqual(workload["transfer_oracle_fits"], 4)

    def test_selected_attacks_and_first_row_helpers(self) -> None:
        suite_cfg = {"datasets": {"nb15": {"attacks": ["Bot", "Worms"]}}}
        base_cfg = {"data": {"dataset": "nb15"}}
        with patch("rdsynth.pipeline.reviewer_suite.resolve_attacks", return_value=["Bot", "Worms"]) as resolve_mock:
            attacks = rs.selected_attacks(
                "nb15",
                suite_cfg=suite_cfg,
                base_cfg=base_cfg,
                override_attacks=[],
                max_attacks=1,
            )
        resolve_mock.assert_called_once()
        self.assertEqual(attacks, ["Bot"])
        self.assertEqual(rs.first_row([{"name": "a"}, {"name": "b"}], key="name", value="b"), {"name": "b"})
        self.assertEqual(rs.first_row([], key="name", value="b"), {})

    def test_dataset_progress_total_matches_unsw_global_shape(self) -> None:
        total = runner._dataset_progress_total(
            global_binary_mode=True,
            attacks=["Exploits", "Generic"],
            seeds=[42, 43],
            ablation_variants=["full", "backbone_gan", "random_remap"],
            transfer_oracles=["logistic_small"],
            main_only=False,
            skip_transfer=False,
        )
        self.assertEqual(total, 11)

    def test_flush_dataset_outputs_writes_checkpoint_csvs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            runner._flush_dataset_outputs(
                dataset_root=root,
                main_runs={("nb15", "GLOBAL", "42"): {"dataset": "nb15", "attack_type": "GLOBAL", "seed": "42"}},
                stage1_attack_runs={
                    ("nb15", "Exploits", "42"): {"dataset": "nb15", "attack_type": "Exploits", "seed": "42"}
                },
                stage2_attack_runs={},
                main_stage2_baselines={},
                main_stage3_baselines={},
                transfer_runs={},
                ablation_runs={
                    ("nb15", "GLOBAL", "42", "full"): {
                        "dataset": "nb15",
                        "attack_type": "GLOBAL",
                        "seed": "42",
                        "variant": "full",
                    }
                },
                main_only=False,
            )
            self.assertTrue((root / "main_runs.csv").exists())
            self.assertTrue((root / "stage1_attack_runs.csv").exists())
            self.assertTrue((root / "ablation_runs.csv").exists())

    def test_ablation_reuses_main_stage3_selected_pcap(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            selected_pcap = root / "carrier.pcap"
            selected_pcap.write_bytes(b"pcap")
            main_out = root / "main"
            (main_out / "stage3").mkdir(parents=True)
            (main_out / "stage3" / "metrics.json").write_text(
                json.dumps({"pcap_selected_path": str(selected_pcap)}),
                encoding="utf-8",
            )
            cfg = {"stage3": {"pcap_path": "", "pcap_scan_dir": "data/PCAPs/malicious"}}

            runner._reuse_main_pcap_for_ablation(cfg, main_out)

            self.assertEqual(cfg["stage3"]["pcap_path"], str(selected_pcap))
            self.assertEqual(cfg["stage3"]["pcap_scan_dir"], "data/PCAPs/malicious")
            self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "fixed")
            self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 1)

    def test_collect_stage2_attack_rows_falls_back_to_metrics_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metrics_dir = root / "stage2" / "attack_eval" / "CommandInjection"
            metrics_dir.mkdir(parents=True)
            metrics_path = metrics_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "attack_type": "CommandInjection",
                        "stage2_eval_attack_rows": 812,
                        "asr_oracle": 0.98,
                        "asr_surrogate": 0.95,
                        "norm_FFD": 12.5,
                        "norm_SWD": 0.2,
                        "norm_AdvToMal_L2": 4.0,
                        "attack_score_queries_per_success_oracle": 0.0,
                    }
                ),
                encoding="utf-8",
            )

            rows = runner._collect_stage2_attack_rows("iot23", 42, root, ["CommandInjection"])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dataset"], "iot23")
        self.assertEqual(rows[0]["seed"], "42")
        self.assertEqual(rows[0]["attack_type"], "CommandInjection")
        self.assertEqual(rows[0]["stage2_eval_attack_rows"], "812")
        self.assertEqual(rows[0]["metrics_path"], str(metrics_path))


class ReviewerSuiteAttackResolutionTest(unittest.TestCase):
    def test_discover_attack_labels_uses_cache_and_iot_directory_mode(self) -> None:
        cfg = {"data": {"dataset": "cic_iot2023", "csv_dir": "ignored"}}
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "CSV" / "MITM-ArpSpoofing").mkdir(parents=True)
            (root / "CSV" / "Backdoor").mkdir(parents=True)
            (root / "CSV" / "Benign_Final").mkdir(parents=True)
            cfg["data"]["csv_dir"] = str(root / "CSV")
            with (
                patch("rdsynth.pipeline.reviewer_suite._load_cached_attack_labels", return_value=None),
                patch("rdsynth.pipeline.reviewer_suite._save_cached_attack_labels") as save_cache,
            ):
                attacks = rs.discover_attack_labels(cfg)
                save_cache.assert_called_once()
            with patch("rdsynth.pipeline.reviewer_suite._load_cached_attack_labels", return_value=attacks):
                cached = rs.discover_attack_labels(cfg)

        self.assertEqual(attacks, ["Backdoor", "MITM-ArpSpoofing"])
        self.assertEqual(cached, attacks)

    def test_discover_attack_labels_from_csv_and_resolve_attacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "sample.csv"
            csv_path.write_text(
                "Label,Value\nBenign,1\nBot,2\nBot,3\nSQL Injection,4\n",
                encoding="utf-8",
            )
            cfg = {"data": {"dataset": "custom"}}
            fake_profile = type(
                "Profile",
                (),
                {
                    "csv_path": str(csv_path),
                    "csv_dir": "",
                    "csv_glob": "*.csv",
                    "label_source": "column",
                    "label_col": "Label",
                    "benign_labels": ["Benign"],
                    "drop_cols": [],
                },
            )()
            with patch("rdsynth.pipeline.reviewer_suite.resolve_dataset_profile", return_value=fake_profile):
                with patch("rdsynth.pipeline.reviewer_suite._save_cached_attack_labels"):
                    attacks = rs.discover_attack_labels(cfg)
                    resolved = rs.resolve_attacks(cfg, ["sql  injection", "bot"])
                    with self.assertRaises(SystemExit):
                        rs.resolve_attacks(cfg, ["missing"])

        self.assertEqual(attacks, ["Bot", "SQL Injection"])
        self.assertEqual(resolved, ["SQL Injection", "Bot"])


class ReviewerSuiteConfigAssemblyTest(unittest.TestCase):
    def test_apply_speed_profile_updates_standard_and_quick_modes(self) -> None:
        base_cfg = {
            "project": {"num_threads": 8, "num_interop_threads": 4},
            "oracle_models": [{"epochs": 9, "max_batches_per_epoch": 50, "batch_size": 512}],
            "stage1": {"steps": 500, "real_warmup_steps": 100, "baseline_steps": 200, "eval_max_rows": 3000},
            "stage2": {
                "epochs": 18,
                "ae_epochs": 12,
                "timesteps": 150,
                "baselines": {"pgd_steps": 12, "eval_metrics": True},
            },
            "stage3": {
                "epochs": 20,
                "batch_size": 256,
                "pcap_scan_limit": 30,
                "pcap_eval_batch_size": 512,
                "pcap_apply_n": 2,
            },
        }

        standard = rs.apply_speed_profile(json.loads(json.dumps(base_cfg)), "standard")
        quick = rs.apply_speed_profile(json.loads(json.dumps(base_cfg)), "quick")

        self.assertLessEqual(standard["stage1"]["steps"], 240)
        self.assertLessEqual(standard["stage2"]["epochs"], 12)
        self.assertTrue(standard["stage2"]["baselines"]["eval_metrics"])
        self.assertLessEqual(quick["stage1"]["steps"], 60)
        self.assertFalse(quick["stage1"]["compare_baseline"])
        self.assertFalse(quick["stage2"]["baselines"]["eval_metrics"])
        self.assertFalse(quick["stage3"]["save_intermediate_results"])

    def test_build_run_config_applies_defaults_and_patch(self) -> None:
        base_cfg = {
            "project": {"seed": 1},
            "data": {"benign_labels": ["Benign"]},
            "stage1": {},
            "stage2": {},
            "stage3": {},
            "oracle_models": [{"name": "mlp_a"}],
        }
        out_dir = ROOT / "outputs" / "cfg_test"
        cfg = rs.build_run_config(
            base_cfg=base_cfg,
            attack="Bot",
            seed=7,
            out_dir=out_dir,
            profile="quick",
            stage2_baselines_enabled=False,
            stage3_baselines_enabled=False,
            stage2_baselines=["identity"],
            patch={"stage3": {"protocol_auto_fix": False}},
        )

        self.assertEqual(cfg["project"]["seed"], 7)
        self.assertIn("/_shared/stage1", str(cfg["project"]["stage1_shared_root"]))
        self.assertNotIn("include_labels", cfg["data"])
        self.assertEqual(cfg["project"]["attack_type"], "Bot")
        self.assertEqual(cfg["data"]["eval_attack_label"], "Bot")
        self.assertFalse(cfg["stage2"]["baselines"]["enable"])
        self.assertEqual(cfg["stage2"]["baselines"]["methods"], ["identity"])
        self.assertEqual(cfg["stage2"]["sample_pullback_alpha"], 0.10)
        self.assertEqual(cfg["stage2"]["sample_moment_alpha"], 0.10)
        self.assertTrue(cfg["stage2"]["pareto_eval"]["enable"])
        self.assertTrue(cfg["stage2"]["pareto_eval"]["auto_select"])
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["candidate_selection"], "stage3_closed_loop")
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["min_asr_oracle"], 0.95)
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["min_asr_surrogate"], 0.90)
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["per_sample_distance_weight"], 0.10)
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["per_sample_support_weight"], 0.15)
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["per_sample_remapability_weight"], 0.30)
        self.assertEqual(cfg["stage2"]["pareto_eval"]["selection"]["per_sample_stage3_closed_loop_weight"], 0.25)
        self.assertFalse(cfg["stage3"]["pcap_compare_baselines"])
        self.assertTrue(cfg["stage3"]["pcap_eval_use_ids"])
        self.assertTrue(cfg["stage3"]["pcap_eval_use_oracle"])
        self.assertFalse(cfg["stage3"]["protocol_auto_fix"])
        self.assertEqual(cfg["stage3"]["pcap_scan_limit"], 8)
        self.assertEqual(cfg["stage3"]["pcap_scan_min_prob"], 0.5)
        self.assertEqual(cfg["stage3"]["pcap_source_selection_mode"], "best")
        self.assertEqual(cfg["stage3"]["pcap_source_sample_n"], 1)
        self.assertEqual(cfg["stage3"]["pcap_dst_port_policy"], "flow_vocab_closest")
        self.assertEqual(cfg["stage3"]["pcap_path"], "")
        self.assertIn("stage3/pcap", cfg["stage3"]["pcap_out_dir"])

    def test_reviewer_stage1_shared_root_handles_ablation_directory(self) -> None:
        out_dir = ROOT / "outputs" / "suite" / "nb15" / "ablation" / "w_o_stage1" / "seed_42" / "GLOBAL"

        shared_root = rs.reviewer_stage1_shared_root(out_dir)

        self.assertTrue(str(shared_root).replace("\\", "/").endswith("nb15/_shared/stage1/seed_42"))

    def test_main_oracle_name_prefers_stage3_then_stage2_then_oracle_models(self) -> None:
        self.assertEqual(rs.main_ids_name({"stage3": {"main_ids_name": "ids3"}}), "ids3")
        self.assertEqual(rs.main_ids_name({"stage2": {"ids_name": "ids2"}}), "ids2")
        self.assertEqual(rs.main_oracle_name({"stage3": {"oracle_name": "s3"}}), "s3")
        self.assertEqual(rs.main_oracle_name({"stage2": {"oracle_name": "s2"}}), "s2")
        self.assertEqual(rs.main_oracle_name({"oracle_models": [{"name": "model_a"}]}), "model_a")
        self.assertEqual(rs.main_oracle_name({}), "mlp_small")


if __name__ == "__main__":
    unittest.main()
