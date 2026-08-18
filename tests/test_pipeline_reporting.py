from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.reporting import (
    make_stage2_paper_table_records,
    make_stage3_pcap_table_records,
    stage3_evidence_summary,
)


class PipelineReportingSmokeTest(unittest.TestCase):
    def test_pipeline_summary_writes_paper_tables(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            out_dir = root / "outputs" / "paper"
            (out_dir / "stage1" / "mlp_small").mkdir(parents=True)
            (out_dir / "stage1" / "data_quality").mkdir(parents=True)
            (out_dir / "stage2").mkdir(parents=True)
            (out_dir / "stage3").mkdir(parents=True)

            (out_dir / "stage1" / "mlp_small" / "metrics.json").write_text(
                json.dumps(
                    {
                        "oracle_eval_acc": 0.97,
                        "oracle_eval_f1": 0.96,
                        "agreement": 0.99,
                        "surrogate_val_acc": 0.95,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "stage1" / "data_quality" / "metrics.json").write_text(
                json.dumps({"rows": 1000, "features": 10, "label_positive_rate": 0.1}),
                encoding="utf-8",
            )
            (out_dir / "stage2" / "metrics.json").write_text(
                json.dumps(
                    {
                        "asr_oracle": 0.9,
                        "asr_surrogate": 0.92,
                        "adv_prob_malicious_mean_oracle": 0.1,
                        "adv_prob_malicious_mean": 0.08,
                        "norm_FFD": 12.0,
                        "norm_SWD": 0.2,
                        "norm_C2ST-AUC": 0.7,
                        "norm_AdvToMal_L2": 3.0,
                        "sample_generation_time_sec": 0.5,
                        "baseline_global_random_asr_oracle": 0.8,
                        "baseline_global_random_asr_surrogate": 0.81,
                        "baseline_global_random_adv_pmal_oracle": 0.2,
                        "baseline_global_random_adv_pmal_surrogate": 0.19,
                        "baseline_global_random_norm_FFD": 8.0,
                        "baseline_global_random_norm_SWD": 0.15,
                        "baseline_global_random_norm_C2ST-AUC": 0.6,
                        "baseline_global_random_norm_AdvToMal_L2": 5.0,
                        "baseline_global_random_time_cost_sec": 0.02,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "stage3" / "metrics.json").write_text(
                json.dumps(
                    {
                        "paper_pcap_attack_success_rate": 1.0,
                        "paper_pcap_detection_rate": 0.0,
                        "pcap_adv_prob_malicious_mean": 0.4,
                        "pcap_target_l2_mean": 20.0,
                        "pcap_target_mae_mean": 2.0,
                        "paper_pcap_alignment_coverage": 1.0,
                        "pcap_written_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (out_dir / "stage3" / "pcap_eval.csv").write_text(
                "pcap,prob_malicious,pred_label\norig.pcap,0.99,1\nadv_0000.pcap,0.40,0\n",
                encoding="utf-8",
            )

            cfg = {
                "project": {"out_dir": str(out_dir), "seed": 42, "device": "cpu"},
                "data": {"dataset": "toy"},
                "stage1": {},
                "stage2": {"oracle_name": "mlp_small"},
                "stage3": {"pcap_path": "orig.pcap"},
                "oracle_models": [{"name": "mlp_small"}],
            }
            cfg_path = root / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_pipeline.py"),
                    "--config",
                    str(cfg_path),
                    "--skip-stage1",
                    "--skip-stage2",
                    "--skip-stage3",
                ],
                check=True,
                cwd=ROOT,
            )

            stage2_table = out_dir / "pipeline" / "paper_stage2_table.csv"
            stage3_table = out_dir / "pipeline" / "paper_stage3_pcap_table.csv"
            metadata_path = out_dir / "pipeline" / "run_metadata.json"
            self.assertTrue(stage2_table.exists())
            self.assertTrue(stage3_table.exists())
            self.assertTrue(metadata_path.exists())
            stage2_text = stage2_table.read_text(encoding="utf-8")
            self.assertIn("method,family,baseline_level,feature_space,traffic_space", stage2_text)
            self.assertIn("asr_oracle", stage2_text)
            self.assertIn("global_random,control", stage2_text)
            stage3_text = stage3_table.read_text(encoding="utf-8")
            self.assertIn(
                "method,family,baseline_level,baseline_group,evaluation_mode,score_scope,evidence_scope,full_evidence",
                stage3_text,
            )
            self.assertIn("RDSynth,RDSynth,", stage3_text)
            self.assertIn("1.000000", stage3_text)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["project"]["seed"], 42)
            self.assertIn(metadata["oracle_name"], {"mlp_small", "pcap_ids"})
            self.assertEqual(metadata["dataset"], "toy")
            self.assertEqual(metadata["stage"], "pipeline")
            self.assertEqual(metadata["rq"], "RQ1-RQ5")
            self.assertEqual(metadata["status"], "success")
            self.assertIn("config_hash", metadata)
            self.assertIn("config_sha256", metadata["project"]["runtime"])

    def test_stage3_pcap_table_marks_remap_only_scope(self) -> None:
        records = make_stage3_pcap_table_records(
            {
                "stage3_evidence_scope": "remap_only_evidence",
                "pcap_skip_reason": "pcap_not_found",
            }
        )
        self.assertEqual(records[0]["score_scope"], "remap_only")
        self.assertEqual(records[0]["evidence_scope"], "remap_only_evidence")
        self.assertEqual(records[0]["full_evidence"], "False")
        self.assertEqual(records[0]["pcap_skip_reason"], "pcap_not_found")

    def test_stage3_pcap_table_blocks_degraded_feature_metrics(self) -> None:
        records = make_stage3_pcap_table_records(
            {
                "pcap_feature_quality_strict": False,
                "pcap_feature_quality_block_reason": "feature_fallback_used",
                "pcap_eval": True,
                "pcap_modified": True,
                "paper_pcap_attack_success_rate": 1.0,
                "paper_pcap_detection_rate": 0.0,
                "pcap_adv_prob_malicious_mean": 0.1,
                "paper_pcap_alignment_coverage": 0.9,
                "pcap_written_count": 2,
                "stage3_evidence_scope": "remap_only_evidence",
                "stage3_evidence_block_reason": "feature_fallback_used",
            }
        )
        self.assertEqual(records[0]["pcap_status"], "degraded_features")
        self.assertEqual(records[0]["pcap_skip_reason"], "feature_fallback_used")
        self.assertEqual(records[0]["evidence_block_reason"], "feature_fallback_used")
        self.assertEqual(records[0]["pcap_attack_success_rate"], "")

    def test_stage3_evidence_summary_marks_full_evidence(self) -> None:
        evidence = stage3_evidence_summary(
            {
                "stage3_evidence_scope": "full_evidence",
                "pcap_feature_quality_strict": True,
            }
        )
        self.assertEqual(evidence["evidence_scope"], "full_evidence")
        self.assertTrue(evidence["full_evidence"])

    def test_paper_tables_include_baseline_credibility_levels(self) -> None:
        stage2_records = make_stage2_paper_table_records(
            {
                "asr_oracle": 0.9,
                "asr_surrogate": 0.91,
                "adv_prob_malicious_mean_oracle": 0.2,
                "adv_prob_malicious_mean": 0.1,
                "norm_FFD": 10.0,
                "norm_SWD": 0.2,
                "norm_C2ST-AUC": 0.8,
                "norm_AdvToMal_L2": 2.0,
                "sample_generation_time_sec": 0.5,
                "baseline_global_random_asr_oracle": 0.8,
                "baseline_global_random_asr_surrogate": 0.82,
                "baseline_global_random_adv_pmal_oracle": 0.3,
                "baseline_global_random_adv_pmal_surrogate": 0.25,
                "baseline_global_random_norm_FFD": 5.0,
                "baseline_global_random_norm_SWD": 0.1,
                "baseline_global_random_norm_C2ST-AUC": 0.7,
                "baseline_global_random_norm_AdvToMal_L2": 4.0,
                "baseline_global_random_time_cost_sec": 0.02,
            }
        )
        self.assertEqual(stage2_records[1]["baseline_level"], "control")

        stage3_records = make_stage3_pcap_table_records(
            {
                "baseline_pgd_paper_pcap_attack_success_rate": 0.5,
                "baseline_pgd_paper_pcap_detection_rate": 0.5,
                "baseline_pgd_pcap_adv_prob_malicious_mean": 0.5,
                "baseline_pgd_pcap_target_l2_mean": 1.0,
                "baseline_pgd_pcap_target_mae_mean": 0.5,
                "baseline_pgd_paper_pcap_alignment_coverage": 1.0,
            }
        )
        self.assertEqual(stage3_records[0]["baseline_level"], "standard")

    def test_stage2_paper_table_prefers_end_to_end_throughput_semantics(self) -> None:
        records = make_stage2_paper_table_records(
            {
                "asr_oracle": 0.9,
                "asr_surrogate": 0.91,
                "adv_prob_malicious_mean_oracle": 0.2,
                "adv_prob_malicious_mean": 0.1,
                "norm_FFD": 10.0,
                "norm_SWD": 0.2,
                "norm_C2ST-AUC": 0.8,
                "norm_AdvToMal_L2": 2.0,
                "sample_generation_time_sec": 0.5,
                "sample_end_to_end_time_sec": 20.0,
                "sample_generation_samples_per_sec": 2000.0,
                "sample_end_to_end_samples_per_sec": 50.0,
                "baseline_global_random_asr_oracle": 0.8,
                "baseline_global_random_asr_surrogate": 0.82,
                "baseline_global_random_adv_pmal_oracle": 0.3,
                "baseline_global_random_adv_pmal_surrogate": 0.25,
                "baseline_global_random_norm_FFD": 5.0,
                "baseline_global_random_norm_SWD": 0.1,
                "baseline_global_random_norm_C2ST-AUC": 0.7,
                "baseline_global_random_norm_AdvToMal_L2": 4.0,
                "baseline_global_random_attack_time_cost_sec": 0.02,
                "baseline_global_random_end_to_end_time_sec": 0.10,
                "baseline_global_random_samples_per_sec": 10000.0,
                "baseline_global_random_end_to_end_samples_per_sec": 10000.0,
            }
        )
        self.assertEqual(records[0]["time_cost_sec"], "0.500000")
        self.assertEqual(records[0]["end_to_end_time_sec"], "20.000000")
        self.assertEqual(records[0]["samples_per_sec"], "50.000000")
        self.assertEqual(records[1]["time_cost_sec"], "0.020000")
        self.assertEqual(records[1]["end_to_end_time_sec"], "0.100000")


if __name__ == "__main__":
    unittest.main()
