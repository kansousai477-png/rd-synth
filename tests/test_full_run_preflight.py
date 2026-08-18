from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.append(str(SCRIPTS))

import run_full_preflight as preflight


class FullRunPreflightPolicyTest(unittest.TestCase):
    def test_dataset_preflight_checks_global_stage3_policy(self) -> None:
        suite_cfg = preflight.load_yaml(ROOT / "configs" / "reviewer_suite.yaml")
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = preflight._dataset_preflight(
                dataset="nb15",
                suite_cfg=suite_cfg,
                profile_name="paper",
                out_root=Path(tmp_dir),
            )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.extra["sample_stage3_source_mode"], "best")
        self.assertGreater(int(result.extra["sample_stage3_scan_limit"]), 0)
        self.assertGreater(int(result.extra["sample_stage3_scan_max_bytes"]), 0)
        self.assertTrue(result.extra["sample_stage3_semantic_attack_labels"])
        self.assertEqual(result.extra["stage3_policy_errors"], [])


if __name__ == "__main__":
    unittest.main()
