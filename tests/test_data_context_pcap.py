from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.data import _augment_with_malicious_pcaps


class DataContextPcapAugmentTest(unittest.TestCase):
    def test_augment_with_malicious_pcaps_appends_malicious_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcap_path = Path(tmp_dir) / "malicious.pcap"
            pcap_path.write_bytes(b"pcap")
            features = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
            labels = np.array([0, 1], dtype=np.int64)
            data_cfg = {
                "dataset": "cic_unsw",
                "pcap_malicious_path": str(pcap_path),
            }

            with (
                patch(
                    "rdsynth.pipeline.data.importlib.util.find_spec",
                    side_effect=lambda name: object() if name == "nfstream" else None,
                ),
                patch(
                    "rdsynth.pipeline.data.extract_pcap_features_nfstream",
                    return_value=(np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64), {"status": "ok"}),
                ),
                patch("rdsynth.pipeline.data._check_java_available", return_value=False),
            ):
                merged_features, merged_labels = _augment_with_malicious_pcaps(features, labels, data_cfg)

        self.assertEqual(merged_features.shape, (4, 2))
        np.testing.assert_array_equal(merged_labels, np.array([0, 1, 1, 1], dtype=np.int64))
        self.assertEqual(float(merged_features.iloc[-1]["f2"]), 40.0)

    def test_augment_with_malicious_pcaps_respects_flow_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcap_path = Path(tmp_dir) / "malicious.pcap"
            pcap_path.write_bytes(b"pcap")
            features = pd.DataFrame({"f1": [1.0], "f2": [2.0]})
            labels = np.array([0], dtype=np.int64)
            data_cfg = {
                "dataset": "cic_unsw",
                "pcap_malicious_path": str(pcap_path),
                "pcap_malicious_max_flows_per_pcap": 1,
            }

            with (
                patch(
                    "rdsynth.pipeline.data.importlib.util.find_spec",
                    side_effect=lambda name: object() if name == "nfstream" else None,
                ),
                patch(
                    "rdsynth.pipeline.data.extract_pcap_features_nfstream",
                    return_value=(np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64), {"status": "ok"}),
                ),
                patch("rdsynth.pipeline.data._check_java_available", return_value=False),
            ):
                merged_features, merged_labels = _augment_with_malicious_pcaps(features, labels, data_cfg)

        self.assertEqual(merged_features.shape, (2, 2))
        np.testing.assert_array_equal(merged_labels, np.array([0, 1], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
