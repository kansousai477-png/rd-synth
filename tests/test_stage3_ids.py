from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_ids import build_stage3_ids_model_cfg, train_stage3_ids


class _IdentityPreprocessor:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


class Stage3IdsTest(unittest.TestCase):
    def test_build_stage3_ids_model_cfg_defaults_to_extra_trees(self) -> None:
        settings = SimpleNamespace(
            pcap_ids_model_type="extra_trees",
            pcap_ids_hidden_dims=[64, 32],
            pcap_ids_epochs=3,
            pcap_ids_batch_size=16,
            pcap_ids_lr=1.0e-3,
            pcap_ids_max_batches_per_epoch=4,
        )
        cfg = build_stage3_ids_model_cfg(settings)
        self.assertEqual(cfg["type"], "extra_trees")
        self.assertEqual(cfg["class_weight"], "balanced")

    def test_train_stage3_ids_uses_selected_malicious_and_benign_pcap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            malicious = root / "malicious.pcap"
            benign = root / "benign.pcap"
            malicious.write_bytes(b"pcap")
            benign.write_bytes(b"pcap")
            settings = SimpleNamespace(
                ids_name="pcap_ids",
                pcap_ids_benign_path=str(benign),
                pcap_ids_benign_paths=[],
                pcap_ids_benign_dir="",
                pcap_ids_benign_glob="*.pcap",
                pcap_ids_benign_max_pcaps=1,
                pcap_ids_benign_max_flows_per_pcap=8,
                pcap_ids_malicious_max_flows_per_pcap=8,
                pcap_ids_feature_backend="nfstream",
                feature_backend="nfstream",
                pcap_ids_model_type="extra_trees",
                pcap_ids_hidden_dims=[64, 32],
                pcap_ids_epochs=2,
                pcap_ids_batch_size=8,
                pcap_ids_lr=1.0e-3,
                pcap_ids_max_batches_per_epoch=4,
            )
            with (
                mock.patch("rdsynth.pipeline.stage3_ids.importlib.util.find_spec", side_effect=[object(), object()]),
                mock.patch(
                    "rdsynth.pipeline.stage3_ids.extract_pcap_features_nfstream",
                    side_effect=[
                        (np.asarray([[9.0, 9.5], [8.5, 8.0]], dtype=np.float32), {"status": "ok"}),
                        (np.asarray([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4]], dtype=np.float32), {"status": "ok"}),
                    ],
                ),
                mock.patch(
                    "rdsynth.pipeline.stage3_ids.train_oracle_from_config",
                    return_value=(SimpleNamespace(model_type="extra_trees", model=object()), 1.0),
                ) as train_mock,
            ):
                result = train_stage3_ids(
                    malicious_pcap=malicious,
                    settings=settings,
                    feature_names=["f0", "f1"],
                    raw_feature_mean=np.asarray([0.0, 0.0], dtype=np.float32),
                    alias_map={},
                    preprocessor=_IdentityPreprocessor(),
                    device="cpu",
                    seed=7,
                )
            self.assertIsNotNone(result.ids_bundle)
            self.assertEqual(result.metrics["pcap_ids_malicious_rows"], 2)
            self.assertEqual(result.metrics["pcap_ids_benign_rows"], 3)
            self.assertEqual(result.metrics["pcap_ids_model_type"], "extra_trees")
            train_mock.assert_called_once()

    def test_train_stage3_ids_can_use_bounded_malicious_pool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            malicious_a = root / "malicious_a.pcap"
            malicious_b = root / "malicious_b.pcap"
            benign = root / "benign.pcap"
            for path in (malicious_a, malicious_b, benign):
                path.write_bytes(b"pcap")
            settings = SimpleNamespace(
                ids_name="pcap_ids",
                pcap_ids_benign_path=str(benign),
                pcap_ids_benign_paths=[],
                pcap_ids_benign_dir="",
                pcap_ids_benign_glob="*.pcap",
                pcap_ids_benign_max_pcaps=1,
                pcap_ids_benign_max_flows_per_pcap=8,
                pcap_ids_malicious_max_flows_per_pcap=8,
                pcap_ids_feature_backend="nfstream",
                feature_backend="nfstream",
                pcap_ids_model_type="extra_trees",
                pcap_ids_hidden_dims=[64, 32],
                pcap_ids_epochs=2,
                pcap_ids_batch_size=8,
                pcap_ids_lr=1.0e-3,
                pcap_ids_max_batches_per_epoch=4,
            )
            with (
                mock.patch("rdsynth.pipeline.stage3_ids.importlib.util.find_spec", side_effect=[object(), object()]),
                mock.patch(
                    "rdsynth.pipeline.stage3_ids.extract_pcap_features_nfstream",
                    side_effect=[
                        (np.asarray([[9.0, 9.5]], dtype=np.float32), {"status": "ok"}),
                        (np.asarray([[8.5, 8.0]], dtype=np.float32), {"status": "ok"}),
                        (np.asarray([[0.1, 0.2], [0.2, 0.3]], dtype=np.float32), {"status": "ok"}),
                    ],
                ),
                mock.patch(
                    "rdsynth.pipeline.stage3_ids.train_oracle_from_config",
                    return_value=(SimpleNamespace(model_type="extra_trees", model=object()), 1.0),
                ),
            ):
                result = train_stage3_ids(
                    malicious_pcap=malicious_a,
                    malicious_pcaps=[malicious_a, malicious_b],
                    settings=settings,
                    feature_names=["f0", "f1"],
                    raw_feature_mean=np.asarray([0.0, 0.0], dtype=np.float32),
                    alias_map={},
                    preprocessor=_IdentityPreprocessor(),
                    device="cpu",
                    seed=7,
                )

        self.assertEqual(result.metrics["pcap_ids_malicious_rows"], 2)
        self.assertEqual(result.metrics["pcap_ids_malicious_pcaps_used"], [str(malicious_a), str(malicious_b)])


if __name__ == "__main__":
    unittest.main()
