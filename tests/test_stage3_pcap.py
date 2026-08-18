from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.pipeline.stage3_pcap import PcapFeatureExtractionError, PcapFeatureExtractor


class _IdentityPreprocessor:
    def transform(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float32)


class Stage3PcapExtractorTest(unittest.TestCase):
    def _make_extractor(
        self,
        *,
        fail_closed: bool,
        fail_on_partial_alignment: bool = False,
        feature_backend: str = "scapy",
        scapy_available: bool = False,
        nfstream_available: bool = False,
        cicflowmeter_available: bool = False,
        cicflowmeter_cmd: str = "java -jar CICFlowMeter.jar",
        cicflowmeter_timeout: int = 300,
        cache_enable: bool = False,
        cache_dir: Path | None = None,
        pcap_eval_model: object | None = None,
        pcap_eval_model_name: str = "none",
        oracle: object | None = None,
        surrogate: object | None = None,
        max_pcap_bytes: int = 0,
    ) -> PcapFeatureExtractor:
        return PcapFeatureExtractor(
            feature_backend=feature_backend,
            feature_names=["f1", "f2"],
            raw_feature_mean=np.array([1.0, 2.0], dtype=np.float64),
            alias_map={},
            align_min_cov=0.85,
            scapy_available=scapy_available,
            nfstream_available=nfstream_available,
            cicflowmeter_available=cicflowmeter_available,
            cicflowmeter_cmd=cicflowmeter_cmd,
            cicflowmeter_timeout=cicflowmeter_timeout,
            fail_closed=fail_closed,
            fail_on_partial_alignment=fail_on_partial_alignment,
            preprocessor=_IdentityPreprocessor(),
            pcap_eval_model=pcap_eval_model,
            pcap_eval_model_name=pcap_eval_model_name,
            oracle=oracle,
            surrogate=surrogate,
            pcap_eval_batch_size=16,
            seed=42,
            device=torch.device("cpu"),
            max_pcap_bytes=max_pcap_bytes,
            cache_enable=cache_enable,
            cache_dir=cache_dir,
        )

    def test_fail_closed_rejects_fill_value_features(self) -> None:
        extractor = self._make_extractor(fail_closed=True)
        with self.assertRaises(PcapFeatureExtractionError):
            extractor.extract("missing.pcap")

    def test_non_fail_closed_keeps_status_metrics(self) -> None:
        extractor = self._make_extractor(fail_closed=False)
        features, backend, meta = extractor.extract("missing.pcap")
        np.testing.assert_allclose(features, np.array([[1.0, 2.0]], dtype=np.float64))
        self.assertEqual(backend, "none")
        self.assertEqual(meta["status"], "dependency_missing")
        snapshot = extractor.metrics_snapshot()
        self.assertEqual(snapshot["pcap_feature_fill_count"], 1)
        self.assertEqual(snapshot["pcap_feature_status_count_dependency_missing"], 1)

    def test_score_pcap_candidate_skips_oversized_pcap_before_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pcap_path = Path(tmp_dir) / "large.pcap"
            pcap_path.write_bytes(b"x" * 12)
            extractor = self._make_extractor(fail_closed=False, max_pcap_bytes=8)

            row = extractor.score_pcap_candidate(pcap_path)

        self.assertEqual(row["status"], "skipped_too_large")
        self.assertEqual(row["skip_reason"], "pcap_too_large")
        self.assertEqual(row["pcap_size_bytes"], 12)
        self.assertEqual(row["selection_score"], float("-inf"))
        self.assertEqual(extractor.metrics_snapshot()["pcap_feature_fill_count"], 0)

    def test_rank_pcaps_uses_scan_budget_on_under_cap_candidates_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            large = root / "large.pcap"
            small_a = root / "small_a.pcap"
            small_b = root / "small_b.pcap"
            large.write_bytes(b"x" * 128)
            small_a.write_bytes(b"x")
            small_b.write_bytes(b"xx")
            extractor = self._make_extractor(fail_closed=False, max_pcap_bytes=8)

            rows = extractor.rank_pcaps(root, "*.pcap", 2)

        self.assertEqual(len(rows), 2)
        self.assertEqual({row["name"] for row in rows}, {"small_a.pcap", "small_b.pcap"})
        self.assertTrue(all(row["skip_reason"] is None for row in rows))

    def test_partial_alignment_rejection_does_not_require_fail_closed(self) -> None:
        extractor = self._make_extractor(fail_closed=False, fail_on_partial_alignment=True)
        partial_meta = {
            "backend": "nfstream",
            "status": "alignment_partial",
            "alignment": {"missing_features": ["f2"]},
            "used_fill_values": False,
        }
        with self.assertRaises(PcapFeatureExtractionError):
            extractor._finalize("partial.pcap", np.array([[1.0, 2.0]], dtype=np.float64), "nfstream", partial_meta)

    def test_partial_alignment_tolerates_single_missing_feature_when_coverage_is_high(self) -> None:
        extractor = self._make_extractor(fail_closed=False, fail_on_partial_alignment=True)
        partial_meta = {
            "backend": "nfstream",
            "status": "alignment_partial",
            "alignment": {
                "coverage": 0.95,
                "missing": 1,
                "missing_features": ["f2"],
            },
            "used_fill_values": False,
        }
        _, _, meta = extractor._finalize(
            "partial_ok.pcap",
            np.array([[1.0, 2.0]], dtype=np.float64),
            "nfstream",
            partial_meta,
        )
        self.assertEqual(meta["status"], "alignment_partial_tolerated")
        self.assertTrue(meta["alignment_tolerated"])

    def test_rank_pcaps_prefers_hybrid_score_over_raw_pmal(self) -> None:
        extractor = self._make_extractor(fail_closed=False)
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            low_fit = root / "low_fit.pcap"
            high_fit = root / "high_fit.pcap"
            low_fit.write_bytes(b"")
            high_fit.write_bytes(b"")

            fake_rows = {
                low_fit.name: {
                    "path": str(low_fit),
                    "name": low_fit.name,
                    "prob_malicious": 0.90,
                    "pred_label": 1,
                    "selection_score": 0.70,
                    "target_feature_l2": 9.0,
                    "target_mod_l2": 5.0,
                },
                high_fit.name: {
                    "path": str(high_fit),
                    "name": high_fit.name,
                    "prob_malicious": 0.80,
                    "pred_label": 1,
                    "selection_score": 0.85,
                    "target_feature_l2": 1.0,
                    "target_mod_l2": 0.5,
                },
            }

            with mock.patch.object(
                extractor,
                "score_pcap_candidate",
                side_effect=lambda path, **_: fake_rows[path.name],
            ):
                ranked = extractor.rank_pcaps(root, "*.pcap", 10, target_pre=np.array([0.0, 0.0], dtype=np.float32))

            self.assertEqual([row["name"] for row in ranked], [high_fit.name, low_fit.name])
            self.assertGreater(ranked[0]["selection_score"], ranked[1]["selection_score"])

    def test_disk_cache_round_trip_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            pcap = root / "sample.pcap"
            pcap.write_bytes(b"pcap")
            extractor = self._make_extractor(
                fail_closed=False,
                feature_backend="scapy",
                scapy_available=True,
                cache_enable=True,
                cache_dir=root / "cache",
            )
            with mock.patch(
                "rdsynth.pipeline.stage3_pcap.extract_pcap_features_scapy",
                return_value=(
                    np.array([[3.0, 4.0]], dtype=np.float64),
                    {"status": "ok", "used_fill_values": False},
                ),
            ) as extract_scapy:
                feat1, backend1, meta1 = extractor.extract(str(pcap))
            self.assertEqual(backend1, "scapy")
            self.assertEqual(meta1["status"], "ok")
            np.testing.assert_allclose(feat1, np.array([[3.0, 4.0]], dtype=np.float64))
            extract_scapy.assert_called_once()

            extractor2 = self._make_extractor(
                fail_closed=False,
                feature_backend="scapy",
                scapy_available=True,
                cache_enable=True,
                cache_dir=root / "cache",
            )
            with mock.patch(
                "rdsynth.pipeline.stage3_pcap.extract_pcap_features_scapy",
                side_effect=AssertionError("disk cache should serve this read"),
            ):
                feat2, backend2, meta2 = extractor2.extract(str(pcap))
            self.assertEqual(backend2, "scapy")
            self.assertEqual(meta2["status"], "ok")
            np.testing.assert_allclose(feat2, np.array([[3.0, 4.0]], dtype=np.float64))
            snapshot = extractor2.metrics_snapshot()
            self.assertEqual(snapshot["pcap_feature_disk_cache_hits"], 1)

    def test_nfstream_backend_falls_back_to_scapy_on_zero_coverage(self) -> None:
        extractor = self._make_extractor(
            fail_closed=False,
            feature_backend="nfstream",
            nfstream_available=True,
            scapy_available=True,
        )
        with (
            mock.patch(
                "rdsynth.pipeline.stage3_pcap.extract_pcap_features_nfstream",
                return_value=(
                    np.array([[5.0, 6.0]], dtype=np.float64),
                    {"status": "alignment_partial", "alignment": {"coverage": 0.0}, "flow_count": 2},
                ),
            ),
            mock.patch(
                "rdsynth.pipeline.stage3_pcap.extract_pcap_features_scapy",
                return_value=(
                    np.array([[7.0, 8.0]], dtype=np.float64),
                    {
                        "status": "ok",
                        "used_fill_values": False,
                        "alignment": {"coverage": 1.0, "matched": 2, "missing": 0, "missing_features": []},
                    },
                ),
            ),
        ):
            feat, backend, meta = extractor.extract("fallback.pcap")
        self.assertEqual(backend, "scapy")
        self.assertEqual(meta["fallback_from"], "nfstream")
        self.assertEqual(meta["fallback_reason"], "nfstream_zero_coverage")
        self.assertEqual(meta["alignment"]["coverage"], 1.0)
        np.testing.assert_allclose(feat, np.array([[7.0, 8.0]], dtype=np.float64))

    def test_predict_probs_handles_sklearn_oracle_and_classify_pcap_cache(self) -> None:
        oracle_model = mock.Mock()
        oracle_model.predict.return_value = np.array([1, 0], dtype=np.int64)
        oracle = mock.Mock(model_type="xgboost", model=oracle_model)
        extractor = self._make_extractor(
            fail_closed=False,
            pcap_eval_model=object(),
            pcap_eval_model_name="oracle",
            oracle=oracle,
        )
        with mock.patch(
            "rdsynth.pipeline.stage3_pcap.predict_sklearn_probs",
            return_value=np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32),
        ):
            preds, probs = extractor.predict_probs(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        np.testing.assert_array_equal(preds, np.array([1, 0], dtype=np.int64))
        np.testing.assert_allclose(probs, np.array([[0.2, 0.8], [0.7, 0.3]], dtype=np.float32))

        ids_model = mock.Mock()
        ids_model.predict.return_value = np.array([0, 1], dtype=np.int64)
        ids_bundle = mock.Mock(model_type="extra_trees", model=ids_model)
        ids_extractor = self._make_extractor(
            fail_closed=False,
            pcap_eval_model=object(),
            pcap_eval_model_name="ids",
            oracle=oracle,
        )
        ids_extractor.ids = ids_bundle
        with mock.patch(
            "rdsynth.pipeline.stage3_pcap.predict_sklearn_probs",
            return_value=np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32),
        ):
            ids_preds, ids_probs = ids_extractor.predict_probs(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
        np.testing.assert_array_equal(ids_preds, np.array([0, 1], dtype=np.int64))
        np.testing.assert_allclose(ids_probs, np.array([[0.8, 0.2], [0.3, 0.7]], dtype=np.float32))

        cached = self._make_extractor(fail_closed=False)
        with (
            mock.patch.object(
                cached,
                "extract",
                return_value=(
                    np.array([[1.0, 2.0]], dtype=np.float64),
                    "scapy",
                    {"status": "ok"},
                ),
            ) as extract_mock,
            mock.patch.object(
                cached,
                "classify_features",
                return_value=(
                    np.array([0.25, 0.75], dtype=np.float32),
                    np.array([[0.1, 0.2]], dtype=np.float32),
                ),
            ) as classify_mock,
        ):
            first = cached.classify_pcap("cached.pcap")
            second = cached.classify_pcap("cached.pcap")
        extract_mock.assert_called_once()
        classify_mock.assert_called_once()
        np.testing.assert_allclose(first[3], second[3])
        np.testing.assert_allclose(first[4], second[4])

    def test_extract_normalizes_none_meta_from_backend(self) -> None:
        extractor = self._make_extractor(
            fail_closed=False,
            feature_backend="nfstream",
            nfstream_available=True,
            scapy_available=False,
        )
        with mock.patch(
            "rdsynth.pipeline.stage3_pcap.extract_pcap_features_nfstream",
            return_value=(np.array([[5.0, 6.0]], dtype=np.float64), None),
        ):
            feat, backend, meta = extractor.extract("none_meta.pcap")
        self.assertEqual(backend, "nfstream")
        self.assertEqual(meta["status"], "ok")
        self.assertEqual(meta["alignment"], {})
        np.testing.assert_allclose(feat, np.array([[5.0, 6.0]], dtype=np.float64))

    def test_pcap_prob_scan_and_score_candidate_handle_edge_cases(self) -> None:
        extractor = self._make_extractor(fail_closed=False)
        with mock.patch.object(
            extractor,
            "classify_pcap",
            return_value=(
                np.array([[1.0, 2.0]], dtype=np.float64),
                "scapy",
                {"status": "ok"},
                np.array([np.nan, np.nan], dtype=np.float32),
                np.array([[0.0, 0.0]], dtype=np.float32),
            ),
        ):
            pmal, pred, backend, meta = extractor.pcap_prob(Path("bad.pcap"))
        self.assertIsNone(pmal)
        self.assertIsNone(pred)
        self.assertEqual(backend, "scapy")
        self.assertEqual(meta["status"], "ok")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            a = root / "a.pcap"
            b = root / "b.pcap"
            a.write_bytes(b"")
            b.write_bytes(b"")
            values = {a.name: 0.2, b.name: 0.9}
            with mock.patch.object(
                extractor,
                "pcap_prob",
                side_effect=lambda path: (values[path.name], 1, "scapy", {"status": "ok"}),
            ):
                best_path, best_prob, scanned = extractor.scan_pcaps(root, "*.pcap", 2)
        self.assertEqual(best_path.name, "b.pcap")
        self.assertEqual(best_prob, 0.9)
        self.assertEqual(scanned, 2)

        scored = self._make_extractor(fail_closed=False)
        with (
            mock.patch.object(
                scored,
                "classify_pcap",
                return_value=(
                    np.array([[1.0, 2.0]], dtype=np.float64),
                    "scapy",
                    {"status": "ok"},
                    np.array([0.1, 0.9], dtype=np.float32),
                    np.array([[1.5, 2.5]], dtype=np.float32),
                ),
            ),
            mock.patch(
                "rdsynth.pipeline.stage3_pcap.build_remap_targets",
                return_value=np.array([[0.5, 0.25]], dtype=np.float64),
            ),
        ):
            row = scored.score_pcap_candidate(
                Path("candidate.pcap"),
                target_pre=np.array([1.0, 2.0], dtype=np.float64),
                target_mod=np.array([0.0, 0.0], dtype=np.float64),
            )
        self.assertEqual(row["pred_label"], 1)
        self.assertGreater(row["selection_score"], 0.0)
        self.assertIsNotNone(row["target_feature_l2"])
        self.assertIsNotNone(row["target_mod_l2"])


if __name__ == "__main__":
    unittest.main()
