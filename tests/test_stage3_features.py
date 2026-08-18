from __future__ import annotations

import builtins
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.stages.stage3_features import (
    _nfstream_alignment_specs,
    extract_pcap_features_nfstream,
    extract_pcap_features_scapy,
)
from rdsynth.utils.feature_align import alignment_report


class Stage3FeatureExtractionMetaTest(unittest.TestCase):
    def test_nfstream_alignment_specs_cover_cic_ids2017_name_variants(self) -> None:
        feature_names = [
            "Destination Port",
            "Flow Duration",
            "Total Fwd Packets",
            "Total Backward Packets",
            "Total Length of Fwd Packets",
            "Total Length of Bwd Packets",
            "Min Packet Length",
            "Max Packet Length",
            "Avg Fwd Segment Size",
            "Avg Bwd Segment Size",
            "Fwd Header Length.1",
            "Init_Win_bytes_forward",
            "Init_Win_bytes_backward",
            "act_data_pkt_fwd",
            "min_seg_size_forward",
        ]
        source_cols = [
            "dst_port",
            "bidirectional_duration_ms",
            "src2dst_packets",
            "dst2src_packets",
            "src2dst_bytes",
            "dst2src_bytes",
            "src2dst_min_ps",
            "bidirectional_min_ps",
            "bidirectional_max_ps",
            "src2dst_mean_ps",
            "dst2src_mean_ps",
            "src2dst_init_win_bytes",
            "dst2src_init_win_bytes",
        ]
        aliases, _, derived = _nfstream_alignment_specs()

        report = alignment_report(source_cols, feature_names, alias_map=aliases, derived=derived)

        self.assertEqual(report["missing"], 0)
        self.assertEqual(report["coverage"], 1.0)

    def test_statistical_aliases_cover_cic_ids2018_short_names(self) -> None:
        from rdsynth.utils.feature_align import build_statistical_feature_aliases

        feature_names = [
            "Dst Port",
            "Protocol",
            "Flow Duration",
            "Tot Fwd Pkts",
            "Tot Bwd Pkts",
            "TotLen Fwd Pkts",
            "TotLen Bwd Pkts",
            "Fwd Pkt Len Max",
            "Fwd Pkt Len Min",
            "Fwd Pkt Len Mean",
            "Fwd Pkt Len Std",
            "Bwd Pkt Len Max",
            "Bwd Pkt Len Min",
            "Bwd Pkt Len Mean",
            "Bwd Pkt Len Std",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Flow IAT Max",
            "Flow IAT Min",
            "Fwd IAT Tot",
            "Fwd IAT Mean",
            "Fwd IAT Std",
            "Fwd IAT Max",
            "Fwd IAT Min",
            "Bwd IAT Tot",
            "Bwd IAT Mean",
            "Bwd IAT Std",
            "Bwd IAT Max",
            "Bwd IAT Min",
            "Fwd PSH Flags",
            "Fwd URG Flags",
            "Fwd Header Len",
            "Bwd Header Len",
            "Fwd Pkts/s",
            "Bwd Pkts/s",
            "Pkt Len Min",
            "Pkt Len Max",
            "Pkt Len Mean",
            "Pkt Len Std",
            "Pkt Len Var",
            "FIN Flag Cnt",
            "SYN Flag Cnt",
            "RST Flag Cnt",
            "PSH Flag Cnt",
            "ACK Flag Cnt",
            "URG Flag Cnt",
            "ECE Flag Cnt",
            "Down/Up Ratio",
            "Pkt Size Avg",
            "Fwd Seg Size Avg",
            "Bwd Seg Size Avg",
            "Subflow Fwd Pkts",
            "Subflow Fwd Byts",
            "Subflow Bwd Pkts",
            "Subflow Bwd Byts",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts",
            "Fwd Act Data Pkts",
            "Fwd Seg Size Min",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
        ]
        source_cols = [
            "dst_port",
            "protocol",
            "bidirectional_duration_ms",
            "src2dst_packets",
            "dst2src_packets",
            "src2dst_bytes",
            "dst2src_bytes",
            "src2dst_max_ps",
            "src2dst_min_ps",
            "src2dst_mean_ps",
            "src2dst_stddev_ps",
            "dst2src_max_ps",
            "dst2src_min_ps",
            "dst2src_mean_ps",
            "dst2src_stddev_ps",
            "bidirectional_mean_piat_ms",
            "bidirectional_stddev_piat_ms",
            "bidirectional_max_piat_ms",
            "bidirectional_min_piat_ms",
            "src2dst_duration_ms",
            "src2dst_mean_piat_ms",
            "src2dst_stddev_piat_ms",
            "src2dst_max_piat_ms",
            "src2dst_min_piat_ms",
            "dst2src_duration_ms",
            "dst2src_mean_piat_ms",
            "dst2src_stddev_piat_ms",
            "dst2src_max_piat_ms",
            "dst2src_min_piat_ms",
            "src2dst_psh_packets",
            "src2dst_urg_packets",
            "bidirectional_min_ps",
            "bidirectional_max_ps",
            "bidirectional_mean_ps",
            "bidirectional_stddev_ps",
            "bidirectional_fin_packets",
            "bidirectional_syn_packets",
            "bidirectional_rst_packets",
            "bidirectional_psh_packets",
            "bidirectional_ack_packets",
            "bidirectional_urg_packets",
            "bidirectional_ece_packets",
            "src2dst_init_win_bytes",
            "dst2src_init_win_bytes",
        ]

        alias_map = build_statistical_feature_aliases(feature_names, dataset_name="cic_ids2018")
        fill_values = np.zeros((len(feature_names),), dtype=np.float64)
        flow_df = pd.DataFrame({name: [1.0] for name in source_cols})

        class _FakeNFStreamer:
            def __init__(self, source, statistical_analysis=True, max_nflows=None):
                self.source = source
                self.statistical_analysis = statistical_analysis

            def to_pandas(self):
                return flow_df

        fake_nfstream = types.SimpleNamespace(NFStreamer=_FakeNFStreamer)
        scapy_supplement = np.zeros((1, len(feature_names)), dtype=np.float64)
        for name in [
            "Fwd Header Len",
            "Bwd Header Len",
            "Down/Up Ratio",
            "Fwd Seg Size Avg",
            "Bwd Seg Size Avg",
            "Init Fwd Win Byts",
            "Init Bwd Win Byts",
            "Fwd Act Data Pkts",
            "Active Mean",
            "Active Std",
            "Active Max",
            "Active Min",
            "Idle Mean",
            "Idle Std",
            "Idle Max",
            "Idle Min",
        ]:
            scapy_supplement[0, feature_names.index(name)] = 1.0

        with (
            mock.patch.dict(sys.modules, {"nfstream": fake_nfstream}),
            mock.patch(
                "rdsynth.stages.stage3_features.extract_pcap_features_scapy",
                return_value=scapy_supplement,
            ),
        ):
            _, meta = extract_pcap_features_nfstream(
                "synthetic.pcap",
                feature_names,
                fill_values,
                alias_map=alias_map,
                return_meta=True,
            )

        self.assertEqual(meta["alignment"]["missing_features"], [])
        self.assertEqual(meta["alignment"]["coverage"], 1.0)
        self.assertTrue(meta["supplemented_from_scapy"])

    def test_scapy_extraction_reports_dependency_missing(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("scapy"):
                raise ImportError("scapy missing for test")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            features, meta = extract_pcap_features_scapy(
                "missing.pcap",
                ["f1", "f2"],
                np.array([1.0, 2.0], dtype=np.float64),
                return_meta=True,
            )

        np.testing.assert_allclose(features, np.array([[1.0, 2.0]], dtype=np.float64))
        self.assertEqual(meta["backend"], "scapy")
        self.assertEqual(meta["status"], "dependency_missing")
        self.assertTrue(meta["used_fill_values"])

    def test_nfstream_extraction_reports_dependency_missing(self) -> None:
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "nfstream":
                raise ImportError("nfstream missing for test")
            return original_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            features, meta = extract_pcap_features_nfstream(
                "missing.pcap",
                ["f1", "f2"],
                np.array([3.0, 4.0], dtype=np.float64),
                return_meta=True,
            )

        np.testing.assert_allclose(features, np.array([[3.0, 4.0]], dtype=np.float64))
        self.assertEqual(meta["backend"], "nfstream")
        self.assertEqual(meta["status"], "dependency_missing")
        self.assertTrue(meta["used_fill_values"])

    def test_nfstream_meta_marks_scapy_supplement_only_when_applied(self) -> None:
        feature_names = ["f1", "Active Mean"]
        fill_values = np.array([0.0, 0.0], dtype=np.float64)
        flow_df = pd.DataFrame({"f1": [7.0]})

        class _FakeNFStreamer:
            def __init__(self, source, statistical_analysis=True, max_nflows=None):
                self.source = source
                self.statistical_analysis = statistical_analysis

            def to_pandas(self):
                return flow_df

        def fake_alignment_report(columns, requested, alias_map=None, derived=None):
            return {
                "matched": 1,
                "missing": 1,
                "total": 2,
                "coverage": 0.5,
                "missing_features": ["Active Mean"],
            }

        fake_nfstream = types.SimpleNamespace(NFStreamer=_FakeNFStreamer)
        with (
            mock.patch.dict(sys.modules, {"nfstream": fake_nfstream}),
            mock.patch("rdsynth.stages.stage3_features.alignment_report", side_effect=fake_alignment_report),
            mock.patch(
                "rdsynth.stages.stage3_features.extract_pcap_features_scapy",
                return_value=np.array([[7.0, 11.0]], dtype=np.float64),
            ),
        ):
            features, meta = extract_pcap_features_nfstream(
                "synthetic.pcap",
                feature_names,
                fill_values,
                return_meta=True,
            )

        np.testing.assert_allclose(features, np.array([[7.0, 11.0]], dtype=np.float64))
        self.assertFalse(meta["alignment"]["missing_features"])
        self.assertTrue(meta["supplemented_from_scapy"])
        self.assertEqual(meta["status"], "ok")

    def test_scapy_extraction_populates_generic_feature_schema(self) -> None:
        feature_names = [
            "Header_Length",
            "Protocol Type",
            "Time_To_Live",
            "Rate",
            "psh_flag_number",
            "ack_flag_number",
            "HTTP",
            "TCP",
            "Tot size",
            "AVG",
            "IAT",
            "Number",
            "Variance",
        ]
        fill_values = np.zeros((len(feature_names),), dtype=np.float64)

        class _FakeIP:
            def __init__(self, src, dst, ttl=64, proto=6):
                self.src = src
                self.dst = dst
                self.ttl = ttl
                self.proto = proto
                self.ihl = 5

        class _FakeTCP:
            def __init__(self, sport, dport, flags=0x18):
                self.sport = sport
                self.dport = dport
                self.flags = flags
                self.window = 1024
                self.dataofs = 5
                self.payload = b"abc"

        class _FakePkt:
            def __init__(self, t, src="1.1.1.1", dst="2.2.2.2"):
                self.time = t
                self.ip = _FakeIP(src, dst)
                self.tcp = _FakeTCP(1234, 80)

            def __contains__(self, item):
                return item in {_FakeIPSentinel, _FakeTCPSentinel}

            def __getitem__(self, item):
                if item is _FakeIPSentinel:
                    return self.ip
                if item is _FakeTCPSentinel:
                    return self.tcp
                raise KeyError(item)

            def __bytes__(self):
                return b"x" * 60

        _FakeIPSentinel = object()
        _FakeTCPSentinel = object()

        def fake_rdpcap(_path):
            return [_FakePkt(0.0), _FakePkt(0.1)]

        fake_scapy = types.SimpleNamespace(IP=_FakeIPSentinel, TCP=_FakeTCPSentinel, UDP=object(), rdpcap=fake_rdpcap)
        with mock.patch.dict(sys.modules, {"scapy.all": fake_scapy}):
            features, meta = extract_pcap_features_scapy(
                "synthetic.pcap",
                feature_names,
                fill_values,
                return_meta=True,
            )

        self.assertEqual(meta["status"], "ok")
        self.assertEqual(features.shape, (1, len(feature_names)))
        row = features[0]
        self.assertGreater(row[feature_names.index("Header_Length")], 0.0)
        self.assertEqual(row[feature_names.index("Protocol Type")], 6.0)
        self.assertEqual(row[feature_names.index("HTTP")], 1.0)
        self.assertEqual(row[feature_names.index("TCP")], 1.0)
        self.assertEqual(row[feature_names.index("Number")], 2.0)
        self.assertEqual(meta["alignment"]["missing"], 0)
        self.assertEqual(meta["alignment"]["coverage"], 1.0)

    def test_scapy_alignment_reports_missing_unavailable_features(self) -> None:
        feature_names = ["Header_Length", "Protocol Type", "CWE Flag Count"]
        fill_values = np.zeros((len(feature_names),), dtype=np.float64)

        class _FakeIP:
            def __init__(self, src, dst, ttl=64, proto=6):
                self.src = src
                self.dst = dst
                self.ttl = ttl
                self.proto = proto
                self.ihl = 5

        class _FakeTCP:
            def __init__(self, sport, dport, flags=0x18):
                self.sport = sport
                self.dport = dport
                self.flags = flags
                self.window = 1024
                self.dataofs = 5
                self.payload = b"abc"

        class _FakePkt:
            def __init__(self, t):
                self.time = t
                self.ip = _FakeIP("1.1.1.1", "2.2.2.2")
                self.tcp = _FakeTCP(1234, 80)

            def __contains__(self, item):
                return item in {_FakeIPSentinel, _FakeTCPSentinel}

            def __getitem__(self, item):
                if item is _FakeIPSentinel:
                    return self.ip
                if item is _FakeTCPSentinel:
                    return self.tcp
                raise KeyError(item)

            def __bytes__(self):
                return b"x" * 60

        _FakeIPSentinel = object()
        _FakeTCPSentinel = object()

        def fake_rdpcap(_path):
            return [_FakePkt(0.0), _FakePkt(0.1)]

        fake_scapy = types.SimpleNamespace(IP=_FakeIPSentinel, TCP=_FakeTCPSentinel, UDP=object(), rdpcap=fake_rdpcap)
        with mock.patch.dict(sys.modules, {"scapy.all": fake_scapy}):
            _, meta = extract_pcap_features_scapy(
                "synthetic.pcap",
                feature_names,
                fill_values,
                return_meta=True,
            )

        self.assertEqual(meta["alignment"]["matched"], 2)
        self.assertEqual(meta["alignment"]["missing"], 1)
        self.assertIn("CWE Flag Count", meta["alignment"]["missing_features"])


if __name__ == "__main__":
    unittest.main()
