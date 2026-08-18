from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.stages.stage3_remap import (
    _build_port_vocab,
    _compute_r2_from_sse,
    _compute_timing_schedule,
    _decode_ports,
    _encode_ports,
    apply_mod_using_scapy,
    build_random_remap_modifications,
    build_rule_based_modifications,
)


class Stage3RemapUtilsTest(unittest.TestCase):
    def test_port_vocab_keeps_frequent_ports_and_http_default(self) -> None:
        ports = np.array([443, 443, 443, 53, 53, 8080], dtype=np.int64)
        vocab = _build_port_vocab(ports, max_classes=2)
        self.assertIn(443, vocab.tolist())
        self.assertIn(80, vocab.tolist())

    def test_encode_ports_maps_to_nearest_vocab(self) -> None:
        vocab = np.array([80, 443, 8080], dtype=np.int64)
        encoded = _encode_ports(np.array([81, 450, 9000], dtype=np.int64), vocab)
        np.testing.assert_array_equal(encoded, np.array([0, 1, 2], dtype=np.int64))

    def test_decode_ports_uses_argmax(self) -> None:
        vocab = np.array([80, 443, 8080], dtype=np.int64)
        logits = np.array([[0.1, 2.0, 0.0], [0.2, 0.1, 3.0]], dtype=np.float32)
        decoded = _decode_ports(logits, vocab)
        np.testing.assert_array_equal(decoded, np.array([443.0, 8080.0], dtype=np.float32))

    def test_rule_based_modifications_project_ports_to_benign_vocab(self) -> None:
        feature_names = [
            "Dst Port",
            "ACK Flag Count",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Average Packet Size",
            "Flow Duration",
            "Total Packets",
        ]
        x_ben = np.array(
            [
                [80, 1, 10.0, 2.0, 100.0, 500.0, 10.0],
                [443, 0, 20.0, 3.0, 120.0, 800.0, 12.0],
            ],
            dtype=np.float32,
        )
        x_adv = np.array(
            [
                [9999, 4, 5000.0, 4000.0, 2000.0, 90000.0, 1.0],
            ],
            dtype=np.float32,
        )
        mods = build_rule_based_modifications(x_adv, x_ben, feature_names, top_port_k=2)
        self.assertIn(int(mods[0, 3]), {80, 443})
        self.assertGreaterEqual(mods[0, 4], 0.0)
        self.assertLessEqual(mods[0, 4], 1.0)
        self.assertGreaterEqual(mods[0, 6], 0.25)
        self.assertLessEqual(mods[0, 6], 4.0)

    def test_rule_based_modifications_preserve_relative_order_within_benign_band(self) -> None:
        feature_names = [
            "Dst Port",
            "ACK Flag Count",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Average Packet Size",
            "Flow Duration",
            "Total Packets",
        ]
        x_ben = np.array(
            [
                [80, 1, 10.0, 1.0, 100.0, 500.0, 10.0],
                [80, 1, 20.0, 2.0, 120.0, 800.0, 12.0],
                [443, 1, 30.0, 3.0, 140.0, 900.0, 14.0],
            ],
            dtype=np.float32,
        )
        x_adv = np.array(
            [
                [9999, 5, 1000.0, 10.0, 1000.0, 10000.0, 5.0],
                [9999, 5, 2000.0, 20.0, 1500.0, 20000.0, 6.0],
            ],
            dtype=np.float32,
        )
        mods = build_rule_based_modifications(x_adv, x_ben, feature_names, top_port_k=2)
        self.assertLess(mods[0, 0], mods[1, 0])
        self.assertLessEqual(mods[0, 2], mods[1, 2])

    def test_compute_timing_schedule_preserves_shape_and_scales_deterministically(self) -> None:
        times = np.array([0.0, 0.1, 0.2, 0.4], dtype=np.float64)
        schedule = _compute_timing_schedule(
            times,
            mean_iat_s=0.2,
            std_iat_s=0.0,
            flow_scale=0.5,
            apply_mean_std=True,
        )
        np.testing.assert_allclose(schedule, np.array([0.0, 0.1, 0.1, 0.1], dtype=np.float64), atol=1.0e-8)

    def test_compute_r2_from_sse_matches_standard_definition(self) -> None:
        target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        pred = np.array([[1.0, 1.0], [2.0, 4.0]], dtype=np.float64)
        sse = float(np.sum((pred - target) ** 2))
        target_var = float(np.sum((target - np.mean(target, axis=0)) ** 2)) + 1.0e-6
        self.assertAlmostEqual(_compute_r2_from_sse(sse, target_var), 1.0 - (sse / target_var), places=8)

    def test_apply_mod_using_scapy_payload_scale_can_shrink_payload(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether, Raw
        except ImportError:
            self.skipTest("scapy not installed")
        pkt = (
            Ether()
            / IP(src="1.1.1.1", dst="2.2.2.2")
            / TCP(sport=1234, dport=80, seq=100, ack=1, flags="PA")
            / Raw(b"abcdefghij")
        )
        pkt.time = 0.0
        out = apply_mod_using_scapy(
            [pkt],
            np.array([10.0, 1.0, 0.0, 80.0, 0.0, 1.0, 0.5], dtype=np.float32),
            apply_fields=["payload_scale"],
        )
        self.assertEqual(bytes(out[0][Raw].load), b"abcde")

    def test_apply_mod_using_scapy_payload_scale_prefers_forward_payload(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether, Raw
        except ImportError:
            self.skipTest("scapy not installed")
        pkt_fwd = (
            Ether()
            / IP(src="1.1.1.1", dst="2.2.2.2")
            / TCP(sport=1234, dport=80, seq=100, ack=1, flags="PA")
            / Raw(b"abcdefghij")
        )
        pkt_bwd = (
            Ether()
            / IP(src="2.2.2.2", dst="1.1.1.1")
            / TCP(sport=80, dport=1234, seq=200, ack=110, flags="PA")
            / Raw(b"klmnopqrst")
        )
        pkt_fwd.time = 0.0
        pkt_bwd.time = 0.1
        out = apply_mod_using_scapy(
            [pkt_fwd, pkt_bwd],
            np.array([10.0, 1.0, 0.0, 80.0, 0.0, 1.0, 0.5], dtype=np.float32),
            apply_fields=["payload_scale"],
        )
        self.assertEqual(bytes(out[0][Raw].load), b"abcde")
        self.assertEqual(bytes(out[1][Raw].load), b"klmnopqrst")

    def test_apply_mod_using_scapy_flag_ratio_targets_forward_data_packets(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether, Raw
        except ImportError:
            self.skipTest("scapy not installed")
        pkt_fwd = (
            Ether()
            / IP(src="1.1.1.1", dst="2.2.2.2")
            / TCP(sport=1234, dport=80, seq=100, ack=1, flags="A")
            / Raw(b"abcdefghij")
        )
        pkt_bwd = (
            Ether()
            / IP(src="2.2.2.2", dst="1.1.1.1")
            / TCP(sport=80, dport=1234, seq=200, ack=110, flags="A")
            / Raw(b"klmnopqrst")
        )
        pkt_fwd.time = 0.0
        pkt_bwd.time = 0.1
        out = apply_mod_using_scapy(
            [pkt_fwd, pkt_bwd],
            np.array([10.0, 1.0, 0.0, 80.0, 1.0, 1.0, 1.0], dtype=np.float32),
            apply_fields=["flag_ratio"],
        )
        self.assertTrue(int(out[0][TCP].flags) & 0x08)
        self.assertFalse(int(out[1][TCP].flags) & 0x08)

    def test_random_remap_samples_ports_from_benign_vocab(self) -> None:
        feature_names = [
            "Dst Port",
            "ACK Flag Count",
            "Flow IAT Mean",
            "Flow IAT Std",
            "Average Packet Size",
            "Flow Duration",
            "Total Packets",
        ]
        x_ben = np.array(
            [
                [80, 1, 10.0, 1.0, 100.0, 500.0, 10.0],
                [443, 0, 20.0, 2.0, 120.0, 800.0, 12.0],
                [8080, 1, 30.0, 3.0, 140.0, 900.0, 14.0],
            ],
            dtype=np.float32,
        )
        x_adv = np.array([[9999, 5, 1000.0, 10.0, 1500.0, 20000.0, 6.0]], dtype=np.float32)
        mods = build_random_remap_modifications(x_adv, x_ben, feature_names, seed=7)
        self.assertIn(int(mods[0, 3]), {80, 443, 8080})
        self.assertGreaterEqual(mods[0, 4], 0.0)
        self.assertLessEqual(mods[0, 4], 1.0)

    def test_apply_mod_using_scapy_protocol_auto_fix_clamps_ttl(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether
        except ImportError:
            self.skipTest("scapy not installed")
        pkt = Ether() / IP(src="1.1.1.1", dst="2.2.2.2", ttl=0) / TCP(sport=1234, dport=80, seq=1, ack=0, flags="S")
        pkt.time = 0.0
        out = apply_mod_using_scapy(
            [pkt],
            np.array([10.0, 1.0, 0.0, 80.0, 0.0, 1.0, 1.0], dtype=np.float32),
            apply_fields=[],
            protocol_auto_fix=True,
        )
        self.assertEqual(int(out[0][IP].ttl), 1)

    def test_apply_mod_using_scapy_flow_vocab_closest_uses_observed_ports(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether
        except ImportError:
            self.skipTest("scapy not installed")
        pkt_a = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=443, seq=1, ack=0, flags="S")
        pkt_b = Ether() / IP(src="2.2.2.2", dst="1.1.1.1") / TCP(sport=443, dport=1234, seq=2, ack=2, flags="SA")
        pkt_a.time = 0.0
        pkt_b.time = 0.1
        out = apply_mod_using_scapy(
            [pkt_a, pkt_b],
            np.array([10.0, 1.0, 0.0, 444.0, 0.0, 1.0, 1.0], dtype=np.float32),
            apply_fields=["dst_port_new"],
            dst_port_policy="flow_vocab_closest",
            dst_port_allowlist=[80],
        )
        self.assertEqual(int(out[0][TCP].dport), 443)

    def test_apply_mod_using_scapy_flow_vocab_closest_prefers_original_flow_port_when_context_beats_allowlist(
        self,
    ) -> None:
        try:
            from scapy.all import IP, TCP, Ether
        except ImportError:
            self.skipTest("scapy not installed")
        pkt_a = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=443, seq=1, ack=0, flags="S")
        pkt_b = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=443, seq=2, ack=1, flags="A")
        pkt_a.time = 0.0
        pkt_b.time = 0.1
        out = apply_mod_using_scapy(
            [pkt_a, pkt_b],
            np.array([10.0, 1.0, 0.0, 430.0, 0.0, 1.0, 1.0], dtype=np.float32),
            apply_fields=["dst_port_new"],
            dst_port_policy="flow_vocab_closest",
            dst_port_allowlist=[80, 8080],
        )
        self.assertEqual(int(out[0][TCP].dport), 443)

    def test_apply_mod_using_scapy_clamps_invalid_target_port_before_writing(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether
        except ImportError:
            self.skipTest("scapy not installed")
        pkt = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=1234, dport=443, seq=1, ack=0, flags="S")
        pkt.time = 0.0

        out = apply_mod_using_scapy(
            [pkt],
            np.array([10.0, 1.0, 0.0, 70000.0, 0.0, 1.0, 1.0], dtype=np.float32),
            apply_fields=["dst_port_new"],
            dst_port_policy="set",
            dst_port_allowlist=[],
        )

        self.assertEqual(int(out[0][TCP].dport), 443)

    def test_apply_mod_using_scapy_clamps_invalid_existing_transport_ports(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether
        except ImportError:
            self.skipTest("scapy not installed")
        pkt = Ether() / IP(src="1.1.1.1", dst="2.2.2.2") / TCP(sport=70000, dport=80000, seq=1, ack=0, flags="S")
        pkt.time = 0.0

        out = apply_mod_using_scapy(
            [pkt],
            np.array([10.0, 1.0, 0.0, 70000.0, 0.0, 1.0, 1.0], dtype=np.float32),
            apply_fields=[],
            protocol_auto_fix=True,
        )

        self.assertEqual(int(out[0][TCP].sport), 80)
        self.assertEqual(int(out[0][TCP].dport), 80)

    def test_apply_mod_using_scapy_caps_oversized_payload_before_checksum_rebuild(self) -> None:
        try:
            from scapy.all import IP, TCP, Ether, Raw, raw
        except ImportError:
            self.skipTest("scapy not installed")
        pkt = (
            Ether()
            / IP(src="1.1.1.1", dst="2.2.2.2")
            / TCP(sport=1234, dport=80, seq=1, ack=0, flags="PA")
            / Raw(bytes([1]) * 40000)
        )
        pkt.time = 0.0

        out = apply_mod_using_scapy(
            [pkt],
            np.array([10.0, 1.0, 0.0, 80.0, 0.0, 1.0, 2.0], dtype=np.float32),
            apply_fields=["payload_scale"],
            protocol_auto_fix=True,
        )

        self.assertLessEqual(len(bytes(out[0][Raw].load)), 60000)
        self.assertLessEqual(len(raw(out[0][IP])), 65535)


if __name__ == "__main__":
    unittest.main()
