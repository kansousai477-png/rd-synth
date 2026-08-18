from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.append(str(SRC))

from rdsynth.utils.traffic_schema import apply_schema_projection, infer_traffic_feature_schema


class TrafficSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.feature_names = [
            "Dst Port",
            "ACK Flag Count",
            "Flow IAT Mean",
            "Packet Length Mean",
            "Down/Up Ratio",
        ]
        self.x_ben = np.array(
            [
                [80, 1, 10.0, 100.0, 0.3],
                [443, 0, 20.0, 120.0, 0.6],
            ],
            dtype=np.float64,
        )
        self.x_mal = np.array(
            [
                [8080, 1, 200.0, 500.0, 2.0],
                [9000, 0, 150.0, 600.0, 1.5],
            ],
            dtype=np.float64,
        )

    def test_schema_infers_key_network_groups(self) -> None:
        schema = infer_traffic_feature_schema(self.feature_names, self.x_ben)
        self.assertEqual(schema.port_idx.tolist(), [0])
        self.assertEqual(schema.flag_idx.tolist(), [1])
        self.assertEqual(schema.temporal_idx.tolist(), [2])
        self.assertEqual(schema.ratio_idx.tolist(), [4])

    def test_schema_projection_enforces_network_constraints(self) -> None:
        schema = infer_traffic_feature_schema(self.feature_names, self.x_ben)
        x_adv = np.array([[12345, 4.7, -5.0, -3.2, 4.0]], dtype=np.float64)
        projected = apply_schema_projection(
            x_adv=x_adv,
            x_mal=self.x_mal[:1],
            x_ben=self.x_ben,
            schema=schema,
            port_policy="keep",
            flag_policy="clip",
            temporal_policy="clip_benign",
        )
        self.assertEqual(projected[0, 0], self.x_mal[0, 0])
        self.assertGreaterEqual(projected[0, 2], self.x_ben[:, 2].min())
        self.assertLessEqual(projected[0, 2], self.x_ben[:, 2].max())
        self.assertGreaterEqual(projected[0, 4], 0.0)
        self.assertLessEqual(projected[0, 4], 1.0)


if __name__ == "__main__":
    unittest.main()
