from __future__ import annotations

import numpy as np

from rdsynth.pipeline.stage3_calibration import (
    CalibratedPreprocessor,
    PcapFeatureCalibration,
    compute_pcap_calibration,
)


class TestComputePcapCalibration:
    def test_inactive_when_too_few_samples(self):
        cal = compute_pcap_calibration(
            dataset_benign_raw=np.ones((5, 3)),
            pcap_benign_raw=np.ones((5, 3)),
            min_samples=10,
        )
        assert not cal.is_active
        assert cal.pcap_sample_count == 0

    def test_inactive_when_empty(self):
        cal = compute_pcap_calibration(
            dataset_benign_raw=np.zeros((0, 3)),
            pcap_benign_raw=np.zeros((0, 3)),
            min_samples=1,
        )
        assert not cal.is_active

    def test_identity_for_matching_distributions(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 3.0, (500, 4))
        cal = compute_pcap_calibration(
            dataset_benign_raw=data[:250],
            pcap_benign_raw=data[250:],
            min_samples=10,
        )
        assert cal.is_active
        assert cal.feature_count == 4
        for i in range(4):
            assert np.isclose(cal.scale[i], 1.0, atol=0.3)
            assert np.isclose(cal.offset[i], 0.0, atol=1.5)

    def test_scale_correction_for_unit_difference(self):
        rng = np.random.default_rng(42)
        dataset = rng.normal(loc=1000000.0, scale=500000.0, size=(200, 2))  # μs
        pcap = dataset[:100] / 1000.0  # ms (1000x smaller)
        cal = compute_pcap_calibration(
            dataset_benign_raw=dataset[:100],
            pcap_benign_raw=pcap,
            min_samples=10,
            max_scale=2000.0,
        )
        assert cal.is_active
        # Scale should be ~1000
        for i in range(2):
            assert 500 < cal.scale[i] < 2000, f"feature {i}: scale={cal.scale[i]:.1f}"

    def test_roundtrip(self):
        rng = np.random.default_rng(42)
        dataset = rng.normal(1000.0, 100.0, (200, 3))
        pcap = dataset[:100] / 5.0  # 5x smaller — within max_scale
        cal = compute_pcap_calibration(
            dataset_benign_raw=dataset[:100],
            pcap_benign_raw=pcap,
            min_samples=10,
        )
        assert cal.is_active
        x_pcap = pcap[:10]
        x_cal = cal.calibrate(x_pcap)
        x_back = cal.uncalibrate(x_cal)
        assert np.allclose(x_back, x_pcap, atol=1e-6)

    def test_very_different_distribution_uses_identity(self):
        rng = np.random.default_rng(42)
        dataset = rng.normal(0.0, 1.0, (100, 3))
        # PCAP features 20 stds away — extreme shift
        pcap = rng.normal(25.0, 0.5, (50, 3))
        cal = compute_pcap_calibration(
            dataset_benign_raw=dataset,
            pcap_benign_raw=pcap,
            min_samples=10,
            max_shift_std=5.0,
        )
        assert cal.is_active
        # Features with extreme shifts should use identity (scale=1, offset=0)
        for i in range(3):
            assert cal.scale[i] == 1.0
            assert cal.offset[i] == 0.0

    def test_clamps_scale_to_bounds(self):
        rng = np.random.default_rng(42)
        dataset = rng.normal(10.0, 3.0, (100, 1))
        # 50x smaller std → scale should be clamped to max_scale
        pcap = rng.normal(10.0, 0.03, (50, 1))
        cal = compute_pcap_calibration(
            dataset_benign_raw=dataset,
            pcap_benign_raw=pcap,
            min_samples=10,
            min_scale=0.1,
            max_scale=20.0,
        )
        assert cal.is_active
        assert cal.scale[0] <= 20.0


class TestCalibratedPreprocessor:
    def test_passthrough_when_inactive(self):
        cal = PcapFeatureCalibration(
            scale=np.array([], dtype=np.float64),
            offset=np.array([], dtype=np.float64),
            pcap_mean=np.array([], dtype=np.float64),
            pcap_std=np.array([], dtype=np.float64),
            dataset_mean=np.array([], dtype=np.float64),
            dataset_std=np.array([], dtype=np.float64),
            feature_count=0,
            is_active=False,
        )

        # A minimal mock preprocessor
        class MockPP:
            def transform(self, x):
                return x + 1.0

            def inverse_transform(self, x):
                return x - 1.0

        cpp = CalibratedPreprocessor(MockPP(), cal)
        x = np.random.randn(10, 3)
        assert np.allclose(cpp.transform(x), x + 1.0)
        assert np.allclose(cpp.inverse_transform(x + 1.0), x)

    def test_roundtrip_when_active(self):
        rng = np.random.default_rng(42)
        dataset = rng.normal(100.0, 20.0, (100, 3))
        pcap = dataset[:50] / 10.0  # 10x smaller

        cal = compute_pcap_calibration(
            dataset_benign_raw=dataset[:50],
            pcap_benign_raw=pcap,
            min_samples=10,
        )
        assert cal.is_active

        class MockPP:
            """Simple StandardScaler-like transform."""

            def __init__(self, mean, std):
                self._mean = mean
                self._std = std

            def transform(self, x):
                return (np.asarray(x) - self._mean) / (self._std + 1e-8)

            def inverse_transform(self, x):
                return np.asarray(x) * (self._std + 1e-8) + self._mean

        # Base preprocessor uses dataset statistics
        base = MockPP(dataset[:50].mean(axis=0), dataset[:50].std(axis=0))
        cpp = CalibratedPreprocessor(base, cal)

        # Test forward: PCAP raw → calibrated preprocessed
        x_pcap = rng.normal(10.0, 2.0, (5, 3))  # values in PCAP scale
        x_pre = cpp.transform(x_pcap)

        # Test backward: preprocessed → PCAP raw
        x_roundtrip = cpp.inverse_transform(x_pre)
        assert np.allclose(x_roundtrip, x_pcap, atol=1e-4)

    def test_calibrated_values_are_dataset_scaled(self):
        """Calibration maps PCAP features closer to the dataset's normalized space."""
        rng = np.random.default_rng(42)
        # Calibration reference: PCAP features from a different distribution
        pcap_ref = rng.normal(50.0, 10.0, (200, 2))
        dataset_ref = rng.normal(1000.0, 200.0, (200, 2))

        cal = compute_pcap_calibration(
            dataset_benign_raw=dataset_ref,
            pcap_benign_raw=pcap_ref,
            min_samples=10,
        )
        assert cal.is_active

        class MockPP:
            def __init__(self):
                self._mean = dataset_ref.mean(axis=0)
                self._std = dataset_ref.std(axis=0)

            def transform(self, x):
                return (np.asarray(x) - self._mean) / (self._std + 1e-8)

            def inverse_transform(self, x):
                return np.asarray(x) * (self._std + 1e-8) + self._mean

        cpp = CalibratedPreprocessor(MockPP(), cal)
        base_pp = MockPP()

        # Test on a holdout set from the same pcap distribution
        x_pcap = rng.normal(50.0, 10.0, (30, 2))
        x_pre_cal = cpp.transform(x_pcap)
        x_pre_raw = base_pp.transform(x_pcap)

        # Calibrated values should be closer to standard-normal (mean ≈ 0)
        cal_mean_abs = float(np.mean(np.abs(x_pre_cal)))
        raw_mean_abs = float(np.mean(np.abs(x_pre_raw)))
        assert cal_mean_abs < raw_mean_abs, (
            f"calibrated |mean|={cal_mean_abs:.3f} should be < raw |mean|={raw_mean_abs:.3f}"
        )


class TestPcapFeatureCalibrationDict:
    def test_to_dict_inactive(self):
        cal = compute_pcap_calibration(
            dataset_benign_raw=np.zeros((0, 3)),
            pcap_benign_raw=np.zeros((0, 3)),
            min_samples=1,
        )
        d = cal.to_dict()
        assert d["is_active"] is False

    def test_to_dict_active(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 3.0, (100, 3))
        cal = compute_pcap_calibration(
            dataset_benign_raw=data[:50],
            pcap_benign_raw=data[50:],
            min_samples=10,
        )
        d = cal.to_dict()
        assert d["is_active"] is True
        assert d["feature_count"] == 3
        assert len(d["scale"]) == 3
        assert d["pcap_sample_count"] == 50
