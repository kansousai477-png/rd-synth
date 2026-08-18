"""PCAP feature calibration to bridge nfstream/scapy → dataset feature distributions.

When Stage 3 runs in ``pcap_conditioned`` mode, PCAP-extracted flow features are fed
into the Stage 2 diffusion model as conditioning. The diffusion model was trained on
dataset features extracted by CICFlowMeter, but PCAP features are extracted by
nfstream/scapy which produce different scales (e.g. ms vs μs for durations).

Linear calibration maps PCAP-extracted raw features into the datasetʼs raw-feature
space so the preprocessor and diffusion model both see in-distribution values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PcapFeatureCalibration:
    """Per-feature linear calibration: dataset_raw ≈ scale * pcap_raw + offset."""

    scale: np.ndarray
    offset: np.ndarray
    pcap_mean: np.ndarray
    pcap_std: np.ndarray
    dataset_mean: np.ndarray
    dataset_std: np.ndarray
    feature_count: int
    is_active: bool = True
    pcap_sample_count: int = 0
    _eps: float = field(default=1.0e-8)

    def calibrate(self, x: np.ndarray) -> np.ndarray:
        """Map PCAP raw features → dataset raw space."""
        if not self.is_active or self.scale.size == 0:
            return np.asarray(x, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return x * self.scale.reshape(1, -1) + self.offset.reshape(1, -1)

    def uncalibrate(self, x: np.ndarray) -> np.ndarray:
        """Map dataset raw space → PCAP raw features (inverse of calibrate)."""
        if not self.is_active or self.scale.size == 0:
            return np.asarray(x, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        inv_scale = 1.0 / np.maximum(self.scale.reshape(1, -1), self._eps)
        return (x - self.offset.reshape(1, -1)) * inv_scale

    def to_dict(self) -> dict[str, object]:
        return {
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
            "pcap_mean": self.pcap_mean.tolist(),
            "pcap_std": self.pcap_std.tolist(),
            "dataset_mean": self.dataset_mean.tolist(),
            "dataset_std": self.dataset_std.tolist(),
            "feature_count": self.feature_count,
            "is_active": self.is_active,
            "pcap_sample_count": self.pcap_sample_count,
        }


_INACTIVE = PcapFeatureCalibration(
    scale=np.zeros((0,), dtype=np.float64),
    offset=np.zeros((0,), dtype=np.float64),
    pcap_mean=np.zeros((0,), dtype=np.float64),
    pcap_std=np.zeros((0,), dtype=np.float64),
    dataset_mean=np.zeros((0,), dtype=np.float64),
    dataset_std=np.zeros((0,), dtype=np.float64),
    feature_count=0,
    is_active=False,
    pcap_sample_count=0,
)


def compute_pcap_calibration(
    *,
    dataset_benign_raw: np.ndarray,
    pcap_benign_raw: np.ndarray,
    min_scale: float = 0.05,
    max_scale: float = 200.0,
    min_samples: int = 10,
    max_shift_std: float = 5.0,
) -> PcapFeatureCalibration:
    """Compute per-feature linear calibration from a reference benign PCAP pool.

    Parameters
    ----------
    dataset_benign_raw:
        Benign features from the dataset in *raw* (inverse-transformed) space,
        shape (N_dataset, D).
    pcap_benign_raw:
        Benign features extracted from reference PCAPs via nfstream/scapy,
        already aligned to the same feature order as *dataset_benign_raw*,
        shape (N_pcap, D).
    min_scale / max_scale:
        Per-feature scale is clamped to [min_scale, max_scale].
    min_samples:
        Require at least this many PCAP samples; otherwise return inactive calibration.
    max_shift_std:
        If |dataset_mean - pcap_mean| / max(dataset_std, eps) > max_shift_std
        the calibration for that feature is skipped (scale=1, offset=0).

    Returns
    -------
    PcapFeatureCalibration (``is_active=False`` when calibration is skipped).
    """
    dataset_benign_raw = np.asarray(dataset_benign_raw, dtype=np.float64)
    pcap_benign_raw = np.asarray(pcap_benign_raw, dtype=np.float64)

    if pcap_benign_raw.shape[0] < min_samples:
        return _INACTIVE

    feature_count = dataset_benign_raw.shape[1]
    if pcap_benign_raw.shape[1] != feature_count:
        raise ValueError(f"Feature count mismatch: dataset={feature_count}, pcap={pcap_benign_raw.shape[1]}")

    pcap_mean = np.mean(pcap_benign_raw, axis=0)
    pcap_std = np.std(pcap_benign_raw, axis=0)
    dataset_mean = np.mean(dataset_benign_raw, axis=0)
    dataset_std = np.std(dataset_benign_raw, axis=0)

    eps = 1.0e-8
    scale = np.ones(feature_count, dtype=np.float64)
    offset = np.zeros(feature_count, dtype=np.float64)

    for i in range(feature_count):
        d_std = max(float(dataset_std[i]), eps)
        p_std = max(float(pcap_std[i]), eps)
        d_mean = float(dataset_mean[i])
        p_mean = float(pcap_mean[i])

        shift_std = abs(d_mean - p_mean) / d_std
        if shift_std > max_shift_std:
            # Extreme shift — keep identity; let the preprocessor handle it.
            # The diffusion model may still struggle, but clamping blindly is worse.
            continue

        raw_scale = d_std / p_std
        raw_scale = max(min_scale, min(max_scale, raw_scale))
        scale[i] = float(raw_scale)
        offset[i] = float(d_mean - raw_scale * p_mean)

    return PcapFeatureCalibration(
        scale=scale,
        offset=offset,
        pcap_mean=pcap_mean,
        pcap_std=pcap_std,
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
        feature_count=feature_count,
        is_active=True,
        pcap_sample_count=int(pcap_benign_raw.shape[0]),
    )


class CalibratedPreprocessor:
    """Wraps a DatasetPreprocessor so PCAP features are calibration-transparent.

    ``transform`` calibrates PCAP raw → dataset raw then applies the base
    preprocessor. ``inverse_transform`` applies the base inverse then uncalibrates
    → PCAP raw space. When calibration is inactive this is a transparent passthrough.
    """

    def __init__(self, base: Any, calibration: PcapFeatureCalibration):
        self._base = base
        self._cal = calibration

    @property
    def is_active(self) -> bool:
        return self._cal.is_active

    def transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_active:
            return self._base.transform(x)
        calibrated = self._cal.calibrate(x)
        return self._base.transform(calibrated)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if not self.is_active:
            return self._base.inverse_transform(x)
        calibrated_raw = self._base.inverse_transform(x)
        return self._cal.uncalibrate(calibrated_raw)

    def feature_mean(self, x: np.ndarray) -> np.ndarray:
        mean = self._base.feature_mean(x)
        if self.is_active:
            return self._cal.calibrate(mean.reshape(1, -1)).ravel()
        return mean
