from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rdsynth.utils.feature_preprocessing import apply_feature_transform, invert_feature_transform


@dataclass(frozen=True)
class DatasetPreprocessor:
    scaler: Any
    power_transformer: Any | None
    feature_names: list[str]
    impute_strategy: str = "zero"
    impute_values: np.ndarray | None = None
    winsorize_lower: np.ndarray | None = None
    winsorize_upper: np.ndarray | None = None
    log1p_mask: np.ndarray | None = None
    log1p_shift: np.ndarray | None = None
    feature_transform_plan: Any | None = None

    @classmethod
    def from_bundle(cls, bundle: Any) -> "DatasetPreprocessor":
        return cls(
            scaler=getattr(bundle, "scaler", None),
            power_transformer=getattr(bundle, "power_transformer", None),
            feature_names=list(getattr(bundle, "feature_names", [])),
            impute_strategy=str(getattr(bundle, "impute_strategy", "zero")),
            impute_values=getattr(bundle, "impute_values", None),
            winsorize_lower=getattr(bundle, "winsorize_lower", None),
            winsorize_upper=getattr(bundle, "winsorize_upper", None),
            log1p_mask=getattr(bundle, "log1p_mask", None),
            log1p_shift=getattr(bundle, "log1p_shift", None),
            feature_transform_plan=getattr(bundle, "feature_transform_plan", None),
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        array, squeeze = self._coerce_2d(values)
        array = self._apply_input_transforms(array)
        if self.feature_transform_plan is not None:
            array = apply_feature_transform(array, self.feature_transform_plan)
        elif self.power_transformer is not None:
            array = self.power_transformer.transform(array)
        if self.feature_transform_plan is None and self.scaler is not None:
            array = self.scaler.transform(array)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        return array[0] if squeeze else array

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        array, squeeze = self._coerce_2d(values)
        array = np.asarray(array, dtype=np.float64)
        if self.feature_transform_plan is not None:
            array = invert_feature_transform(array, self.feature_transform_plan)
        elif self.scaler is not None:
            array = self.scaler.inverse_transform(array)
        if self.feature_transform_plan is None and self.power_transformer is not None:
            array = self._safe_power_inverse(array)
        if self.log1p_mask is not None and self.log1p_shift is not None:
            mask = np.asarray(self.log1p_mask, dtype=bool)
            if np.any(mask):
                shift = np.asarray(self.log1p_shift, dtype=np.float64)
                array[:, mask] = np.expm1(array[:, mask]) - shift[mask]
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64, copy=False)
        return array[0] if squeeze else array

    def feature_mean(self, values: np.ndarray) -> np.ndarray:
        array, _ = self._coerce_2d(values)
        if array.shape[0] == 0:
            raise ValueError("Cannot compute feature mean for an empty array.")
        return np.mean(self.inverse_transform(array), axis=0)

    def _coerce_2d(self, values: np.ndarray) -> tuple[np.ndarray, bool]:
        array = np.asarray(values, dtype=np.float64)
        squeeze = False
        if array.ndim == 1:
            array = array.reshape(1, -1)
            squeeze = True
        if array.ndim != 2:
            raise ValueError(f"Expected a 1D or 2D array, got shape {array.shape}.")
        expected_dim = len(self.feature_names)
        if expected_dim and array.shape[1] != expected_dim:
            raise ValueError(f"Expected {expected_dim} features, got {array.shape[1]}.")
        return array, squeeze

    def _apply_input_transforms(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).copy()
        array = np.nan_to_num(array, nan=np.nan, posinf=np.nan, neginf=np.nan)
        if self.log1p_mask is not None and self.log1p_shift is not None:
            mask = np.asarray(self.log1p_mask, dtype=bool)
            if np.any(mask):
                shift = np.asarray(self.log1p_shift, dtype=np.float64)
                array[:, mask] = np.log1p(np.clip(array[:, mask] + shift[mask], a_min=0.0, a_max=None))
        if self.winsorize_lower is not None and self.winsorize_upper is not None:
            lower = np.asarray(self.winsorize_lower, dtype=np.float64)
            upper = np.asarray(self.winsorize_upper, dtype=np.float64)
            array = np.clip(array, lower, upper)
        if self.impute_strategy == "median" and self.impute_values is not None:
            fill = np.asarray(self.impute_values, dtype=np.float64)
            array = np.where(np.isnan(array), fill, array)
        return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)

    def _safe_power_inverse(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64).copy()
        method = getattr(self.power_transformer, "method", "yeo-johnson")
        if method == "yeo-johnson":
            lambdas = getattr(self.power_transformer, "lambdas_", None)
            if lambdas is not None:
                for index, lam in enumerate(lambdas):
                    if lam < 0:
                        upper = (-1.0 / lam) - 1.0e-6
                        array[:, index] = np.minimum(array[:, index], upper)
                    if lam > 2:
                        lower = (1.0 / (2.0 - lam)) + 1.0e-6
                        array[:, index] = np.maximum(array[:, index], lower)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            return self.power_transformer.inverse_transform(array)
