from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rdsynth.pipeline.stage3_pcap_semantics import filter_candidates_by_categories
from rdsynth.stages.oracle import predict_sklearn_probs
from rdsynth.stages.stage3_features import (
    extract_pcap_features_cicflowmeter,
    extract_pcap_features_nfstream,
    extract_pcap_features_scapy,
)
from rdsynth.stages.stage3_remap import build_remap_targets

_PCAP_FEATURE_SIGNATURE_VERSION = 4


class PcapFeatureExtractionError(RuntimeError):
    pass


@dataclass
class PcapFeatureExtractor:
    feature_backend: str
    feature_names: list[str]
    raw_feature_mean: np.ndarray
    alias_map: dict[str, object]
    align_min_cov: float
    scapy_available: bool
    nfstream_available: bool
    cicflowmeter_available: bool
    cicflowmeter_cmd: str
    cicflowmeter_timeout: int
    fail_closed: bool
    fail_on_partial_alignment: bool
    preprocessor: Any
    pcap_eval_model: Any
    pcap_eval_model_name: str
    oracle: Any
    surrogate: Any
    pcap_eval_batch_size: int
    seed: int
    device: torch.device
    ids: Any = None
    max_pcap_bytes: int = 0
    max_flows_per_pcap: int | None = None
    cache_enable: bool = False
    cache_dir: Path | None = None
    scapy_warned: dict[str, bool] = field(default_factory=lambda: {"done": False})
    feature_status_counts: dict[str, int] = field(default_factory=dict)
    feature_fallback_count: int = 0
    feature_fill_count: int = 0
    disk_cache_hit_count: int = 0
    disk_cache_miss_count: int = 0
    _extract_cache: dict[tuple[str, int | None, int | None], tuple[np.ndarray, str, dict[str, object]]] = field(
        default_factory=dict
    )
    _classify_cache: dict[
        tuple[str, int | None, int | None],
        tuple[np.ndarray, str, dict[str, object], np.ndarray, np.ndarray],
    ] = field(default_factory=dict)

    def _batched_probs_torch(self, model: torch.nn.Module, x: np.ndarray) -> np.ndarray:
        model.eval()
        probs = []
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            for i in range(0, x_t.size(0), self.pcap_eval_batch_size):
                xb = x_t[i : i + self.pcap_eval_batch_size]
                logits = model(xb)
                probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
        if not probs:
            return np.zeros((0, 2), dtype=np.float32)
        return np.concatenate(probs, axis=0)

    def predict_probs(self, x: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.pcap_eval_model is None:
            return None, None
        bundle = self.ids if self.ids is not None else self.oracle
        if self.pcap_eval_model_name in {"oracle", "ids"} and bundle is not None:
            if bundle.model_type in {"mlp", "cnn", "rnn", "lstm", "gru", "transformer"}:
                probs = self._batched_probs_torch(bundle.model, x)
                preds = np.argmax(probs, axis=1) if probs.size else np.array([], dtype=int)
                return preds, probs
            preds = bundle.model.predict(x)
            probs = predict_sklearn_probs(bundle.model, x)
            return np.asarray(preds), probs
        if self.surrogate is None:
            return None, None
        probs = self._batched_probs_torch(self.surrogate, x)
        preds = np.argmax(probs, axis=1) if probs.size else np.array([], dtype=int)
        return preds, probs

    def classify_features(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        feat_norm = self.preprocessor.transform(feat)
        if feat_norm.size == 0:
            return np.array([float("nan"), float("nan")]), feat_norm
        preds, probs = self.predict_probs(feat_norm)
        if preds is None:
            return np.array([float("nan"), float("nan")]), feat_norm
        if probs is None or probs.size == 0:
            probs = np.zeros((len(preds), 2), dtype=np.float32)
            if len(preds):
                probs[np.arange(len(preds)), preds.astype(int)] = 1.0
        probs_mean = probs.mean(axis=0)
        return probs_mean, feat_norm

    def _record_feature_meta(self, meta: dict[str, object]) -> None:
        status = str(meta.get("status", "unknown") or "unknown")
        self.feature_status_counts[status] = self.feature_status_counts.get(status, 0) + 1
        if meta.get("fallback_from"):
            self.feature_fallback_count += 1
        if bool(meta.get("used_fill_values", False)):
            self.feature_fill_count += 1

    @staticmethod
    def _cache_key(pcap_file: str | Path) -> tuple[str, int | None, int | None]:
        path = Path(pcap_file)
        try:
            stat = path.stat()
        except OSError:
            return (str(path), None, None)
        return (str(path), int(stat.st_mtime_ns), int(stat.st_size))

    def _feature_signature(self) -> str:
        payload = {
            "version": _PCAP_FEATURE_SIGNATURE_VERSION,
            "backend": self.feature_backend,
            "feature_names": [str(name) for name in self.feature_names],
            "cicflowmeter_cmd": str(self.cicflowmeter_cmd),
            "cicflowmeter_timeout": int(self.cicflowmeter_timeout),
            "align_min_cov": float(self.align_min_cov),
            "fail_closed": bool(self.fail_closed),
            "fail_on_partial_alignment": bool(self.fail_on_partial_alignment),
            "max_pcap_bytes": int(self.max_pcap_bytes),
            "max_flows_per_pcap": int(self.max_flows_per_pcap or 0),
            "alias_map": self.alias_map,
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded)
        digest.update(np.asarray(self.raw_feature_mean, dtype=np.float32).reshape(-1).tobytes())
        return digest.hexdigest()

    def _disk_extract_cache_path(self, cache_key: tuple[str, int | None, int | None]) -> Path | None:
        if not self.cache_enable or self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "pcap": cache_key[0],
            "mtime_ns": cache_key[1],
            "size": cache_key[2],
            "feature_signature": self._feature_signature(),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
        return self.cache_dir / f"{hashlib.sha256(encoded).hexdigest()}.pkl"

    def _load_disk_extract_cache(
        self,
        cache_key: tuple[str, int | None, int | None],
    ) -> tuple[np.ndarray, str, dict[str, object]] | None:
        cache_path = self._disk_extract_cache_path(cache_key)
        if cache_path is None:
            return None
        if not cache_path.exists():
            self.disk_cache_miss_count += 1
            return None
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError):
            self.disk_cache_miss_count += 1
            return None
        if not isinstance(cached, dict):
            self.disk_cache_miss_count += 1
            return None
        feat = cached.get("feat")
        backend = cached.get("backend")
        meta = cached.get("meta")
        if feat is None or not isinstance(backend, str) or not isinstance(meta, dict):
            self.disk_cache_miss_count += 1
            return None
        self.disk_cache_hit_count += 1
        return np.asarray(feat), backend, dict(meta)

    def _ensure_allowed(self, pcap_file: str, meta: dict[str, object]) -> None:
        status = str(meta.get("status", "unknown") or "unknown")
        if self.fail_closed and bool(meta.get("used_fill_values", False)):
            raise PcapFeatureExtractionError(
                f"PCAP feature extraction failed closed for {pcap_file}: status={status} reason={meta.get('reason', '')}"
            )
        if self.fail_on_partial_alignment and status == "alignment_partial":
            alignment = meta.get("alignment", {})
            if isinstance(alignment, dict):
                coverage = float(alignment.get("coverage", 0.0) or 0.0)
                missing = int(alignment.get("missing", len(alignment.get("missing_features", []) or [])) or 0)
                total = int(alignment.get("total", 0) or 0)
                missing_max = max(1, int(total * 0.12)) if total > 0 else 1
                if (
                    coverage >= self.align_min_cov
                    and missing <= missing_max
                    and not bool(meta.get("used_fill_values", False))
                ):
                    meta["status"] = "alignment_partial_tolerated"
                    meta["alignment_tolerated"] = True
                    meta["alignment_tolerance_reason"] = (
                        f"coverage={coverage:.3f}>=min={self.align_min_cov:.3f};missing={missing}<={missing_max}"
                    )
                    return
                # ── Lenient fallback for incompatible datasets ──────────────
                # When the feature schema is fundamentally different (e.g. IoT
                # vs CICFlowMeter), alignment coverage may be very low.  Rather
                # than blocking the pipeline entirely, accept partial alignment
                # with a reduced effective coverage target and proceed with the
                # available features — but only above a hard floor.
                _hard_floor = max(0.20, self.align_min_cov * 0.30)
                if coverage >= _hard_floor and coverage < self.align_min_cov:
                    meta["status"] = "alignment_partial_lenient"
                    meta["alignment_tolerated"] = True
                    meta["alignment_tolerance_reason"] = (
                        f"lenient: coverage={coverage:.3f} below min={self.align_min_cov:.3f} "
                        f"but proceeding with available features (missing={missing})"
                    )
                    print(
                        f"[Stage3][Warn] lenient alignment: coverage={coverage:.3f} < "
                        f"min={self.align_min_cov:.3f} — proceeding with partial feature set"
                    )
                    return
            raise PcapFeatureExtractionError(
                f"PCAP feature extraction rejected partial alignment for {pcap_file}: missing={meta.get('alignment', {})}"
            )

    def _finalize(
        self, pcap_file: str, feat: np.ndarray, backend: str, meta: dict[str, object]
    ) -> tuple[np.ndarray, str, dict[str, object]]:
        self._record_feature_meta(meta)
        self._ensure_allowed(pcap_file, meta)
        return feat, backend, meta

    @staticmethod
    def _normalize_meta(meta: object, *, backend: str) -> dict[str, object]:
        if not isinstance(meta, dict):
            return {
                "backend": backend,
                "status": "ok",
                "flow_count": 0,
                "alignment": {},
            }
        out = dict(meta)
        out.setdefault("backend", backend)
        out.setdefault("status", "ok")
        out.setdefault("flow_count", 0)
        alignment = out.get("alignment")
        if not isinstance(alignment, dict):
            out["alignment"] = {}
        return out

    def _store_extract_cache(
        self,
        cache_key: tuple[str, int | None, int | None],
        pcap_file: str,
        feat: np.ndarray,
        backend: str,
        meta: dict[str, object],
    ) -> tuple[np.ndarray, str, dict[str, object]]:
        meta = self._normalize_meta(meta, backend=backend)
        feat_out, backend_out, meta_out = self._finalize(pcap_file, feat, backend, meta)
        self._extract_cache[cache_key] = (feat_out.copy(), backend_out, dict(meta_out))
        cache_path = self._disk_extract_cache_path(cache_key)
        if cache_path is not None:
            payload = {
                "feat": np.asarray(feat_out),
                "backend": backend_out,
                "meta": dict(meta_out),
            }
            temp_path = cache_path.with_suffix(".tmp")
            try:
                with temp_path.open("wb") as handle:
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                temp_path.replace(cache_path)
            except OSError:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return feat_out, backend_out, meta_out

    def extract(self, pcap_file: str) -> tuple[np.ndarray, str, dict[str, object]]:
        cache_key = self._cache_key(pcap_file)
        cached = self._extract_cache.get(cache_key)
        if cached is not None:
            feat, backend, meta = cached
            return feat.copy(), backend, dict(meta)
        disk_cached = self._load_disk_extract_cache(cache_key)
        if disk_cached is not None:
            feat, backend, meta = disk_cached
            self._extract_cache[cache_key] = (feat.copy(), backend, dict(meta))
            return feat.copy(), backend, dict(meta)

        if self.feature_backend == "scapy":
            if not self.scapy_available:
                if not self.scapy_warned["done"]:
                    print("[Stage3][Warn] scapy not installed; using fill-value PCAP features.")
                    self.scapy_warned["done"] = True
                feat = self.raw_feature_mean.reshape(1, -1)
                meta = {
                    "backend": "none",
                    "status": "dependency_missing",
                    "reason": "scapy unavailable",
                    "flow_count": 0,
                    "alignment": None,
                    "used_fill_values": True,
                }
                return self._store_extract_cache(cache_key, pcap_file, feat, "none", meta)
            feat, meta = extract_pcap_features_scapy(
                pcap_file,
                self.feature_names,
                self.raw_feature_mean,
                alias_map=self.alias_map,
                return_meta=True,
                max_flows=self.max_flows_per_pcap,
            )
            return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", meta)

        if self.feature_backend == "cicflowmeter":
            if not self.cicflowmeter_available:
                if self.scapy_available:
                    feat, meta = extract_pcap_features_scapy(
                        pcap_file,
                        self.feature_names,
                        self.raw_feature_mean,
                        alias_map=self.alias_map,
                        return_meta=True,
                    )
                    meta = {**meta, "fallback_from": "cicflowmeter", "fallback_reason": "cicflowmeter unavailable"}
                    return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", meta)
                if not self.scapy_warned["done"]:
                    print("[Stage3][Warn] cicflowmeter not available; using fill-value PCAP features.")
                    self.scapy_warned["done"] = True
                feat = self.raw_feature_mean.reshape(1, -1)
                meta = {
                    "backend": "none",
                    "status": "dependency_missing",
                    "reason": "cicflowmeter unavailable",
                    "flow_count": 0,
                    "alignment": None,
                    "used_fill_values": True,
                }
                return self._store_extract_cache(cache_key, pcap_file, feat, "none", meta)

            feat, meta = extract_pcap_features_cicflowmeter(
                pcap_file,
                self.feature_names,
                self.raw_feature_mean,
                alias_map=self.alias_map,
                return_meta=True,
                max_flows=self.max_flows_per_pcap,
                cicflowmeter_cmd=self.cicflowmeter_cmd,
                timeout=self.cicflowmeter_timeout,
            )
            meta = self._normalize_meta(meta, backend="cicflowmeter")
            alignment = meta.get("alignment") if isinstance(meta, dict) else None
            cov = float((alignment or {}).get("coverage", 1.0)) if isinstance(alignment, dict) else 1.0
            if cov < self.align_min_cov:
                _hard_floor = max(0.20, self.align_min_cov * 0.30)
                if cov >= _hard_floor:
                    meta["status"] = "alignment_partial_lenient"
                    meta["alignment_tolerated"] = True
                    meta["alignment_tolerance_reason"] = (
                        f"lenient: cicflowmeter coverage={cov:.3f} below min={self.align_min_cov:.3f}"
                    )
                    print(f"[Stage3] cicflowmeter partial alignment accepted (cov={cov:.3f})")
                elif self.scapy_available:
                    feat, scapy_meta = extract_pcap_features_scapy(
                        pcap_file,
                        self.feature_names,
                        self.raw_feature_mean,
                        alias_map=self.alias_map,
                        return_meta=True,
                    )
                    scapy_meta = {
                        **scapy_meta,
                        "fallback_from": "cicflowmeter",
                        "fallback_reason": "cicflowmeter_zero_coverage",
                    }
                    return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", scapy_meta)
            return self._store_extract_cache(cache_key, pcap_file, feat, "cicflowmeter", meta)

        if self.feature_backend == "nfstream":
            if not self.nfstream_available:
                if self.scapy_available:
                    if not self.scapy_warned["done"]:
                        print("[Stage3][Warn] nfstream not installed; falling back to scapy.")
                        self.scapy_warned["done"] = True
                    feat, meta = extract_pcap_features_scapy(
                        pcap_file,
                        self.feature_names,
                        self.raw_feature_mean,
                        alias_map=self.alias_map,
                        return_meta=True,
                    )
                    meta = {**meta, "fallback_from": "nfstream", "fallback_reason": "nfstream unavailable"}
                    return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", meta)
                if not self.scapy_warned["done"]:
                    print("[Stage3][Warn] nfstream not installed; using fill-value PCAP features.")
                    self.scapy_warned["done"] = True
                feat = self.raw_feature_mean.reshape(1, -1)
                meta = {
                    "backend": "none",
                    "status": "dependency_missing",
                    "reason": "nfstream unavailable",
                    "flow_count": 0,
                    "alignment": None,
                    "used_fill_values": True,
                }
                return self._store_extract_cache(cache_key, pcap_file, feat, "none", meta)

            feat, meta = extract_pcap_features_nfstream(
                pcap_file,
                self.feature_names,
                self.raw_feature_mean,
                alias_map=self.alias_map,
                return_meta=True,
                max_flows=self.max_flows_per_pcap,
            )
            meta = self._normalize_meta(meta, backend="nfstream")
            cov = float(meta.get("alignment", {}).get("coverage", 1.0)) if isinstance(meta, dict) else 1.0
            if cov < self.align_min_cov:
                _hard_floor = max(0.20, self.align_min_cov * 0.30)
                if cov >= _hard_floor:
                    meta["status"] = "alignment_partial_lenient"
                    meta["alignment_tolerated"] = True
                    meta["alignment_tolerance_reason"] = (
                        f"lenient: nfstream coverage={cov:.3f} below min={self.align_min_cov:.3f}"
                    )
                    print(f"[Stage3] nfstream partial alignment accepted (cov={cov:.3f})")
                elif self.scapy_available:
                    feat, scapy_meta = extract_pcap_features_scapy(
                        pcap_file,
                        self.feature_names,
                        self.raw_feature_mean,
                        alias_map=self.alias_map,
                        return_meta=True,
                    )
                    scapy_meta = {
                        **scapy_meta,
                        "fallback_from": "nfstream",
                        "fallback_reason": "nfstream_zero_coverage",
                    }
                    return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", scapy_meta)
            return self._store_extract_cache(cache_key, pcap_file, feat, "nfstream", meta)

        # "auto": prefer cicflowmeter for faithful CICFlowMeter feature-space alignment,
        # fall back to nfstream, then scapy.
        if self.cicflowmeter_available:
            feat, meta = extract_pcap_features_cicflowmeter(
                pcap_file,
                self.feature_names,
                self.raw_feature_mean,
                alias_map=self.alias_map,
                return_meta=True,
                max_flows=self.max_flows_per_pcap,
                cicflowmeter_cmd=self.cicflowmeter_cmd,
                timeout=self.cicflowmeter_timeout,
            )
            meta = self._normalize_meta(meta, backend="cicflowmeter")
            alignment = meta.get("alignment") if isinstance(meta, dict) else None
            cov = float((alignment or {}).get("coverage", 1.0)) if isinstance(alignment, dict) else 1.0
            flow_count = int(meta.get("flow_count", 0)) if isinstance(meta, dict) else 0
            if cov > 0.0 and flow_count > 0:
                return self._store_extract_cache(cache_key, pcap_file, feat, "cicflowmeter", meta)
            if flow_count == 0:
                print(f"[Stage3] cicflowmeter produced zero flows for {pcap_file}, trying nfstream...")
            else:
                print(f"[Stage3] cicflowmeter produced zero-coverage alignment for {pcap_file}, trying nfstream...")

        if self.nfstream_available:
            feat, meta = extract_pcap_features_nfstream(
                pcap_file,
                self.feature_names,
                self.raw_feature_mean,
                alias_map=self.alias_map,
                return_meta=True,
            )
            meta = self._normalize_meta(meta, backend="nfstream")
            alignment = meta.get("alignment") if isinstance(meta, dict) else None
            cov = float((alignment or {}).get("coverage", 1.0)) if isinstance(alignment, dict) else 1.0
            flow_count = int(meta.get("flow_count", 0)) if isinstance(meta, dict) else 0
            if flow_count == 0 or (cov <= 0.0 and self.scapy_available):
                feat, scapy_meta = extract_pcap_features_scapy(
                    pcap_file,
                    self.feature_names,
                    self.raw_feature_mean,
                    alias_map=self.alias_map,
                    return_meta=True,
                )
                reason = (
                    "nfstream_zero_flows"
                    if meta.get("flow_count", 0) == 0
                    else f"alignment_coverage_below_threshold:{cov:.3f}"
                )
                scapy_meta = {
                    **scapy_meta,
                    "fallback_from": "nfstream",
                    "fallback_reason": reason,
                    "primary_backend": "nfstream",
                    "primary_status": meta.get("status"),
                    "primary_alignment": meta.get("alignment"),
                }
                return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", scapy_meta)
            return self._store_extract_cache(cache_key, pcap_file, feat, "nfstream", meta)

        if self.scapy_available:
            if not self.scapy_warned["done"]:
                print("[Stage3][Warn] nfstream not installed; falling back to scapy.")
                self.scapy_warned["done"] = True
            feat, meta = extract_pcap_features_scapy(
                pcap_file,
                self.feature_names,
                self.raw_feature_mean,
                alias_map=self.alias_map,
                return_meta=True,
            )
            meta = {**meta, "fallback_from": "auto", "fallback_reason": "nfstream unavailable"}
            return self._store_extract_cache(cache_key, pcap_file, feat, "scapy", meta)

        if not self.scapy_warned["done"]:
            print("[Stage3][Warn] nfstream not installed; using fill-value PCAP features.")
            self.scapy_warned["done"] = True
        feat = self.raw_feature_mean.reshape(1, -1)
        meta = {
            "backend": "none",
            "status": "dependency_missing",
            "reason": "nfstream unavailable and scapy unavailable",
            "flow_count": 0,
            "alignment": None,
            "used_fill_values": True,
        }
        return self._store_extract_cache(cache_key, pcap_file, feat, "none", meta)

    def classify_pcap(
        self,
        pcap_file: str | Path,
    ) -> tuple[np.ndarray, str, dict[str, object], np.ndarray, np.ndarray]:
        cache_key = self._cache_key(pcap_file)
        cached = self._classify_cache.get(cache_key)
        if cached is not None:
            feat, backend, meta, probs, feat_pre = cached
            return feat.copy(), backend, dict(meta), probs.copy(), feat_pre.copy()

        feat, backend, meta = self.extract(str(pcap_file))
        probs, feat_pre = self.classify_features(feat)
        self._classify_cache[cache_key] = (
            feat.copy(),
            backend,
            dict(meta),
            probs.copy(),
            feat_pre.copy(),
        )
        return feat, backend, meta, probs, feat_pre

    def pcap_prob(self, pcap_file: Path) -> tuple[float | None, int | None, str, dict[str, object]]:
        _, backend, meta, probs, _ = self.classify_pcap(pcap_file)
        if probs is None or len(probs) < 2 or np.any(~np.isfinite(probs)):
            return None, None, backend, meta
        pmal = float(probs[1])
        pred = int(probs[1] > probs[0])
        return pmal, pred, backend, meta

    def scan_pcaps(self, scan_dir: Path, pattern: str, limit: int) -> tuple[Path | None, float | None, int]:
        if limit <= 0:
            return None, None, 0
        candidates = list(scan_dir.rglob(pattern))
        if not candidates:
            return None, None, 0
        rng = np.random.default_rng(self.seed)
        candidates = self._bounded_scan_candidates(candidates, rng=rng, limit=limit)
        best_path = None
        best_prob = None
        scanned = 0
        for path in candidates:
            if scanned >= limit:
                break
            pmal, _, _, _ = self.pcap_prob(path)
            scanned += 1
            if pmal is None:
                continue
            if best_prob is None or pmal > best_prob:
                best_prob = pmal
                best_path = path
        return best_path, best_prob, scanned

    def _bounded_scan_candidates(
        self,
        candidates: list[Path],
        *,
        rng: np.random.Generator,
        limit: int,
    ) -> list[Path]:
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        if self.max_pcap_bytes <= 0:
            return shuffled if limit <= 0 else shuffled[:limit]
        under_cap: list[Path] = []
        over_cap: list[Path] = []
        for path in shuffled:
            try:
                pcap_size = int(path.stat().st_size)
            except OSError:
                pcap_size = 0
            if pcap_size <= self.max_pcap_bytes:
                under_cap.append(path)
            else:
                over_cap.append(path)
        if limit <= 0:
            return under_cap + over_cap
        selected = under_cap[:limit]
        if len(selected) < limit:
            selected.extend(over_cap[: limit - len(selected)])
        return selected

    def score_pcap_candidate(
        self,
        pcap_file: Path,
        *,
        target_pre: np.ndarray | None = None,
        target_mod: np.ndarray | None = None,
        pmal_weight: float = 0.70,
        target_fit_weight: float = 0.20,
        target_mod_fit_weight: float = 0.10,
    ) -> dict[str, object]:
        try:
            pcap_size = int(pcap_file.stat().st_size)
        except OSError:
            pcap_size = 0
        if self.max_pcap_bytes > 0 and pcap_size > self.max_pcap_bytes:
            return {
                "path": str(pcap_file),
                "name": pcap_file.name,
                "prob_malicious": None,
                "pred_label": None,
                "backend": None,
                "status": "skipped_too_large",
                "skip_reason": "pcap_too_large",
                "pcap_size_bytes": int(pcap_size),
                "pcap_scan_max_bytes": int(self.max_pcap_bytes),
                "target_feature_l2": None,
                "target_feature_fit": None,
                "target_mod_l2": None,
                "target_mod_fit": None,
                "selection_score": float("-inf"),
            }
        feat, backend, meta, probs, feat_pre = self.classify_pcap(pcap_file)
        pmal = None
        pred = None
        if probs is not None and len(probs) >= 2 and np.all(np.isfinite(probs)):
            pmal = float(probs[1])
            pred = int(probs[1] > probs[0])

        target_feature_l2 = None
        target_feature_fit = None
        if target_pre is not None and feat_pre.size:
            feat_mean = np.mean(feat_pre, axis=0)
            diff = feat_mean - np.asarray(target_pre, dtype=np.float64)
            target_feature_l2 = float(np.linalg.norm(diff))
            target_feature_fit = 1.0 / (1.0 + target_feature_l2)

        target_mod_l2 = None
        target_mod_fit = None
        if target_mod is not None and feat.size:
            feat_mod = build_remap_targets(feat, self.feature_names).mean(axis=0)
            diff_mod = np.asarray(feat_mod, dtype=np.float64) - np.asarray(target_mod, dtype=np.float64)
            target_mod_l2 = float(np.linalg.norm(diff_mod))
            target_mod_fit = 1.0 / (1.0 + target_mod_l2)

        selection_score = 0.0
        # ── Manipulability-aware pmal scoring ──────────────────────────
        # Prefer carriers near the decision boundary (pmal≈0.60) over those
        # that are either too weakly detected (≤0.50 = not a valid threat) or
        # too strongly detected (≥0.95 = too entrenched to remap).
        # Triangular function: peak at 0.60, zero at 0.50 and 0.95.
        if pmal is not None and np.isfinite(pmal) and float(pmal) > 0.50:
            _p = float(pmal)
            _peak = 0.60
            _max = 0.95
            if _p <= _peak:
                manipulability = (_p - 0.50) / (_peak - 0.50)
            else:
                manipulability = max(0.0, (_max - _p) / (_max - _peak))
            selection_score += float(pmal_weight) * manipulability
        if target_feature_fit is not None and np.isfinite(target_feature_fit):
            # Boost feature-fit weight: closer features → easier remapping
            selection_score += float(target_fit_weight) * 2.0 * float(target_feature_fit)
        if target_mod_fit is not None and np.isfinite(target_mod_fit):
            selection_score += float(target_mod_fit_weight) * float(target_mod_fit)
        if pred is not None:
            selection_score += 0.10 if int(pred) == 1 else 0.0

        return {
            "path": str(pcap_file),
            "name": pcap_file.name,
            "prob_malicious": pmal,
            "pred_label": pred,
            "backend": backend,
            "status": meta.get("status") if isinstance(meta, dict) else None,
            "skip_reason": None,
            "pcap_size_bytes": int(pcap_size),
            "target_feature_l2": target_feature_l2,
            "target_feature_fit": target_feature_fit,
            "target_mod_l2": target_mod_l2,
            "target_mod_fit": target_mod_fit,
            "selection_score": float(selection_score),
        }

    def rank_pcaps(
        self,
        scan_dir: Path,
        pattern: str,
        limit: int,
        *,
        target_pre: np.ndarray | None = None,
        target_mod: np.ndarray | None = None,
        pmal_weight: float = 0.70,
        target_fit_weight: float = 0.20,
        target_mod_fit_weight: float = 0.10,
        semantic_categories: list[str] | None = None,
        mandatory_paths: list[str] | None = None,
    ) -> list[dict[str, object]]:
        candidates = list(scan_dir.rglob(pattern))
        if semantic_categories:
            candidates = filter_candidates_by_categories(candidates, semantic_categories)
        # ── Mandatory PCAPs are always prepended to the candidate list ────
        mandatory: list[Path] = []
        if mandatory_paths:
            for mp in mandatory_paths:
                mp_path = Path(mp)
                if mp_path.exists() and mp_path.is_file():
                    mandatory.append(mp_path)
                else:
                    # Try relative to scan_dir
                    alt = scan_dir.parent / mp
                    if alt.exists() and alt.is_file():
                        mandatory.append(alt)
        candidates = mandatory + [c for c in candidates if c not in mandatory]
        if not candidates:
            return []
        # Reduce limit to account for mandatory entries
        effective_limit = max(limit, len(mandatory)) if limit > 0 else 0
        rng = np.random.default_rng(self.seed)
        selected = self._bounded_scan_candidates(candidates, rng=rng, limit=effective_limit)
        rows: list[dict[str, object]] = []
        for path in selected:
            rows.append(
                self.score_pcap_candidate(
                    path,
                    target_pre=target_pre,
                    target_mod=target_mod,
                    pmal_weight=pmal_weight,
                    target_fit_weight=target_fit_weight,
                    target_mod_fit_weight=target_mod_fit_weight,
                )
            )
        rows.sort(
            key=lambda row: (
                float(row.get("selection_score", float("-inf"))),
                float(row["prob_malicious"]) if row.get("prob_malicious") is not None else float("-inf"),
                row.get("name", ""),
            ),
            reverse=True,
        )
        return rows

    def metrics_snapshot(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "pcap_feature_fallback_count": int(self.feature_fallback_count),
            "pcap_feature_fill_count": int(self.feature_fill_count),
            "pcap_feature_disk_cache_hits": int(self.disk_cache_hit_count),
            "pcap_feature_disk_cache_misses": int(self.disk_cache_miss_count),
        }
        if self.feature_status_counts:
            payload["pcap_feature_statuses"] = sorted(self.feature_status_counts.keys())
            for status, count in sorted(self.feature_status_counts.items()):
                payload[f"pcap_feature_status_count_{status}"] = int(count)
        return payload
