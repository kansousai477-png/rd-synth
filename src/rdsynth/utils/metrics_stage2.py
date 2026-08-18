from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

KEY_CORR_DELTA = "CorrDelta"
KEY_CORR_DELTA_ST = "CorrDelta_ST"
KEY_CORR_DELTA_SP = "CorrDelta_SP"
KEY_CORR_DELTA_TP = "CorrDelta_TP"
KEY_DMEAN_S = "DeltaMean_S"
KEY_DMEAN_T = "DeltaMean_T"
KEY_DMEAN_P = "DeltaMean_P"
KEY_DSTD_S = "DeltaStd_S"
KEY_DSTD_T = "DeltaStd_T"
KEY_DSTD_P = "DeltaStd_P"


def _subsample(x: np.ndarray, n: int, seed: int) -> np.ndarray:
    if n >= len(x):
        return x
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), n, replace=False)
    return x[idx]


def _covariance_matrix(x: np.ndarray, eps: float) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return None
    cov = np.cov(x, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    cov = np.atleast_2d(cov)
    if cov.shape != (x.shape[1], x.shape[1]):
        if x.shape[1] == 1 and cov.size == 1:
            cov = cov.reshape(1, 1)
        else:
            return None
    return cov + eps * np.eye(x.shape[1], dtype=np.float64)


def _corrcoef_matrix(x: np.ndarray) -> np.ndarray | None:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.atleast_2d(corr)
    if corr.shape != (x.shape[1], x.shape[1]):
        if x.shape[1] == 1 and corr.size == 1:
            corr = corr.reshape(1, 1)
        else:
            return None
    return corr


def frechet_distance(x_real: np.ndarray, x_gen: np.ndarray, eps: float = 1.0e-8) -> float:
    mu_r = x_real.mean(axis=0)
    mu_g = x_gen.mean(axis=0)
    cr = _covariance_matrix(x_real, eps=eps)
    cg = _covariance_matrix(x_gen, eps=eps)
    if cr is None or cg is None:
        return float("nan")

    vals_r, vecs_r = np.linalg.eigh(cr)
    vals_r = np.clip(vals_r, eps, None)
    cr_sqrt = (vecs_r * np.sqrt(vals_r)) @ vecs_r.T

    mid = cr_sqrt @ cg @ cr_sqrt
    vals_m, vecs_m = np.linalg.eigh(mid + eps * np.eye(mid.shape[0], dtype=np.float64))
    vals_m = np.clip(vals_m, eps, None)
    mid_sqrt = (vecs_m * np.sqrt(vals_m)) @ vecs_m.T

    diff = mu_r - mu_g
    return float(np.sum(diff * diff) + np.trace(cr + cg - 2.0 * mid_sqrt))


def sliced_wasserstein_distance(x_real: np.ndarray, x_gen: np.ndarray, n_proj: int = 128, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    n = min(len(x_real), len(x_gen))
    if n <= 8:
        return float("nan")
    xr = _subsample(x_real, n, seed=seed)
    xg = _subsample(x_gen, n, seed=seed + 1)
    d = xr.shape[1]
    acc = 0.0
    for _ in range(n_proj):
        v = rng.normal(size=d)
        v = v / (np.linalg.norm(v) + 1.0e-12)
        pr = np.sort(xr @ v)
        pg = np.sort(xg @ v)
        acc += float(np.mean(np.abs(pr - pg)))
    return acc / n_proj


def energy_distance(x_real: np.ndarray, x_gen: np.ndarray, n: int = 2000, chunk: int = 400) -> float:
    n = min(n, len(x_real), len(x_gen))
    if n < 50:
        return float("nan")
    xr = _subsample(x_real, n, seed=101).astype(np.float64)
    xg = _subsample(x_gen, n, seed=102).astype(np.float64)

    def mean_cdist(a: np.ndarray, b: np.ndarray, same: bool = False) -> float:
        s = 0.0
        cnt = 0
        for i in range(0, len(a), chunk):
            ai = a[i : i + chunk]
            d = np.linalg.norm(ai[:, None, :] - b[None, :, :], axis=2)
            if same:
                start = i
                for r in range(len(ai)):
                    j = start + r
                    if j < len(b):
                        d[r, j] = np.nan
            s += float(np.nansum(d))
            cnt += int(np.sum(np.isfinite(d)))
        return s / max(1, cnt)

    exy = mean_cdist(xr, xg, same=False)
    exx = mean_cdist(xr, xr, same=True)
    eyy = mean_cdist(xg, xg, same=True)
    return float(exy - 0.5 * exx - 0.5 * eyy)


def c2st_metrics(x_real: np.ndarray, x_gen: np.ndarray, seed: int = 42, test_size: float = 0.3) -> Tuple[float, float]:
    x_real = np.nan_to_num(np.asarray(x_real, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x_gen = np.nan_to_num(np.asarray(x_gen, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if len(x_real) < 8 or len(x_gen) < 8:
        return float("nan"), float("nan")
    if np.all(np.std(x_real, axis=0) < 1e-12) or np.all(np.std(x_gen, axis=0) < 1e-12):
        return float("nan"), float("nan")
    x = np.vstack([x_real, x_gen])
    y = np.hstack([np.zeros(len(x_real)), np.ones(len(x_gen))])
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)
    if xtr.shape[0] < 4 or xte.shape[0] < 4:
        return float("nan"), float("nan")
    sc = StandardScaler()
    xtr = sc.fit_transform(xtr)
    xte = sc.transform(xte)
    pca_dim = min(32, xtr.shape[1], max(1, xtr.shape[0] // 15))
    if pca_dim < 1:
        return float("nan"), float("nan")
    pca = PCA(n_components=pca_dim)
    xtr = pca.fit_transform(xtr)
    xte = pca.transform(xte)
    clf = LogisticRegression(max_iter=500, n_jobs=1)
    clf.fit(xtr, ytr)
    p = clf.predict_proba(xte)[:, 1]
    try:
        auc = float(roc_auc_score(yte, p))
    except ValueError:
        auc = float("nan")
    acc = float(np.mean((p >= 0.5) == yte))
    return auc, acc


def _knn_radii(x: np.ndarray, k: int, metric: str) -> np.ndarray:
    if len(x) <= 1:
        return np.full((len(x),), float("nan"), dtype=np.float64)
    k_eff = max(1, min(k, len(x) - 1))
    nnm = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric).fit(x)
    dist, _ = nnm.kneighbors(x, return_distance=True)
    return dist[:, k_eff]


def coverage_at_k(x_real: np.ndarray, x_gen: np.ndarray, k: int = 5, metric: str = "euclidean") -> float:
    if len(x_real) == 0 or len(x_gen) == 0:
        return float("nan")
    r = _knn_radii(x_real, k=k, metric=metric)
    nnm = NearestNeighbors(n_neighbors=1, metric=metric).fit(x_gen)
    dist, _ = nnm.kneighbors(x_real, return_distance=True)
    return float(np.mean(dist[:, 0] <= r))


def knn_precision_recall(
    x_real: np.ndarray, x_gen: np.ndarray, k: int = 5, metric: str = "euclidean"
) -> Tuple[float, float]:
    if len(x_real) == 0 or len(x_gen) == 0:
        return float("nan"), float("nan")
    r_real = _knn_radii(x_real, k=k, metric=metric)
    r_gen = _knn_radii(x_gen, k=k, metric=metric)

    nn_real = NearestNeighbors(n_neighbors=1, metric=metric).fit(x_real)
    d_g2r, idx_g2r = nn_real.kneighbors(x_gen, return_distance=True)
    precision = float(np.mean(d_g2r[:, 0] <= r_real[idx_g2r[:, 0]]))

    nn_gen = NearestNeighbors(n_neighbors=1, metric=metric).fit(x_gen)
    d_r2g, idx_r2g = nn_gen.kneighbors(x_real, return_distance=True)
    recall = float(np.mean(d_r2g[:, 0] <= r_gen[idx_r2g[:, 0]]))
    return precision, recall


def corr_delta(x_real: np.ndarray, x_gen: np.ndarray) -> float:
    cr = _corrcoef_matrix(x_real)
    cg = _corrcoef_matrix(x_gen)
    if cr is None or cg is None:
        return float("nan")
    return float(np.linalg.norm(cr - cg, ord="fro") / (cr.shape[0] + 1.0e-12))


def _infer_groups(feature_names):
    groups = {"temporal": [], "spatial": [], "protocol": []}
    for idx, name in enumerate(feature_names):
        n = name.lower()
        if any(k in n for k in ["duration", "iat", "active", "idle", "time"]):
            groups["temporal"].append(idx)
        elif any(
            k in n
            for k in [
                "packet",
                "pkt",
                "bytes",
                "size",
                "segment",
                "subflow",
                "rate",
                "ps",
                "length",
                "mean",
                "std",
                "variance",
            ]
        ):
            groups["spatial"].append(idx)
        elif any(
            k in n
            for k in ["port", "protocol", "flag", "header", "win", "ratio", "ack", "fin", "syn", "urg", "cwr", "ece"]
        ):
            groups["protocol"].append(idx)
    total = sum(len(v) for v in groups.values())
    if total == 0:
        n = len(feature_names)
        groups["temporal"] = list(range(0, n // 3))
        groups["spatial"] = list(range(n // 3, 2 * n // 3))
        groups["protocol"] = list(range(2 * n // 3, n))
    return groups


def corr_delta_blocks(x_real: np.ndarray, x_gen: np.ndarray, groups: dict) -> Dict[str, float]:
    cr = _corrcoef_matrix(x_real)
    cg = _corrcoef_matrix(x_gen)
    if cr is None or cg is None:
        return {
            "Corr螖_ST": float("nan"),
            "Corr螖_SP": float("nan"),
            "Corr螖_TP": float("nan"),
        }

    def _fro(a, b):
        if not a or not b:
            return float("nan")
        br = cr[np.ix_(a, b)]
        bg = cg[np.ix_(a, b)]
        return float(np.linalg.norm(br - bg, ord="fro") / (br.size + 1.0e-12))

    idx_t = groups["temporal"]
    idx_s = groups["spatial"]
    idx_p = groups["protocol"]
    return {
        "CorrΔ_ST": _fro(idx_s, idx_t),
        "CorrΔ_SP": _fro(idx_s, idx_p),
        "CorrΔ_TP": _fro(idx_t, idx_p),
    }


def group_moments_delta(x_real: np.ndarray, x_gen: np.ndarray, groups: dict) -> Dict[str, float]:
    out = {}
    for name, label in [("spatial", "S"), ("temporal", "T"), ("protocol", "P")]:
        idx = groups[name]
        if not idx:
            out[f"ΔMean_{label}"] = float("nan")
            out[f"ΔStd_{label}"] = float("nan")
            continue
        mu_r = np.mean(x_real[:, idx], axis=0)
        mu_g = np.mean(x_gen[:, idx], axis=0)
        sd_r = np.std(x_real[:, idx], axis=0)
        sd_g = np.std(x_gen[:, idx], axis=0)
        out[f"ΔMean_{label}"] = float(np.mean(np.abs(mu_r - mu_g)))
        out[f"ΔStd_{label}"] = float(np.mean(np.abs(sd_r - sd_g)))
    return out


def violation_rates(
    x_real: np.ndarray,
    x_gen: np.ndarray,
    integer_tol: float = 0.05,
    integer_frac: float = 0.95,
    enable_integer: bool = False,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    nonneg_mask: np.ndarray | None = None,
) -> Dict[str, float]:
    if bounds_min is None or bounds_max is None:
        mn = np.min(x_real, axis=0)
        mx = np.max(x_real, axis=0)
    else:
        mn = bounds_min
        mx = bounds_max
    span = np.maximum(mx - mn, 1.0e-6)
    lo = mn - 0.01 * span
    hi = mx + 0.01 * span

    if nonneg_mask is None:
        nonneg = mn >= -1.0e-8
    else:
        nonneg = nonneg_mask
    if enable_integer:
        frac = np.abs(x_real - np.round(x_real)) <= integer_tol
        integerlike = frac.mean(axis=0) >= integer_frac
    else:
        integerlike = np.zeros(x_real.shape[1], dtype=bool)

    xg = np.asarray(x_gen, dtype=np.float64)
    xg = np.nan_to_num(xg, nan=0.0, posinf=0.0, neginf=0.0)

    vio_range = np.any((xg < lo[None, :]) | (xg > hi[None, :]), axis=1)
    vio_nonneg = np.any(xg[:, nonneg] < -1.0e-8, axis=1) if np.any(nonneg) else np.zeros(len(xg), dtype=bool)
    vio_int = (
        np.any(np.abs(xg[:, integerlike] - np.round(xg[:, integerlike])) > integer_tol, axis=1)
        if np.any(integerlike)
        else np.zeros(len(xg), dtype=bool)
    )

    return {
        "Violation_Range": float(np.mean(vio_range)),
        "Violation_NonNeg": float(np.mean(vio_nonneg)),
        "Violation_Integer": float(np.mean(vio_int)),
    }


def cov_spectrum_metrics(x_real: np.ndarray, x_gen: np.ndarray, eps: float = 1.0e-8) -> Tuple[float, float]:
    cr = _covariance_matrix(x_real, eps=eps)
    cg = _covariance_matrix(x_gen, eps=eps)
    if cr is None or cg is None:
        return float("nan"), float("nan")
    er = np.linalg.eigvalsh(cr)
    eg = np.linalg.eigvalsh(cg)
    er = np.clip(er, eps, None)
    eg = np.clip(eg, eps, None)
    lr = np.log(er)
    lg = np.log(eg)
    covspec_l2 = float(np.sqrt(np.mean((lr - lg) ** 2)))
    covtrace_ratio = float(np.trace(cg) / (np.trace(cr) + 1.0e-12))
    return covspec_l2, covtrace_ratio


def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(a)
    b = np.sort(b)
    n = len(a)
    m = len(b)
    if n == 0 or m == 0:
        return float("nan")
    i = j = 0
    cdf_a = cdf_b = 0.0
    d = 0.0
    while i < n and j < m:
        if a[i] <= b[j]:
            i += 1
            cdf_a = i / n
        else:
            j += 1
            cdf_b = j / m
        d = max(d, abs(cdf_a - cdf_b))
    return float(d)


def pairwise_distance_metrics(
    x_real: np.ndarray, x_gen: np.ndarray, n: int = 4000, n_pairs: int = 30000, seed: int = 1234
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = min(n, len(x_real), len(x_gen))
    if n < 50:
        return float("nan"), float("nan")
    xr = _subsample(x_real, n, seed=7)
    xg = _subsample(x_gen, n, seed=8)

    i1 = rng.integers(0, n, size=n_pairs)
    i2 = rng.integers(0, n, size=n_pairs)
    mask = i1 != i2
    i1 = i1[mask]
    i2 = i2[mask]
    if i1.size == 0:
        return float("nan"), float("nan")

    dr = np.linalg.norm(xr[i1] - xr[i2], axis=1)
    dg = np.linalg.norm(xg[i1] - xg[i2], axis=1)
    ks = _ks_statistic(dr, dg)
    mean_ratio = float(np.mean(dg) / (np.mean(dr) + 1.0e-12))
    return ks, mean_ratio


def nearest_reference_distance(
    x_query: np.ndarray,
    x_reference: np.ndarray,
    metric: str = "euclidean",
    chunk_size: int = 4096,
) -> float:
    x_query = np.asarray(x_query, dtype=np.float64)
    x_reference = np.asarray(x_reference, dtype=np.float64)
    if x_query.size == 0 or x_reference.size == 0:
        return float("nan")
    nnm = NearestNeighbors(n_neighbors=1, metric=metric)
    nnm.fit(x_reference)
    distances = []
    for start in range(0, x_query.shape[0], chunk_size):
        batch = x_query[start : start + chunk_size]
        dist, _ = nnm.kneighbors(batch, return_distance=True)
        distances.append(dist[:, 0])
    return float(np.mean(np.concatenate(distances, axis=0)))


def paired_sample_l2(
    x_left: np.ndarray,
    x_right: np.ndarray,
    *,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
) -> float:
    x_left = np.asarray(x_left, dtype=np.float64)
    x_right = np.asarray(x_right, dtype=np.float64)
    if x_left.shape != x_right.shape or x_left.size == 0:
        return float("nan")
    x_left = np.nan_to_num(x_left, nan=0.0, posinf=0.0, neginf=0.0)
    x_right = np.nan_to_num(x_right, nan=0.0, posinf=0.0, neginf=0.0)
    if bounds_min is not None and bounds_max is not None:
        lo = np.asarray(bounds_min, dtype=np.float64)
        hi = np.asarray(bounds_max, dtype=np.float64)
        x_left = np.clip(x_left, lo, hi)
        x_right = np.clip(x_right, lo, hi)
    return float(np.mean(np.linalg.norm(x_left - x_right, axis=1)))


@dataclass
class Stage2Metrics:
    ffd: float = float("nan")
    swd: float = float("nan")
    energy: float = float("nan")
    c2st_auc: float = float("nan")
    c2st_acc: float = float("nan")
    coverage_at_1: float = float("nan")
    coverage_at_5: float = float("nan")
    coverage_at_10: float = float("nan")
    knn_precision_1: float = float("nan")
    knn_recall_1: float = float("nan")
    knn_precision_5: float = float("nan")
    knn_recall_5: float = float("nan")
    knn_precision_10: float = float("nan")
    knn_recall_10: float = float("nan")
    knn_precision: float = float("nan")
    knn_recall: float = float("nan")
    corr_delta: float = float("nan")
    covspec_l2: float = float("nan")
    covtrace_ratio: float = float("nan")
    pairdist_ks: float = float("nan")
    pairmean_ratio: float = float("nan")
    iat_adv_ben_mean_abs: float = float("nan")
    iat_adv_mal_mean_abs: float = float("nan")
    iat_adv_ben_std_abs: float = float("nan")
    iat_adv_mal_std_abs: float = float("nan")
    dmean_s: float = float("nan")
    dmean_t: float = float("nan")
    dmean_p: float = float("nan")
    dstd_s: float = float("nan")
    dstd_t: float = float("nan")
    dstd_p: float = float("nan")
    vio_range: float = float("nan")
    vio_nonneg: float = float("nan")
    vio_integer: float = float("nan")
    ffd_pca: float = float("nan")
    swd_pca: float = float("nan")
    energy_pca: float = float("nan")
    corr_delta_st: float = float("nan")
    corr_delta_sp: float = float("nan")
    corr_delta_tp: float = float("nan")
    knn_precision_1: float
    knn_recall_1: float
    knn_precision_10: float
    knn_recall_10: float
    ffd_pca: float
    swd_pca: float
    energy_pca: float
    corr_delta_st: float
    corr_delta_sp: float
    corr_delta_tp: float
    dmean_s: float
    dmean_t: float
    dmean_p: float
    dstd_s: float
    dstd_t: float
    dstd_p: float
    vio_range: float
    vio_nonneg: float
    vio_integer: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "FFD": self.ffd,
            "SWD": self.swd,
            "Energy": self.energy,
            "C2ST-AUC": self.c2st_auc,
            "C2ST-Acc": self.c2st_acc,
            "Coverage@5": self.coverage_at_5,
            "kNN-P": self.knn_precision,
            "kNN-R": self.knn_recall,
            "CorrΔ": self.corr_delta,
            "CovSpec-L2": self.covspec_l2,
            "CovTrace": self.covtrace_ratio,
            "PairDist-KS": self.pairdist_ks,
            "PairMean": self.pairmean_ratio,
            "Coverage@1": self.coverage_at_1,
            "Coverage@10": self.coverage_at_10,
            "kNN-P@1": self.knn_precision_1,
            "kNN-R@1": self.knn_recall_1,
            "kNN-P@10": self.knn_precision_10,
            "kNN-R@10": self.knn_recall_10,
            "FFD-PCA": self.ffd_pca,
            "SWD-PCA": self.swd_pca,
            "Energy-PCA": self.energy_pca,
            "CorrΔ_ST": self.corr_delta_st,
            "CorrΔ_SP": self.corr_delta_sp,
            "CorrΔ_TP": self.corr_delta_tp,
            "ΔMean_S": self.dmean_s,
            "ΔMean_T": self.dmean_t,
            "ΔMean_P": self.dmean_p,
            "ΔStd_S": self.dstd_s,
            "ΔStd_T": self.dstd_t,
            "ΔStd_P": self.dstd_p,
            "Violation_Range": self.vio_range,
            "Violation_NonNeg": self.vio_nonneg,
            "Violation_Integer": self.vio_integer,
        }


def compute_stage2_metrics(
    x_real: np.ndarray,
    x_gen: np.ndarray,
    feature_names: list[str],
    max_real: int = 5000,
    max_gen: int = 5000,
    seed: int = 42,
    bounds_min: np.ndarray | None = None,
    bounds_max: np.ndarray | None = None,
    nonneg_mask: np.ndarray | None = None,
) -> Stage2Metrics:
    xr = _subsample(x_real, min(max_real, len(x_real)), seed=seed)
    xg = _subsample(x_gen, min(max_gen, len(x_gen)), seed=seed + 1)
    xr = np.nan_to_num(np.asarray(xr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    xg = np.nan_to_num(np.asarray(xg, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if xr.shape[0] < 2 or xg.shape[0] < 2 or xr.shape[1] < 2:
        nan = float("nan")
        return Stage2Metrics(ffd=nan, swd=nan, energy=nan, c2st_auc=nan, c2st_acc=nan,
                             coverage_at_1=nan, coverage_at_5=nan, coverage_at_10=nan,
                             knn_precision_1=nan, knn_recall_1=nan,
                             knn_precision_5=nan, knn_recall_5=nan,
                             knn_precision_10=nan, knn_recall_10=nan,
                             corr_delta=nan, covspec_l2=nan, covtrace_ratio=nan,
                             pairdist_ks=nan, pairmean_ratio=nan,
                             iat_adv_ben_mean_abs=nan, iat_adv_mal_mean_abs=nan,
                             iat_adv_ben_std_abs=nan, iat_adv_mal_std_abs=nan,
                             dmean_s=nan, dmean_t=nan, dmean_p=nan,
                             dstd_s=nan, dstd_t=nan, dstd_p=nan,
                             vio_range=nan, vio_nonneg=nan, vio_integer=nan,
                             ffd_pca=nan, swd_pca=nan, energy_pca=nan,
                             corr_delta_st=nan, corr_delta_sp=nan, corr_delta_tp=nan)
    if xr.shape[0] == 0 or xg.shape[0] == 0:
        nan = float("nan")
        return Stage2Metrics(
            ffd=nan, swd=nan, energy=nan, c2st_auc=nan, c2st_acc=nan,
            coverage_at_1=nan, coverage_at_5=nan, coverage_at_10=nan,
            knn_precision_1=nan, knn_recall_1=nan,
            knn_precision_5=nan, knn_recall_5=nan,
            knn_precision_10=nan, knn_recall_10=nan,
            knn_precision=nan, knn_recall=nan,
            corr_delta=nan, covspec_l2=nan, covtrace_ratio=nan,
            pairdist_ks=nan, pairmean_ratio=nan,
            iat_adv_ben_mean_abs=nan, iat_adv_mal_mean_abs=nan,
            iat_adv_ben_std_abs=nan, iat_adv_mal_std_abs=nan,
            dmean_s=nan, dmean_t=nan, dmean_p=nan,
            dstd_s=nan, dstd_t=nan, dstd_p=nan,
            vio_range=nan, vio_nonneg=nan, vio_integer=nan,
            ffd_pca=nan, swd_pca=nan, energy_pca=nan,
            corr_delta_st=nan, corr_delta_sp=nan, corr_delta_tp=nan,
        )

    ffd = frechet_distance(xr, xg)
    swd = sliced_wasserstein_distance(xr, xg, n_proj=128, seed=seed)
    energy = energy_distance(xr, xg, n=2000, chunk=400)
    c2st_auc, c2st_acc = c2st_metrics(xr, xg, seed=seed, test_size=0.3)
    cov1 = coverage_at_k(xr, xg, k=1, metric="euclidean")
    cov5 = coverage_at_k(xr, xg, k=5, metric="euclidean")
    cov10 = coverage_at_k(xr, xg, k=10, metric="euclidean")
    knn_p1, knn_r1 = knn_precision_recall(xr, xg, k=1, metric="euclidean")
    knn_p5, knn_r5 = knn_precision_recall(xr, xg, k=5, metric="euclidean")
    knn_p10, knn_r10 = knn_precision_recall(xr, xg, k=10, metric="euclidean")
    corr = corr_delta(xr, xg)
    covspec_l2, covtrace = cov_spectrum_metrics(xr, xg)
    pair_ks, pair_mean = pairwise_distance_metrics(xr, xg, n=4000, n_pairs=30000, seed=seed + 2)
    groups = _infer_groups(feature_names)
    corr_blocks = corr_delta_blocks(xr, xg, groups)
    group_mom = group_moments_delta(xr, xg, groups)

    def _resolve_metric(mapping: Dict[str, float], *names: str) -> float:
        for name in names:
            if name in mapping:
                return mapping[name]
        lowered = [(key, key.lower()) for key in mapping]
        for name in names:
            target = name.lower()
            for key, lower in lowered:
                if lower.endswith(target) or target in lower:
                    return mapping[key]
        return float("nan")

    vio = violation_rates(
        xr,
        xg,
        enable_integer=False,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        nonneg_mask=nonneg_mask,
    )

    pca_dim = min(64, xr.shape[1], xr.shape[0], xg.shape[0])
    if pca_dim >= 1:
        pca = PCA(n_components=pca_dim, random_state=seed)
        xr_p = pca.fit_transform(xr)
        xg_p = pca.transform(xg)
        ffd_p = frechet_distance(xr_p, xg_p)
        swd_p = sliced_wasserstein_distance(xr_p, xg_p, n_proj=128, seed=seed)
        energy_p = energy_distance(xr_p, xg_p, n=2000, chunk=400)
    else:
        ffd_p = float("nan")
        swd_p = float("nan")
        energy_p = float("nan")

    return Stage2Metrics(
        ffd=ffd, swd=swd, energy=energy, c2st_auc=c2st_auc, c2st_acc=c2st_acc,
        coverage_at_1=cov1, coverage_at_5=cov5, coverage_at_10=cov10,
        knn_precision_1=knn_p1, knn_recall_1=knn_r1,
        knn_precision_5=knn_p5, knn_recall_5=knn_r5,
        knn_precision_10=knn_p10, knn_recall_10=knn_r10,
        knn_precision=knn_p5, knn_recall=knn_r5,
        corr_delta=corr, covspec_l2=covspec_l2, covtrace_ratio=covtrace,
        pairdist_ks=pair_ks, pairmean_ratio=pair_mean,
        iat_adv_ben_mean_abs=float("nan"),
        iat_adv_mal_mean_abs=float("nan"),
        iat_adv_ben_std_abs=float("nan"),
        iat_adv_mal_std_abs=float("nan"),
        dmean_s=_resolve_metric(group_mom, "DeltaMean_S", "ΔMean_S", "mean_s"),
        dmean_t=_resolve_metric(group_mom, "DeltaMean_T", "ΔMean_T", "mean_t"),
        dmean_p=_resolve_metric(group_mom, "DeltaMean_P", "ΔMean_P", "mean_p"),
        dstd_s=_resolve_metric(group_mom, "DeltaStd_S", "ΔStd_S", "std_s"),
        dstd_t=_resolve_metric(group_mom, "DeltaStd_T", "ΔStd_T", "std_t"),
        dstd_p=_resolve_metric(group_mom, "DeltaStd_P", "ΔStd_P", "std_p"),
        vio_range=vio["Violation_Range"],
        vio_nonneg=vio["Violation_NonNeg"],
        vio_integer=vio["Violation_Integer"],
    )
