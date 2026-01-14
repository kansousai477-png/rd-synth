import os
import json
import glob
import warnings
warnings.filterwarnings("ignore")

from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler as SkStandardScaler
from sklearn.neighbors import NearestNeighbors

from scipy import linalg
from scipy import stats
from scipy.spatial.distance import cdist


# ============================================================
# ✅ Configure here (ONLY edit these paths if needed)
# ============================================================
CFG = {
    # --------------------------
    # DATA (binary CSV)
    # --------------------------
    "csv_path": "../data/unsw/CICFlowMeter_preprocessed.csv",
    "label_col": "Label",
    "benign_value": 0,
    "chunksize": 200000,

    # --------------------------
    # CANONICAL baseline artifacts (meta.json + standardizer.npz)
    # --------------------------
    "baseline_artifact_dir": "artifacts/baselines_light_nb15_v2",

    # --------------------------
    # ✅ Multi-seed entry produced by training script
    # (top-level manifest that lists run_root per seed)
    # --------------------------
    "all_seeds_runs_path": "results_nb15_rdsynth_ablation/all_seeds_runs.json",

    # --------------------------
    # Optional: include baseline_zoo methods via its exported manifest
    # If you don't want baseline rows in the table, set to False.
    # --------------------------
    "baseline_export_subdir": "adv_npz",
    "baseline_export_prefix": "NB15_",  # used to pick manifest; can be "".
    "include_baseline_manifest": False,

    # --------------------------
    # METRICS
    # --------------------------
    "metrics": {
        "n_ben": 150000,
        "max_real": 50000,
        "max_gen":  50000,

        "knn_k": 5,
        "nn_metric": "euclidean",

        "ffd_eps": 1e-6,
        "ffd_diag": True,
        "ffd_clip_negative_trace": True,

        "mmd_gamma": None,
        "rff_dim": 2048,
        "rff_seed": 42,

        "swd_projections": 128,
        "swd_seed": 42,

        "c2st_test_size": 0.3,
        "c2st_seed": 42,
        "c2st_max_iter": 2000,

        "range_margin": 0.02,
        "infer_nonneg": True,
        "infer_integerlike": True,
        "integer_tol": 1e-3,
        "integer_frac_threshold": 0.98,

        "univar_max_dim": None,

        "pairdist_n": 4000,
        "pairdist_pairs": 30000,
        "pairdist_seed": 1234,

        "energy_n": 2000,
        "energy_chunk": 400,

        "mmd_perm_enable": True,
        "mmd_perm_trials": 200,
        "mmd_perm_subset": 4000,
        "mmd_perm_seed": 123,

        # debug
        "debug_print_keys": True,
        "debug_print_stats": True,
        "debug_check_cols_match": True,
        "strict_cols_check": False,
        "debug_check_std_mismatch": True,
        "std_mismatch_warn_thr": 1e-3,
        "debug_force_raw_if_available": True,
        "record_std_mismatch": True,

        # outputs
        "out_dir": "results/metrics_suite_v3_unsw_ablation_multiseed",
        "save_txt": True,
    },
}


# ============================================================
# Utilities
# ============================================================
def ensure_dir(d: str) -> str:
    os.makedirs(d, exist_ok=True)
    return d

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def norm_cols(cols) -> List[str]:
    out = []
    for c in cols:
        if isinstance(c, bytes):
            c = c.decode("utf-8", "ignore")
        out.append(str(c))
    return out

def subsample(X: np.ndarray, n_max: Optional[int], seed: int = 42) -> np.ndarray:
    if n_max is None or len(X) <= int(n_max):
        return X
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), size=int(n_max), replace=False)
    return X[idx]

def quick_stats(name: str, X: np.ndarray):
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    m = float(np.mean(X))
    s = float(np.std(X))
    m_abs = float(np.mean(np.abs(X)))
    mx = float(np.max(np.abs(X))) if X.size else float("nan")
    print(f"[Stats] {name}: mean={m:.4f} std={s:.4f} mean|x|={m_abs:.4f} max|x|={mx:.2f}")


# ============================================================
# Standardizer (shared baseline std)
# ============================================================
class Standardizer:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sigma is not None
        return (X.astype(np.float32) - self.mu[None, :]) / self.sigma[None, :]

    @staticmethod
    def load(path_npz: str) -> "Standardizer":
        d = np.load(path_npz)
        s = Standardizer()
        s.mu = d["mu"].astype(np.float32)
        s.sigma = d["sigma"].astype(np.float32)
        s.sigma = np.where(s.sigma < 1e-6, 1.0, s.sigma).astype(np.float32)
        return s


# ============================================================
# Chunk alignment + benign loader (CANONICAL meta.cols)
# ============================================================
def align_chunk_to_cols(ch: pd.DataFrame, feat_cols: List[str], label_col: str) -> pd.DataFrame:
    for c in feat_cols:
        if c not in ch.columns:
            ch[c] = 0.0

    keep = set(feat_cols + [label_col])
    drop = [c for c in ch.columns if c not in keep]
    if drop:
        ch = ch.drop(columns=drop)

    if label_col not in ch.columns:
        raise ValueError(f"CSV missing label_col={label_col}")
    return ch[feat_cols + [label_col]]

def load_benign_chunked_raw_canonical(
    csv_path: str,
    label_col: str,
    feat_cols: List[str],
    n_ben: int,
    benign_value: int = 0,
    seed: int = 42,
    chunksize: int = 200000,
) -> np.ndarray:
    print(f"[Real] Loading benign (chunked raw, canonical cols): {csv_path}")
    need = int(n_ben)
    buf = []
    for ch in pd.read_csv(csv_path, chunksize=int(chunksize), low_memory=False):
        ch = ch.replace([np.inf, -np.inf], np.nan).fillna(0)
        ch = align_chunk_to_cols(ch, feat_cols, label_col)

        bmask = (ch[label_col] == benign_value)
        if not bmask.any():
            continue

        dfb = ch.loc[bmask, feat_cols].copy()
        take = min(need, len(dfb))
        if take <= 0:
            break

        buf.append(dfb.sample(n=take, random_state=seed))
        need -= take
        if need <= 0:
            break

    if len(buf) == 0:
        raise RuntimeError("No benign rows found.")
    Xb_raw = pd.concat(buf, ignore_index=True).to_numpy(dtype=np.float32)
    print(f"[Real] Sampled benign raw: {Xb_raw.shape}")
    return Xb_raw


# ============================================================
# STP split (same logic as your v3 suite)
# ============================================================
def split_feature_blocks_unsw(cols: List[str]):
    cols_norm = [str(c).strip().lower().replace(" ", "_") for c in cols]

    def has_any(s, keys):
        return any(k in s for k in keys)

    time_keys = ["duration", "iat", "flow_duration", "active", "idle", "flow_iat", "fwd_iat", "bwd_iat", "start_time", "end_time"]
    size_keys = ["packet", "length", "bytes", "pkts", "packets", "total_length", "totlen", "payload", "avg", "mean", "min", "max", "std", "rate", "bytes/s", "packets/s"]
    proto_keys = ["port", "protocol", "flag", "tcp", "udp", "icmp", "syn", "ack", "fin", "rst", "psh", "urg", "header", "window", "mss"]

    idxT, idxS, idxP = [], [], []
    for i, s in enumerate(cols_norm):
        isT = has_any(s, time_keys)
        isP = has_any(s, proto_keys)
        isS = has_any(s, size_keys)

        if isT:
            idxT.append(i)
        elif isP:
            idxP.append(i)
        elif isS:
            idxS.append(i)
        else:
            idxS.append(i)

    print(f"[STP-UNSW] T={len(idxT)}, S={len(idxS)}, P={len(idxP)} | total={len(cols)}")
    return idxT, idxS, idxP


# ============================================================
# Gen loader: always convert to shared std space
# ============================================================
def check_cols_match(npz_obj, meta_cols: list, method: str, strict: bool = False) -> bool:
    if "cols" not in npz_obj:
        print(f"[Cols] {method}: NPZ has no 'cols' key -> skip cols check.")
        return True
    cols_npz = npz_obj["cols"]
    try:
        cols_npz = cols_npz.tolist()
    except Exception:
        cols_npz = list(cols_npz)

    a = norm_cols(cols_npz)
    b = norm_cols(meta_cols)
    same = (a == b)
    print(f"[Cols] {method}: cols match meta.cols = {same}")

    if (not same) and strict:
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                raise ValueError(f"[Cols][STRICT] {method} mismatch idx={i}: npz='{x}' vs meta='{y}'")
        raise ValueError(f"[Cols][STRICT] {method} mismatch: different length/content.")
    return same

def _pick_raw_key(npz_obj) -> Optional[str]:
    for k in ["Xadv_raw", "Xraw", "X_adv_raw"]:
        if k in npz_obj:
            return k
    if "Xadv" in npz_obj:
        return "Xadv"
    return None

def check_std_mismatch(npz_obj, std: Standardizer, method: str, thr: float = 1e-3) -> float:
    raw_k = _pick_raw_key(npz_obj)
    if raw_k is None or ("Xadv_std" not in npz_obj):
        return float("nan")

    Xraw = np.asarray(npz_obj[raw_k], dtype=np.float32)
    Xstd_saved = np.asarray(npz_obj["Xadv_std"], dtype=np.float32)

    if Xraw.ndim != 2:
        Xraw = Xraw.reshape((Xraw.shape[0], -1))
    if Xstd_saved.ndim != 2:
        Xstd_saved = Xstd_saved.reshape((Xstd_saved.shape[0], -1))

    n = min(len(Xraw), len(Xstd_saved))
    Xraw = Xraw[:n]
    Xstd_saved = Xstd_saved[:n]

    Xstd_shared = std.transform(Xraw).astype(np.float32)
    diff = float(np.mean(np.abs(Xstd_shared - Xstd_saved)))

    print(f"[StdCheck] {method}: mean|shared_std({raw_k})-saved_Xadv_std| = {diff:.6g}")
    if diff > thr:
        print(f"[StdCheck][WARN] {method}: mismatch > {thr}. Likely uses a different scaler.")
    return diff

def load_gen_as_std(path: str, std: Standardizer, method: str, meta_cols: list, cfgm: dict) -> Tuple[np.ndarray, float]:
    assert os.path.exists(path), f"File not found: {path}"
    ext = os.path.splitext(path)[1].lower()
    std_diff = float("nan")

    if ext == ".npz":
        d = np.load(path, allow_pickle=True)
        keys = list(d.keys())
        if cfgm.get("debug_print_keys", True):
            print(f"[LoadGen] {method}: NPZ keys = {keys}")

        if cfgm.get("debug_check_cols_match", True):
            check_cols_match(d, meta_cols=meta_cols, method=method, strict=bool(cfgm.get("strict_cols_check", False)))

        if cfgm.get("debug_check_std_mismatch", True):
            std_diff = check_std_mismatch(d, std=std, method=method, thr=float(cfgm.get("std_mismatch_warn_thr", 1e-3)))

        raw_k = _pick_raw_key(d)
        if cfgm.get("debug_force_raw_if_available", True) and (raw_k is not None):
            print(f"[LoadGen] {method}: using key={raw_k} -> shared_std.transform(raw)")
            Xraw = np.asarray(d[raw_k], dtype=np.float32)
            if Xraw.ndim != 2:
                Xraw = np.reshape(Xraw, (Xraw.shape[0], -1))
            return std.transform(Xraw).astype(np.float32), std_diff

        if "Xadv_std" in d:
            print(f"[LoadGen] {method}: using key=Xadv_std -> NO transform (already std)")
            Xstd = np.asarray(d["Xadv_std"], dtype=np.float32)
            if Xstd.ndim != 2:
                Xstd = np.reshape(Xstd, (Xstd.shape[0], -1))
            return Xstd.astype(np.float32), std_diff

        raise KeyError(f"[{method}] NPZ keys {keys} have neither raw nor std arrays.")

    raise ValueError(f"Unsupported file type: {path}")


# ============================================================
# Metrics core
# ============================================================
def _cov(X: np.ndarray) -> np.ndarray:
    return np.cov(X, rowvar=False)

def frechet_distance(Xr: np.ndarray, Xg: np.ndarray, eps: float, diag: bool, clip_neg_tr: bool) -> float:
    mu1 = Xr.mean(axis=0)
    mu2 = Xg.mean(axis=0)
    C1 = _cov(Xr) + eps * np.eye(Xr.shape[1], dtype=np.float64)
    C2 = _cov(Xg) + eps * np.eye(Xg.shape[1], dtype=np.float64)

    diff = mu1 - mu2
    covmean = linalg.sqrtm(C1.dot(C2))

    imag_max = 0.0
    if np.iscomplexobj(covmean):
        imag_max = float(np.max(np.abs(np.imag(covmean))))
        covmean = np.real(covmean)

    tr = float(np.trace(C1 + C2 - 2.0 * covmean))
    if diag:
        if imag_max > 1e-3:
            print(f"[FFD][WARN] sqrtm imag_max={imag_max:.3e}")
        if tr < -1e-3:
            print(f"[FFD][WARN] trace term negative: tr={tr:.6g}")
    if clip_neg_tr and tr < 0:
        tr = max(tr, 0.0)
    return float(diff.dot(diff) + tr)

def median_heuristic_gamma(X: np.ndarray, n_ref: int = 4000, seed: int = 42) -> float:
    Xs = subsample(X, min(len(X), n_ref), seed=seed)
    rng = np.random.RandomState(seed)
    idx1 = rng.choice(len(Xs), size=min(1000, len(Xs)), replace=False)
    idx2 = rng.choice(len(Xs), size=min(1000, len(Xs)), replace=False)
    A, B = Xs[idx1], Xs[idx2]
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2).reshape(-1)
    med = np.median(d2)
    med = max(med, 1e-12)
    return 1.0 / (2.0 * med)

def rff_features(X: np.ndarray, gamma: float, rff_dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    d = X.shape[1]
    W = rng.normal(loc=0.0, scale=np.sqrt(2.0 * gamma), size=(rff_dim, d)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=(rff_dim,)).astype(np.float32)
    Z = X.dot(W.T) + b[None, :]
    Phi = np.sqrt(2.0 / rff_dim) * np.cos(Z)
    return Phi.astype(np.float32)

def rff_mmd(Xr: np.ndarray, Xg: np.ndarray, gamma: float, rff_dim: int, seed: int) -> float:
    Pr = rff_features(Xr, gamma=gamma, rff_dim=rff_dim, seed=seed).mean(axis=0)
    Pg = rff_features(Xg, gamma=gamma, rff_dim=rff_dim, seed=seed).mean(axis=0)
    return float(np.sum((Pr - Pg) ** 2))

def knn_radii(X: np.ndarray, k: int, metric: str) -> np.ndarray:
    nnm = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, _ = nnm.kneighbors(X, return_distance=True)
    return dist[:, k]

def coverage_at_k(X_real: np.ndarray, X_gen: np.ndarray, k: int, metric: str) -> float:
    r = knn_radii(X_real, k=k, metric=metric)
    nnm = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_gen)
    dist, _ = nnm.kneighbors(X_real, return_distance=True)
    return float(np.mean(dist[:, 0] <= r))

def knn_precision_recall(X_real: np.ndarray, X_gen: np.ndarray, k: int, metric: str):
    r_real = knn_radii(X_real, k=k, metric=metric)
    r_gen = knn_radii(X_gen, k=k, metric=metric)

    nn_real = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_real)
    d_g2r, idx_g2r = nn_real.kneighbors(X_gen, return_distance=True)
    precision = float(np.mean(d_g2r[:, 0] <= r_real[idx_g2r[:, 0]]))

    nn_gen = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_gen)
    d_r2g, idx_r2g = nn_gen.kneighbors(X_real, return_distance=True)
    recall = float(np.mean(d_r2g[:, 0] <= r_gen[idx_r2g[:, 0]]))
    return precision, recall

def c2st_metrics(X_real: np.ndarray, X_gen: np.ndarray, seed: int, test_size: float, max_iter: int):
    X = np.vstack([X_real, X_gen])
    y = np.concatenate([np.zeros(len(X_real), dtype=np.int64), np.ones(len(X_gen), dtype=np.int64)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

    sc = SkStandardScaler(with_mean=True, with_std=True)
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)

    clf = LogisticRegression(solver="lbfgs", max_iter=max_iter, n_jobs=-1)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, p))
    acc = float(accuracy_score(yte, (p >= 0.5).astype(np.int64)))
    return auc, acc

def sliced_wasserstein_distance(X_real: np.ndarray, X_gen: np.ndarray, n_proj: int, seed: int) -> float:
    rng = np.random.RandomState(seed)
    d = X_real.shape[1]
    n = min(len(X_real), len(X_gen))
    if n <= 8:
        return float("nan")
    A = subsample(X_real, n, seed=seed)
    B = subsample(X_gen,  n, seed=seed + 1)

    sw = 0.0
    for _ in range(int(n_proj)):
        u = rng.normal(size=(d,))
        u = u / (np.linalg.norm(u) + 1e-12)
        pa = np.sort(A.dot(u))
        pb = np.sort(B.dot(u))
        sw += np.mean(np.abs(pa - pb))
    return float(sw / n_proj)

def corr_distance(X_real: np.ndarray, X_gen: np.ndarray, idxT, idxS, idxP) -> dict:
    def corr(X):
        C = np.corrcoef(X, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        return C

    Cr = corr(X_real)
    Cg = corr(X_gen)
    fro_all = float(np.linalg.norm(Cr - Cg, ord="fro") / (Cr.shape[0] + 1e-12))

    def fro_block(a, b):
        Br = Cr[np.ix_(a, b)]
        Bg = Cg[np.ix_(a, b)]
        return float(np.linalg.norm(Br - Bg, ord="fro") / (Br.size + 1e-12))

    return {
        "CorrΔ_all": fro_all,
        "CorrΔ_TS": fro_block(idxT, idxS) if len(idxT) and len(idxS) else float("nan"),
        "CorrΔ_TP": fro_block(idxT, idxP) if len(idxT) and len(idxP) else float("nan"),
        "CorrΔ_SP": fro_block(idxS, idxP) if len(idxS) and len(idxP) else float("nan"),
    }

def infer_constraints_from_real(X_real: np.ndarray, cfgm) -> dict:
    mn = X_real.min(axis=0)
    mx = X_real.max(axis=0)
    margin = float(cfgm["range_margin"])
    span = mx - mn
    lo = mn - margin * span
    hi = mx + margin * span

    nonneg = np.zeros(X_real.shape[1], dtype=bool)
    if cfgm.get("infer_nonneg", True):
        nonneg = (mn >= -1e-8)

    integerlike = np.zeros(X_real.shape[1], dtype=bool)
    if cfgm.get("infer_integerlike", True):
        tol = float(cfgm["integer_tol"])
        frac = np.abs(X_real - np.round(X_real)) <= tol
        frac_rate = frac.mean(axis=0)
        integerlike = (frac_rate >= float(cfgm["integer_frac_threshold"]))

    return {"lo": lo, "hi": hi, "nonneg": nonneg, "integerlike": integerlike}

def violation_rate(X_gen: np.ndarray, constraints: dict, cfgm) -> float:
    lo, hi = constraints["lo"], constraints["hi"]
    nonneg = constraints["nonneg"]
    integerlike = constraints["integerlike"]
    tol = float(cfgm["integer_tol"])

    Xg = np.asarray(X_gen, dtype=np.float64)
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0)

    vio = np.zeros(len(Xg), dtype=bool)
    vio |= np.any((Xg < lo[None, :]) | (Xg > hi[None, :]), axis=1)

    if np.any(nonneg):
        vio |= np.any(Xg[:, nonneg] < -1e-8, axis=1)

    if np.any(integerlike):
        vio |= np.any(np.abs(Xg[:, integerlike] - np.round(Xg[:, integerlike])) > tol, axis=1)

    return float(np.mean(vio))

def nn_overfit_ratio(X_real: np.ndarray, X_gen: np.ndarray, metric: str) -> float:
    nn_r = NearestNeighbors(n_neighbors=1, metric=metric).fit(X_real)
    d_gr, _ = nn_r.kneighbors(X_gen, return_distance=True)
    med_gr = float(np.median(d_gr[:, 0]))

    nn_rr = NearestNeighbors(n_neighbors=2, metric=metric).fit(X_real)
    d_rr, _ = nn_rr.kneighbors(X_real, return_distance=True)
    med_rr = float(np.median(d_rr[:, 1]))

    return float(med_gr / (med_rr + 1e-12))

def summarize_array(a: np.ndarray, prefix: str) -> dict:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_median": np.nan, f"{prefix}_p95": np.nan, f"{prefix}_max": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(a)),
        f"{prefix}_median": float(np.median(a)),
        f"{prefix}_p95": float(np.quantile(a, 0.95)),
        f"{prefix}_max": float(np.max(a)),
    }

def univariate_ks_w1(Xr: np.ndarray, Xg: np.ndarray, cfgm) -> dict:
    d = Xr.shape[1]
    max_dim = cfgm.get("univar_max_dim", None)
    if max_dim is not None:
        d = min(d, int(max_dim))
    ks_vals, w1_vals = [], []
    for j in range(d):
        a = Xr[:, j]
        b = Xg[:, j]
        try:
            ks = stats.ks_2samp(a, b, method="auto").statistic
        except Exception:
            ks = np.nan
        ks_vals.append(ks)
        try:
            w1 = stats.wasserstein_distance(a, b)
        except Exception:
            w1 = np.nan
        w1_vals.append(w1)
    out = {}
    out.update(summarize_array(np.array(ks_vals), "KS"))
    out.update(summarize_array(np.array(w1_vals), "W1"))
    return out

def moment_diffs(Xr: np.ndarray, Xg: np.ndarray) -> dict:
    Xr = np.nan_to_num(np.asarray(Xr, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Xg = np.nan_to_num(np.asarray(Xg, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def moments(X):
        mu = X.mean(axis=0)
        xc = X - mu[None, :]
        var = np.mean(xc * xc, axis=0)
        sd = np.sqrt(var) + 1e-12
        m3 = np.mean(xc ** 3, axis=0)
        m4 = np.mean(xc ** 4, axis=0)
        skew = m3 / (sd ** 3)
        kurt_fisher = m4 / (sd ** 4) - 3.0
        return mu, sd, skew, kurt_fisher

    mu_r, sd_r, sk_r, ku_r = moments(Xr)
    mu_g, sd_g, sk_g, ku_g = moments(Xg)

    return {
        "ΔMean_L1": float(np.mean(np.abs(mu_r - mu_g))),
        "ΔStd_L1":  float(np.mean(np.abs(sd_r - sd_g))),
        "ΔSkew_L1": float(np.mean(np.abs(sk_r - sk_g))),
        "ΔKurt_L1": float(np.mean(np.abs(ku_r - ku_g))),
    }

def cov_spectrum_metrics(Xr: np.ndarray, Xg: np.ndarray, eps: float = 1e-8) -> dict:
    Cr = _cov(Xr.astype(np.float64)) + eps * np.eye(Xr.shape[1])
    Cg = _cov(Xg.astype(np.float64)) + eps * np.eye(Xg.shape[1])

    er = np.linalg.eigvalsh(Cr)
    eg = np.linalg.eigvalsh(Cg)
    er = np.clip(er, eps, None)
    eg = np.clip(eg, eps, None)

    lr = np.log(er)
    lg = np.log(eg)

    cond_r = float(er.max() / (er.min() + 1e-12))
    cond_g = float(eg.max() / (eg.min() + 1e-12))

    return {
        "CovSpec_L1": float(np.mean(np.abs(lr - lg))),
        "CovSpec_L2": float(np.sqrt(np.mean((lr - lg) ** 2))),
        "CovTrace_Ratio": float(np.trace(Cg) / (np.trace(Cr) + 1e-12)),
        "CovCond_Ratio": float(cond_g / (cond_r + 1e-12)),
    }

def pairwise_distance_distribution(Xr: np.ndarray, Xg: np.ndarray, cfgm) -> dict:
    rng = np.random.RandomState(int(cfgm.get("pairdist_seed", 1234)))
    n = int(cfgm.get("pairdist_n", 4000))
    n_pairs = int(cfgm.get("pairdist_pairs", 30000))
    n = min(n, len(Xr), len(Xg))
    if n < 50:
        return {"PairDist_KS": np.nan, "PairDist_W1": np.nan, "PairDist_MeanRatio": np.nan, "PairDist_StdRatio": np.nan}

    Ar = subsample(Xr, n, seed=7)
    Ag = subsample(Xg, n, seed=8)

    i1 = rng.randint(0, n, size=n_pairs)
    i2 = rng.randint(0, n, size=n_pairs)
    mask = (i1 != i2)
    i1, i2 = i1[mask], i2[mask]
    if i1.size == 0:
        return {"PairDist_KS": np.nan, "PairDist_W1": np.nan, "PairDist_MeanRatio": np.nan, "PairDist_StdRatio": np.nan}

    dr = np.linalg.norm(Ar[i1] - Ar[i2], axis=1)
    dg = np.linalg.norm(Ag[i1] - Ag[i2], axis=1)

    try:
        ks = stats.ks_2samp(dr, dg, method="auto").statistic
    except Exception:
        ks = np.nan
    try:
        w1 = stats.wasserstein_distance(dr, dg)
    except Exception:
        w1 = np.nan

    return {
        "PairDist_KS": float(ks),
        "PairDist_W1": float(w1),
        "PairDist_MeanRatio": float(np.mean(dg) / (np.mean(dr) + 1e-12)),
        "PairDist_StdRatio": float(np.std(dg) / (np.std(dr) + 1e-12)),
    }

def energy_distance_multivariate(Xr: np.ndarray, Xg: np.ndarray, cfgm) -> float:
    n = int(cfgm.get("energy_n", 2000))
    chunk = int(cfgm.get("energy_chunk", 400))
    n = min(n, len(Xr), len(Xg))
    if n < 50:
        return float("nan")

    A = subsample(Xr, n, seed=101).astype(np.float64)
    B = subsample(Xg, n, seed=102).astype(np.float64)

    def mean_cdist(P, Q, same: bool = False):
        s = 0.0
        cnt = 0
        for i in range(0, len(P), chunk):
            Pi = P[i:i + chunk]
            D = cdist(Pi, Q, metric="euclidean")
            if same:
                start = i
                for r in range(len(Pi)):
                    j = start + r
                    if j < len(Q):
                        D[r, j] = np.nan
            s += float(np.nansum(D))
            cnt += int(np.sum(np.isfinite(D)))
        return s / max(1, cnt)

    exy = mean_cdist(A, B, same=False)
    exx = mean_cdist(A, A, same=True)
    eyy = mean_cdist(B, B, same=True)
    return float(exy - 0.5 * exx - 0.5 * eyy)

def rff_mmd_permutation_pvalue(Xr: np.ndarray, Xg: np.ndarray, gamma: float, cfgm) -> float:
    if not cfgm.get("mmd_perm_enable", True):
        return float("nan")
    rng = np.random.RandomState(int(cfgm.get("mmd_perm_seed", 123)))
    n_sub = int(cfgm.get("mmd_perm_subset", 4000))
    trials = int(cfgm.get("mmd_perm_trials", 200))
    rff_dim = int(cfgm.get("rff_dim", 2048))
    seed = int(cfgm.get("rff_seed", 42))

    Xr2 = subsample(Xr, min(len(Xr), n_sub), seed=201)
    Xg2 = subsample(Xg, min(len(Xg), n_sub), seed=202)
    n = min(len(Xr2), len(Xg2))
    if n < 200:
        return float("nan")
    Xr2 = Xr2[:n]
    Xg2 = Xg2[:n]

    Phir = rff_features(Xr2, gamma=gamma, rff_dim=rff_dim, seed=seed)
    Phig = rff_features(Xg2, gamma=gamma, rff_dim=rff_dim, seed=seed)

    def mmd2_from_phi(A, B):
        return float(np.sum((A.mean(axis=0) - B.mean(axis=0)) ** 2))

    obs = mmd2_from_phi(Phir, Phig)
    pool = np.vstack([Phir, Phig])

    count = 0
    for _ in range(trials):
        perm = rng.permutation(pool.shape[0])
        A = pool[perm[:n]]
        B = pool[perm[n:2 * n]]
        t = mmd2_from_phi(A, B)
        if t >= obs - 1e-12:
            count += 1
    return float((count + 1) / (trials + 1))

def evaluate_one_method(X_real_std: np.ndarray, X_gen_std: np.ndarray, idxT, idxS, idxP, cfgm) -> dict:
    Xr = subsample(X_real_std, int(cfgm["max_real"]), seed=int(cfgm.get("rff_seed", 42)))
    Xg = subsample(X_gen_std,  int(cfgm["max_gen"]),  seed=int(cfgm.get("rff_seed", 42)) + 1)

    Xr = np.nan_to_num(Xr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xg = np.nan_to_num(Xg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    gamma = cfgm["mmd_gamma"]
    if gamma is None:
        gamma = median_heuristic_gamma(Xr, seed=int(cfgm.get("rff_seed", 42)))

    cons = infer_constraints_from_real(Xr, cfgm)

    out = {}
    out["FFD"] = frechet_distance(
        Xr.astype(np.float64),
        Xg.astype(np.float64),
        eps=float(cfgm["ffd_eps"]),
        diag=bool(cfgm.get("ffd_diag", True)),
        clip_neg_tr=bool(cfgm.get("ffd_clip_negative_trace", True)),
    )
    out["RFF-MMD"] = rff_mmd(Xr, Xg, gamma=float(gamma), rff_dim=int(cfgm["rff_dim"]), seed=int(cfgm["rff_seed"]))
    out[f"Coverage@{cfgm['knn_k']}"] = coverage_at_k(Xr, Xg, k=int(cfgm["knn_k"]), metric=str(cfgm["nn_metric"]))

    prec, rec = knn_precision_recall(Xr, Xg, k=int(cfgm["knn_k"]), metric=str(cfgm["nn_metric"]))
    out["kNN-Prec"] = prec
    out["kNN-Rec"] = rec

    auc, acc = c2st_metrics(Xr, Xg, seed=int(cfgm["c2st_seed"]), test_size=float(cfgm["c2st_test_size"]), max_iter=int(cfgm["c2st_max_iter"]))
    out["C2ST-AUC"] = auc
    out["C2ST-Acc"] = acc

    out["SWD"] = sliced_wasserstein_distance(Xr, Xg, n_proj=int(cfgm["swd_projections"]), seed=int(cfgm["swd_seed"]))
    out.update(corr_distance(Xr, Xg, idxT, idxS, idxP))

    out["Violation"] = violation_rate(Xg, cons, cfgm)
    out["NN-Overfit"] = nn_overfit_ratio(Xr, Xg, metric=str(cfgm["nn_metric"]))

    out.update(univariate_ks_w1(Xr, Xg, cfgm))
    out.update(moment_diffs(Xr, Xg))
    out.update(cov_spectrum_metrics(Xr, Xg))
    out.update(pairwise_distance_distribution(Xr, Xg, cfgm))
    out["EnergyDist"] = energy_distance_multivariate(Xr, Xg, cfgm)
    out["RFF-MMD_p"] = rff_mmd_permutation_pvalue(Xr, Xg, gamma=float(gamma), cfgm=cfgm)

    return out


# ============================================================
# Optional baseline manifest discovery
# ============================================================
def find_manifest_path(artifact_dir: str, export_subdir: str, export_prefix: str) -> Optional[str]:
    out_dir = os.path.join(artifact_dir, export_subdir)
    if not os.path.isdir(out_dir):
        return None

    expected = os.path.join(out_dir, f"{export_prefix}manifest.json")
    if export_prefix is not None and export_prefix != "" and os.path.exists(expected):
        return expected

    cands = sorted(glob.glob(os.path.join(out_dir, "*manifest.json")))
    if len(cands) == 0:
        return None
    if len(cands) == 1:
        print(f"[Manifest] Auto-picked unique manifest: {cands[0]}")
        return cands[0]

    if export_prefix:
        hits = [p for p in cands if os.path.basename(p).startswith(export_prefix)]
        if len(hits) == 1:
            print(f"[Manifest] Auto-picked prefix-matched manifest: {hits[0]}")
            return hits[0]

    print("[Manifest][WARN] Multiple manifests found; set export_prefix more specifically.")
    return None

def load_manifest_sources(manifest_path: str) -> dict:
    if not manifest_path or (not os.path.exists(manifest_path)):
        return {}
    m = load_json(manifest_path)
    paths = m.get("paths", {})
    if not isinstance(paths, dict) or len(paths) == 0:
        return {}

    base_dir = os.path.dirname(os.path.abspath(manifest_path))
    out = {}
    for k, v in paths.items():
        if not isinstance(v, str):
            continue
        p = v.replace("/", os.sep).replace("\\", os.sep).strip()
        if os.path.isabs(p) and os.path.exists(p):
            out[str(k)] = p
        else:
            cand = os.path.normpath(os.path.join(base_dir, p))
            out[str(k)] = os.path.abspath(cand)
    return out


# ============================================================
# Multi-seed helpers
# ============================================================
def resolve_ablation_manifest_from_run_root(run_root: str) -> str:
    # Training script writes ablation_manifest.json directly under run_root
    cand = os.path.join(run_root, "ablation_manifest.json")
    if os.path.exists(cand):
        return os.path.abspath(cand)
    # fallback: sometimes user points to seed root, pick the latest timestamp folder
    # e.g., results_nb15_rdsynth_ablation_seed42/<ts>/ablation_manifest.json
    cands = sorted(glob.glob(os.path.join(run_root, "*", "ablation_manifest.json")))
    if len(cands) > 0:
        return os.path.abspath(cands[-1])
    raise FileNotFoundError(f"Cannot find ablation_manifest.json under: {run_root}")

def load_all_seeds_runs(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"all_seeds_runs.json not found: {path}")
    d = load_json(path)
    runs = d.get("runs", None)
    if not isinstance(runs, list) or len(runs) == 0:
        raise ValueError("all_seeds_runs.json has no valid 'runs' list")
    out = []
    for r in runs:
        seed = r.get("seed", None)
        run_root = r.get("run_root", None)
        if seed is None or run_root is None:
            continue
        out.append({"seed": int(seed), "run_root": str(run_root)})
    if len(out) == 0:
        raise ValueError("No usable {seed, run_root} entries in all_seeds_runs.json")
    return out


# ============================================================
# Core runner for one seed
# ============================================================
def run_one_seed_metrics(
    seed: int,
    ablation_manifest_path: str,
    X_real: np.ndarray,
    feat_cols: List[str],
    std: Standardizer,
    idxT, idxS, idxP,
    baseline_sources_global: Optional[Dict[str, str]],
    cfgm: dict,
    out_dir: str,
) -> pd.DataFrame:
    print("\n" + "=" * 110)
    print(f"[Seed {seed}] Using manifest: {ablation_manifest_path}")
    print("=" * 110)

    ab = load_json(ablation_manifest_path)
    ab_paths = ab.get("paths", {})
    if not isinstance(ab_paths, dict) or len(ab_paths) == 0:
        raise ValueError(f"[Seed {seed}] ablation_manifest.json has no valid 'paths'")

    # build gen_sources: optional baseline + ablation
    gen_sources: Dict[str, str] = {}
    if baseline_sources_global:
        gen_sources.update(baseline_sources_global)
    for k, v in ab_paths.items():
        gen_sources[str(k)] = str(v)

    rows = []
    for method, path in gen_sources.items():
        print(f"\n[Seed {seed}][{method}] Loading gen -> std: {path}")
        X_gen, std_diff = load_gen_as_std(path, std=std, method=method, meta_cols=feat_cols, cfgm=cfgm)
        if X_gen.shape[1] != X_real.shape[1]:
            raise ValueError(f"[Seed {seed}][{method}] dim mismatch: gen={X_gen.shape}, real={X_real.shape}")

        print(f"[Seed {seed}][{method}] X_gen_std={X_gen.shape}")
        if cfgm.get("debug_print_stats", True):
            quick_stats(f"Seed{seed}_{method}_std", X_gen)

        m = evaluate_one_method(X_real, X_gen, idxT, idxS, idxP, cfgm)
        m["Method"] = method
        m["Seed"] = int(seed)
        if bool(cfgm.get("record_std_mismatch", True)):
            m["StdMismatch"] = float(std_diff)
        rows.append(m)

    df = pd.DataFrame(rows)

    # nice column ordering
    head = [
        "Seed", "Method",
        "StdMismatch",
        "FFD", "RFF-MMD", "RFF-MMD_p",
        f"Coverage@{cfgm['knn_k']}", "kNN-Prec", "kNN-Rec",
        "C2ST-AUC", "C2ST-Acc",
        "SWD", "EnergyDist",
        "CorrΔ_all", "CorrΔ_TS", "CorrΔ_TP", "CorrΔ_SP",
        "Violation", "NN-Overfit",
        "KS_mean", "KS_median", "KS_p95", "KS_max",
        "W1_mean", "W1_median", "W1_p95", "W1_max",
        "ΔMean_L1", "ΔStd_L1", "ΔSkew_L1", "ΔKurt_L1",
        "CovSpec_L1", "CovSpec_L2", "CovTrace_Ratio", "CovCond_Ratio",
        "PairDist_KS", "PairDist_W1", "PairDist_MeanRatio", "PairDist_StdRatio",
    ]
    cols_out = [c for c in head if c in df.columns] + [c for c in df.columns if c not in head]
    df = df[cols_out]

    # save per-seed
    csv_out = os.path.join(out_dir, f"metrics_ablation_table_seed{seed}.csv")
    df.to_csv(csv_out, index=False)

    print(f"\n=== Metrics Table (Seed {seed}; shared standardizer space) ===")
    pd.set_option("display.max_columns", 300)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False))

    if cfgm.get("save_txt", True):
        txt_out = os.path.join(out_dir, f"metrics_ablation_table_seed{seed}.txt")
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
            f.write("\n")
        print(f"\n[Saved] {csv_out}\n[Saved] {txt_out}")
    else:
        print(f"\n[Saved] {csv_out}")

    return df


# ============================================================
# Main
# ============================================================
def main():
    cfgm = CFG["metrics"]
    out_dir = ensure_dir(cfgm["out_dir"])

    # canonical meta + std
    meta_path = os.path.join(CFG["baseline_artifact_dir"], "meta.json")
    std_path = os.path.join(CFG["baseline_artifact_dir"], "standardizer.npz")
    if not os.path.exists(meta_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing meta.json or standardizer.npz under {CFG['baseline_artifact_dir']}")

    meta = load_json(meta_path)
    meta_cols = meta.get("cols", None)
    if not meta_cols:
        raise ValueError("baseline meta.json has no cols")

    feat_cols = norm_cols(meta_cols)
    print(f"[ColsCanon] D={len(feat_cols)} from baseline meta.json")

    std = Standardizer.load(std_path)
    if len(std.mu) != len(feat_cols):
        raise ValueError("Dim mismatch between standardizer and meta.cols")

    # real benign (once)
    Xb_raw = load_benign_chunked_raw_canonical(
        csv_path=CFG["csv_path"],
        label_col=CFG["label_col"],
        feat_cols=feat_cols,
        n_ben=int(cfgm["n_ben"]),
        benign_value=int(CFG["benign_value"]),
        seed=int(cfgm.get("rff_seed", 42)),
        chunksize=int(CFG["chunksize"]),
    )
    X_real = std.transform(Xb_raw).astype(np.float32)
    idxT, idxS, idxP = split_feature_blocks_unsw(feat_cols)
    print(f"[Real] X_real_std={X_real.shape}")
    if cfgm.get("debug_print_stats", True):
        quick_stats("REAL_std", X_real)

    # optional baseline sources (load once)
    baseline_sources_global = None
    if bool(CFG.get("include_baseline_manifest", False)):
        mpath = find_manifest_path(
            artifact_dir=CFG["baseline_artifact_dir"],
            export_subdir=CFG.get("baseline_export_subdir", "adv_npz"),
            export_prefix=CFG.get("baseline_export_prefix", ""),
        )
        if mpath:
            print(f"[BaselineManifest] Using: {mpath}")
            baseline_sources_global = load_manifest_sources(mpath)
        else:
            print("[BaselineManifest][WARN] Not found; skipping baseline rows.")
            baseline_sources_global = None

    # multi-seed runs
    runs = load_all_seeds_runs(CFG["all_seeds_runs_path"])
    print("\n" + "=" * 110)
    print(f"[MultiSeed] Loaded {len(runs)} runs from: {CFG['all_seeds_runs_path']}")
    for r in runs:
        print(f"  - seed={r['seed']} run_root={r['run_root']}")
    print("=" * 110)

    # compute per seed
    dfs = []
    for r in runs:
        seed = int(r["seed"])
        run_root = str(r["run_root"])
        am_path = resolve_ablation_manifest_from_run_root(run_root)
        df_seed = run_one_seed_metrics(
            seed=seed,
            ablation_manifest_path=am_path,
            X_real=X_real,
            feat_cols=feat_cols,
            std=std,
            idxT=idxT, idxS=idxS, idxP=idxP,
            baseline_sources_global=baseline_sources_global,
            cfgm=cfgm,
            out_dir=out_dir,
        )
        dfs.append(df_seed)

    # long table
    df_long = pd.concat(dfs, ignore_index=True)
    long_out = os.path.join(out_dir, "metrics_ablation_all_seeds_long.csv")
    df_long.to_csv(long_out, index=False)
    print(f"\n[Saved] {long_out}")

    # aggregate mean/std across seeds by Method (numeric columns only)
    numeric_cols = [c for c in df_long.columns if c not in ["Method"] and c not in ["Seed"]]
    # keep only truly numeric
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df_long[c])]

    df_mean = df_long.groupby("Method", as_index=False)[numeric_cols].mean(numeric_only=True)
    df_std  = df_long.groupby("Method", as_index=False)[numeric_cols].std(numeric_only=True)

    mean_out = os.path.join(out_dir, "metrics_ablation_mean.csv")
    std_out  = os.path.join(out_dir, "metrics_ablation_std.csv")
    df_mean.to_csv(mean_out, index=False)
    df_std.to_csv(std_out, index=False)

    # combined mean±std table (wide)
    df_mean2 = df_mean.set_index("Method")
    df_std2  = df_std.set_index("Method")
    # build
    cols = list(df_mean2.columns)
    combined = pd.DataFrame(index=df_mean2.index)
    for c in cols:
        combined[f"{c}_mean"] = df_mean2[c]
        combined[f"{c}_std"]  = df_std2[c]
    combined = combined.reset_index()

    combined_out = os.path.join(out_dir, "metrics_ablation_mean_std.csv")
    combined.to_csv(combined_out, index=False)

    print("\n" + "=" * 110)
    print("=== Metrics Table (Mean±Std across seeds; shared standardizer space) ===")
    pd.set_option("display.max_columns", 400)
    pd.set_option("display.width", 240)
    print(combined.to_string(index=False))
    print(f"\n[Saved] {mean_out}\n[Saved] {std_out}\n[Saved] {combined_out}")
    print("=" * 110)


if __name__ == "__main__":
    main()
