import os
import glob
import json
import warnings
warnings.filterwarnings("ignore")

from typing import Any, Optional, Tuple, List, Dict

import numpy as np
import pandas as pd


CFG = {
    "csv_path": "../data/unsw/CICFlowMeter_preprocessed.csv",
    "label_col": "Label",
    "benign_value": 0,
    "chunksize": 200000,
    "n_ben": 150000,

    "baseline": {
        "artifact_dir": "artifacts/baselines_light_nb15_v2",
        "export_subdir": "adv_npz",
        "export_prefix": "NB15_",
    },

    "suite": {
        "max_eval": 50000,
        "seed": 42,

        "int_tol": 1e-3,

        "port_lo": 0.0,
        "port_hi": 65535.0,
        "protocol_lo": 0.0,
        "protocol_hi": 255.0,
        "proto_set_max_size": 16,

        "order_tol": 1e-6,

        # threshold lists (no weighting)
        "fatal_k_list": [0, 1],   # ValidFatal@K (ALL fatal)
        "otherfatal_k_list": [0, 1],  # ValidOtherFatal@K (fatal excluding PortInvalid)
        "cons_r_list": [0, 1],    # ConsValid@R

        # diagnostics (not part of fatal validity)
        "neg_tol": -1e-6,
        "report_nonint_fields": True,
        "include_benign_row": True,

        "out_dir": "results/stp_compliance_suite_nb15",
        "save_txt": True,
        "debug_groups": False,
    },

    "extra_gen_sources": {
        "RD-Synth": "results_nb15_rdsynth/2025-12-20_23-07-47/rd_synth_adv.npz",
    },
}


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

def safe_float_array(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float64)

def is_integerlike(x: np.ndarray, tol: float) -> np.ndarray:
    return np.abs(x - np.round(x)) <= tol

def nan_rate(X: np.ndarray) -> float:
    return float(np.mean(np.isnan(X)))

def inf_rate(X: np.ndarray) -> float:
    return float(np.mean(np.isinf(X)))

def any_naninf_mask_sample(X: np.ndarray) -> np.ndarray:
    return np.any(np.isnan(X) | np.isinf(X), axis=1)


class Standardizer:
    def __init__(self):
        self.mu: Optional[np.ndarray] = None
        self.sigma: Optional[np.ndarray] = None

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        assert self.mu is not None and self.sigma is not None
        return Z.astype(np.float32) * self.sigma[None, :] + self.mu[None, :]

    @staticmethod
    def load(path_npz: str) -> "Standardizer":
        d = np.load(path_npz)
        s = Standardizer()
        s.mu = d["mu"].astype(np.float32)
        s.sigma = d["sigma"].astype(np.float32)
        s.sigma = np.where(s.sigma < 1e-6, 1.0, s.sigma).astype(np.float32)
        return s


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
    print(f"[Real] Loading benign (chunked RAW, canonical cols): {csv_path}")
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


def load_manifest_sources(manifest_path: str) -> dict:
    if not manifest_path:
        return {}
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest_path not found: {manifest_path}")

    m = load_json(manifest_path)
    paths = m.get("paths", {})
    if not isinstance(paths, dict) or len(paths) == 0:
        raise ValueError(f"Manifest has no valid 'paths': {manifest_path}")

    manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.abspath(os.path.join(script_dir, ".."))

    def _norm(p: str) -> str:
        p = str(p).replace("/", os.sep).replace("\\", os.sep).strip()
        return os.path.normpath(p)

    out = {}
    for k, v in paths.items():
        if not isinstance(v, str):
            continue
        p = _norm(v)

        if os.path.isabs(p) and os.path.exists(p):
            out[str(k)] = os.path.abspath(p); continue
        if os.path.exists(p):
            out[str(k)] = os.path.abspath(p); continue

        cand1 = os.path.normpath(os.path.join(manifest_dir, p))
        if os.path.exists(cand1):
            out[str(k)] = os.path.abspath(cand1); continue

        cand2 = os.path.normpath(os.path.join(proj_root, p))
        if os.path.exists(cand2):
            out[str(k)] = os.path.abspath(cand2); continue

        guess = os.path.abspath(cand1)
        print(f"[Manifest][WARN] Path not found for '{k}': '{v}'. Using guess: {guess}")
        out[str(k)] = guess
    return out

def find_manifest_path(artifact_dir: str, export_subdir: str, export_prefix: str) -> str:
    out_dir = os.path.join(artifact_dir, export_subdir)
    if not os.path.isdir(out_dir):
        raise FileNotFoundError(f"export_subdir not found: {out_dir}")

    expected = os.path.join(out_dir, f"{export_prefix}manifest.json")
    if export_prefix and os.path.exists(expected):
        return expected

    cands = sorted(glob.glob(os.path.join(out_dir, "*manifest.json")))
    if len(cands) == 0:
        raise FileNotFoundError(f"No '*manifest.json' found in {out_dir}")

    if len(cands) == 1:
        print(f"[Manifest] Auto-picked unique manifest: {cands[0]}")
        return cands[0]

    if export_prefix:
        hits = [p for p in cands if os.path.basename(p).startswith(export_prefix)]
        if len(hits) == 1:
            print(f"[Manifest] Auto-picked prefix-matched manifest: {hits[0]}")
            return hits[0]
        if len(hits) > 1:
            raise FileNotFoundError("Multiple prefix-matched manifests:\n" + "\n".join(hits))
    raise FileNotFoundError("Multiple manifests found:\n" + "\n".join(cands))


def _pick_raw_key(npz_obj) -> Optional[str]:
    for k in ["Xadv_raw", "Xraw", "X_adv_raw"]:
        if k in npz_obj:
            return k
    if "Xadv" in npz_obj:
        return "Xadv"
    return None

def _pick_std_key(npz_obj) -> Optional[str]:
    for k in ["Xadv_std", "Xstd", "Zadv", "Z"]:
        if k in npz_obj:
            return k
    return None

def load_gen_auto_raw(path: str, std: Standardizer, method: str) -> Tuple[np.ndarray, dict]:
    assert os.path.exists(path), f"File not found: {path}"
    if os.path.splitext(path)[1].lower() != ".npz":
        raise ValueError(f"[{method}] Only .npz supported. Got: {path}")
    info = {"LoadSpace": "unknown", "LoadKey": "", "UsedInverse": False, "Path": path}
    d = np.load(path, allow_pickle=True)

    raw_k = _pick_raw_key(d)
    if raw_k is not None:
        X = np.asarray(d[raw_k], dtype=np.float32)
        if X.ndim != 2:
            X = X.reshape((X.shape[0], -1))
        info.update({"LoadSpace": "raw", "LoadKey": raw_k, "UsedInverse": False})
        return X, info

    std_k = _pick_std_key(d)
    if std_k is not None:
        Z = np.asarray(d[std_k], dtype=np.float32)
        if Z.ndim != 2:
            Z = Z.reshape((Z.shape[0], -1))
        X = std.inverse(Z).astype(np.float32)
        info.update({"LoadSpace": "std", "LoadKey": std_k, "UsedInverse": True})
        return X, info

    for k in ["X", "Xgen", "samples", "data", "arr_0"]:
        if k in d:
            X = np.asarray(d[k], dtype=np.float32)
            if X.ndim != 2:
                X = X.reshape((X.shape[0], -1))
            info.update({"LoadSpace": "raw_assumed", "LoadKey": k, "UsedInverse": False})
            return X, info

    raise KeyError(f"[{method}] NPZ has no recognizable array key. keys={list(d.keys())}")


def _colmask(cols: List[str], keywords: List[str]) -> List[int]:
    cols_norm = [c.lower() for c in cols]
    out = []
    for i, c in enumerate(cols_norm):
        if any(k in c for k in keywords):
            out.append(i)
    return out

def find_port_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["port"])

def find_protocol_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["protocol"])

def find_flag_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["flag"])

def find_duration_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["duration", "flow duration"])

def find_iat_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["iat"])

def find_active_idle_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["active", "idle"])

def find_rate_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["/s", "per_s", "rate"])

def find_count_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["packets", "packet", "pkts", "cnt", "count", "total_fwd_packets", "total_backward_packets"])

def find_length_cols(cols: List[str]) -> List[int]:
    return _colmask(cols, ["length", "bytes"])


def build_min_mean_max_groups(cols: List[str], focus_keywords: List[str]) -> List[Tuple[str, int, int, int]]:
    cols_norm = [c.lower() for c in cols]
    mins: Dict[str, int] = {}
    means: Dict[str, int] = {}
    maxs: Dict[str, int] = {}

    def ok_focus(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in focus_keywords)

    def base_name(c: str) -> str:
        x = c.lower()
        x = x.replace(" max", "").replace(" min", "").replace(" mean", "")
        x = x.replace("_max", "").replace("_min", "").replace("_mean", "")
        x = x.replace("maximum", "").replace("minimum", "").replace("average", "")
        return " ".join(x.split())

    for i, c in enumerate(cols_norm):
        if not ok_focus(c):
            continue
        if (" min" in c) or c.endswith("_min") or ("minimum" in c):
            mins[base_name(cols[i])] = i
        if (" mean" in c) or c.endswith("_mean") or ("average" in c):
            means[base_name(cols[i])] = i
        if (" max" in c) or c.endswith("_max") or ("maximum" in c):
            maxs[base_name(cols[i])] = i

    groups = []
    for b in mins.keys():
        if b in means and b in maxs:
            groups.append((b, mins[b], means[b], maxs[b]))
    return groups


def domain_invalid_mask_any(X: np.ndarray, idxs: List[int], lo: float, hi: float) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros((X.shape[0],), dtype=bool)
    A = X[:, idxs].astype(np.float64)
    m = np.isfinite(A) & ((A < lo) | (A > hi))
    return np.any(m, axis=1)

def nonint_mask_any(X: np.ndarray, idxs: List[int], tol: float) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros((X.shape[0],), dtype=bool)
    A = X[:, idxs].astype(np.float64)
    m = np.isfinite(A) & (~is_integerlike(A, tol))
    return np.any(m, axis=1)

def negative_mask_any_tol(X: np.ndarray, idxs: List[int], neg_tol: float) -> np.ndarray:
    if len(idxs) == 0:
        return np.zeros((X.shape[0],), dtype=bool)
    A = X[:, idxs].astype(np.float64)
    m = np.isfinite(A) & (A < neg_tol)
    return np.any(m, axis=1)

def order_violation_count_per_sample(X: np.ndarray, groups: List[Tuple[str, int, int, int]], order_tol: float) -> np.ndarray:
    if len(groups) == 0:
        return np.zeros((X.shape[0],), dtype=np.int32)
    A = X.astype(np.float64)
    cnt = np.zeros((X.shape[0],), dtype=np.int32)
    for _, i_min, i_mean, i_max in groups:
        mn = A[:, i_min]
        mu = A[:, i_mean]
        mx = A[:, i_max]
        ok = np.isfinite(mn) & np.isfinite(mu) & np.isfinite(mx)
        bad = ok & ((mn > (mu + order_tol)) | (mu > (mx + order_tol)))
        cnt += bad.astype(np.int32)
    return cnt

def zero_duration_nonzero_payload_mask(X: np.ndarray, cols: List[str]) -> np.ndarray:
    cols_norm = [c.lower() for c in cols]
    dur_idxs = [i for i, c in enumerate(cols_norm) if "duration" in c]
    if len(dur_idxs) == 0:
        return np.zeros((X.shape[0],), dtype=bool)

    act_idxs = [i for i, c in enumerate(cols_norm)
                if ("total" in c and ("packets" in c or "bytes" in c or "length" in c))
                or ("totlen" in c)
                or ("total_length" in c)]
    if len(act_idxs) == 0:
        return np.zeros((X.shape[0],), dtype=bool)

    A = X.astype(np.float64)
    dur = A[:, dur_idxs[0]]
    act = A[:, act_idxs]
    ok = np.isfinite(dur) & np.all(np.isfinite(act), axis=1)
    bad = ok & (np.abs(dur) <= 1e-12) & (np.any(act > 0.0, axis=1))
    return bad


def infer_benign_protocol_set(Xb_raw: np.ndarray, cols: List[str], proto_idxs: List[int], cfgs: dict) -> Optional[set]:
    if len(proto_idxs) == 0:
        return None
    if int(cfgs.get("proto_set_max_size", 0)) <= 0:
        return None

    tol = float(cfgs["int_tol"])
    i = proto_idxs[0]
    a = Xb_raw[:, i].astype(np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    a_ok = a[is_integerlike(a, tol)]
    a_ok = a_ok[(a_ok >= cfgs["protocol_lo"]) & (a_ok <= cfgs["protocol_hi"])]
    if a_ok.size == 0:
        return None
    return set(np.unique(np.round(a_ok).astype(np.int64)).tolist())

def protocol_hard_invalid_mask_any(X: np.ndarray, proto_idxs: List[int], benign_proto_set: Optional[set], cfgs: dict) -> np.ndarray:
    if len(proto_idxs) == 0:
        return np.zeros((X.shape[0],), dtype=bool)
    A = X[:, proto_idxs].astype(np.float64)
    invalid = np.zeros((X.shape[0],), dtype=bool)

    lo = float(cfgs["protocol_lo"])
    hi = float(cfgs["protocol_hi"])
    tol = float(cfgs["int_tol"])

    m_dom = np.isfinite(A) & ((A < lo) | (A > hi))
    invalid |= np.any(m_dom, axis=1)

    if benign_proto_set is not None and len(benign_proto_set) <= int(cfgs["proto_set_max_size"]):
        ok_int = np.isfinite(A) & is_integerlike(A, tol)
        Ai = np.round(A).astype(np.int64)
        allowed = np.array(sorted(list(benign_proto_set)), dtype=np.int64)
        m_mem = ok_int & (~np.isin(Ai, allowed))
        invalid |= np.any(m_mem, axis=1)

    return invalid


def evaluate_one_method_v4_1(
    X_raw: np.ndarray,
    cols: List[str],
    benign_proto_set: Optional[set],
    cfgs: dict,
    groups_time: list,
    groups_len: list,
) -> dict:
    Xg = subsample(X_raw, int(cfgs["max_eval"]), seed=int(cfgs["seed"]))
    Xg = safe_float_array(Xg)

    out: Dict[str, Any] = {}
    out["N_eval"] = int(Xg.shape[0])
    out["D"] = int(Xg.shape[1])

    # element-wise
    out["NaN_rate"] = nan_rate(Xg)
    out["Inf_rate"] = inf_rate(Xg)
    naninf_any = any_naninf_mask_sample(Xg)
    out["AnyNaNInf_rate"] = float(np.mean(naninf_any))

    port_idxs = find_port_cols(cols)
    proto_idxs = find_protocol_cols(cols)

    # Fatal checks
    port_dom_any = domain_invalid_mask_any(Xg, port_idxs, float(cfgs["port_lo"]), float(cfgs["port_hi"]))
    proto_hard_any = protocol_hard_invalid_mask_any(Xg, proto_idxs, benign_proto_set, cfgs)
    zero_dur_any = zero_duration_nonzero_payload_mask(Xg, cols)

    out["PortInvalid_rate"] = float(np.mean(port_dom_any))
    out["ProtocolHardInvalid_rate"] = float(np.mean(proto_hard_any))
    out["ZeroDurNonZero_rate"] = float(np.mean(zero_dur_any))

    # ALL-fatal count (kept)
    fatal_flags = np.vstack([naninf_any, port_dom_any, proto_hard_any, zero_dur_any]).T.astype(np.int32)
    fatal_cnt = fatal_flags.sum(axis=1)
    out["FatalVioCount_mean"] = float(np.mean(fatal_cnt))
    out["FatalVioCount_p95"] = float(np.percentile(fatal_cnt, 95))
    for k in cfgs.get("fatal_k_list", [0, 1]):
        out[f"ValidFatal@{int(k)}"] = float(np.mean(fatal_cnt <= int(k)))

    # OTHER-fatal count (exclude PortInvalid to avoid redundancy with PortInv)
    otherfatal_flags = np.vstack([naninf_any, proto_hard_any, zero_dur_any]).T.astype(np.int32)
    otherfatal_cnt = otherfatal_flags.sum(axis=1)
    out["OtherFatalVioCount_mean"] = float(np.mean(otherfatal_cnt))
    out["OtherFatalVioCount_p95"] = float(np.percentile(otherfatal_cnt, 95))
    for k in cfgs.get("otherfatal_k_list", [0, 1]):
        out[f"ValidOtherFatal@{int(k)}"] = float(np.mean(otherfatal_cnt <= int(k)))

    # Consistency counts
    order_tol = float(cfgs["order_tol"])
    iat_vio_cnt = order_violation_count_per_sample(Xg, groups_time, order_tol=order_tol)
    len_vio_cnt = order_violation_count_per_sample(Xg, groups_len, order_tol=order_tol)

    out["IATOrderVio_rate"] = float(np.mean(iat_vio_cnt > 0))
    out["LenOrderVio_rate"] = float(np.mean(len_vio_cnt > 0))

    cons_cnt = iat_vio_cnt + len_vio_cnt
    out["ConsVioCount_mean"] = float(np.mean(cons_cnt))
    out["ConsVioCount_p95"] = float(np.percentile(cons_cnt, 95))
    for r in cfgs.get("cons_r_list", [0, 1]):
        out[f"ConsValid@{int(r)}"] = float(np.mean(cons_cnt <= int(r)))

    out["ValidBoth@Fatal1_Cons1"] = float(np.mean((fatal_cnt <= 1) & (cons_cnt <= 1)))

    # ---- diagnostics (NOT fatal)
    neg_tol = float(cfgs["neg_tol"])
    flag_idxs = find_flag_cols(cols)
    time_idxs = sorted(set(find_duration_cols(cols) + find_iat_cols(cols) + find_active_idle_cols(cols)))
    rate_idxs = find_rate_cols(cols)
    count_idxs = find_count_cols(cols)
    len_idxs = find_length_cols(cols)

    out["FlagNegative_rate"] = float(np.mean(negative_mask_any_tol(Xg, flag_idxs, neg_tol)))
    out["TimeNegative_rate"] = float(np.mean(negative_mask_any_tol(Xg, time_idxs, neg_tol)))
    out["RateNegative_rate"] = float(np.mean(negative_mask_any_tol(Xg, rate_idxs, neg_tol)))
    out["CountNegative_rate"] = float(np.mean(negative_mask_any_tol(Xg, count_idxs, neg_tol)))
    out["LengthNegative_rate"] = float(np.mean(negative_mask_any_tol(Xg, len_idxs, neg_tol)))

    if bool(cfgs.get("report_nonint_fields", True)):
        out["PortNonInt_rate"] = float(np.mean(nonint_mask_any(Xg, port_idxs, tol=float(cfgs["int_tol"]))))
        out["ProtocolNonInt_rate"] = float(np.mean(nonint_mask_any(Xg, proto_idxs, tol=float(cfgs["int_tol"]))))
        out["FlagNonInt_rate"] = float(np.mean(nonint_mask_any(Xg, flag_idxs, tol=float(cfgs["int_tol"]))))
    else:
        out["PortNonInt_rate"] = np.nan
        out["ProtocolNonInt_rate"] = np.nan
        out["FlagNonInt_rate"] = np.nan

    return out


def run_suite(cfg_all: dict, manifest_path: str, extra_gen_sources: Optional[dict] = None) -> None:
    cfgs = cfg_all["suite"]
    out_dir = ensure_dir(cfgs["out_dir"])

    artifact_dir = cfg_all["baseline"]["artifact_dir"]
    meta_path = os.path.join(artifact_dir, "meta.json")
    std_path = os.path.join(artifact_dir, "standardizer.npz")
    if not os.path.exists(meta_path) or not os.path.exists(std_path):
        raise FileNotFoundError(f"Missing baseline meta.json or standardizer.npz:\nmeta={meta_path}\nstd={std_path}")

    meta = load_json(meta_path)
    feat_cols = norm_cols(meta.get("cols", []))
    if not feat_cols:
        raise ValueError("baseline meta.json has no cols")

    std = Standardizer.load(std_path)
    if len(std.mu) != len(feat_cols):
        raise ValueError(f"[Std] Dim mismatch: len(mu)={len(std.mu)} vs cols={len(feat_cols)}. Wrong artifact_dir?")

    Xb_raw = load_benign_chunked_raw_canonical(
        csv_path=cfg_all["csv_path"],
        label_col=cfg_all["label_col"],
        feat_cols=feat_cols,
        n_ben=int(cfg_all["n_ben"]),
        benign_value=int(cfg_all["benign_value"]),
        seed=int(cfgs["seed"]),
        chunksize=int(cfg_all["chunksize"]),
    )

    groups_time = build_min_mean_max_groups(feat_cols, focus_keywords=["iat", "active", "idle"])
    groups_len = build_min_mean_max_groups(feat_cols, focus_keywords=["length", "bytes", "packet length"])

    proto_idxs = find_protocol_cols(feat_cols)
    benign_proto_set = infer_benign_protocol_set(Xb_raw, feat_cols, proto_idxs, cfgs)

    gen_sources = {}
    if manifest_path:
        gen_sources.update(load_manifest_sources(manifest_path))
    if extra_gen_sources:
        gen_sources.update(extra_gen_sources)
    if len(gen_sources) == 0:
        raise RuntimeError("No generated sources found.")

    rows = []

    if bool(cfgs.get("include_benign_row", True)):
        print("\n[Benign(Self-check)] evaluating...")
        m = evaluate_one_method_v4_1(Xb_raw, feat_cols, benign_proto_set, cfgs, groups_time, groups_len)
        m["Method"] = "Benign(Self-check)"
        m["LoadSpace"] = "raw_csv"
        m["LoadKey"] = ""
        m["UsedInverse"] = False
        rows.append(m)

    for method, path in gen_sources.items():
        print(f"\n[{method}] Loading: {path}")
        Xg_raw, info = load_gen_auto_raw(path, std=std, method=method)
        if Xg_raw.shape[1] != len(feat_cols):
            raise ValueError(f"[{method}] dim mismatch: Xg_raw={Xg_raw.shape} vs cols={len(feat_cols)}")

        m = evaluate_one_method_v4_1(Xg_raw, feat_cols, benign_proto_set, cfgs, groups_time, groups_len)
        m["Method"] = method
        m["LoadSpace"] = info.get("LoadSpace", "")
        m["LoadKey"] = info.get("LoadKey", "")
        m["UsedInverse"] = bool(info.get("UsedInverse", False))
        rows.append(m)

    df = pd.DataFrame(rows)

    cols_main = [
        "Method", "N_eval", "D",

        # ALL fatal (kept, but may be redundant with PortInv when others ~0)
        "ValidFatal@0", "ValidFatal@1",
        "FatalVioCount_mean", "FatalVioCount_p95",

        # OTHER fatal (recommended for paper main table)
        "ValidOtherFatal@0", "ValidOtherFatal@1",
        "OtherFatalVioCount_mean", "OtherFatalVioCount_p95",

        # Consistency
        "ConsValid@0", "ConsValid@1",
        "ConsVioCount_mean", "ConsVioCount_p95",
        "ValidBoth@Fatal1_Cons1",

        # fatal components
        "AnyNaNInf_rate", "PortInvalid_rate", "ProtocolHardInvalid_rate", "ZeroDurNonZero_rate",
        "IATOrderVio_rate", "LenOrderVio_rate",

        # diagnostics
        "FlagNegative_rate", "TimeNegative_rate", "RateNegative_rate",
        "CountNegative_rate", "LengthNegative_rate",
        "PortNonInt_rate", "ProtocolNonInt_rate", "FlagNonInt_rate",
        "NaN_rate", "Inf_rate",
        "LoadSpace", "LoadKey", "UsedInverse",
    ]
    cols_out = [c for c in cols_main if c in df.columns] + [c for c in df.columns if c not in cols_main]
    df = df[cols_out]

    # Sort: higher validity first, then fewer violations
    if "Benign(Self-check)" in df["Method"].values:
        df_b = df[df["Method"] == "Benign(Self-check)"].copy()
        df_o = df[df["Method"] != "Benign(Self-check)"].copy()
        df_o = df_o.sort_values(
            by=["ValidFatal@1", "ValidOtherFatal@1", "ConsValid@1", "FatalVioCount_mean", "OtherFatalVioCount_mean", "ConsVioCount_mean"],
            ascending=[False, False, False, True, True, True]
        )
        df = pd.concat([df_b, df_o], ignore_index=True)
    else:
        df = df.sort_values(
            by=["ValidFatal@1", "ValidOtherFatal@1", "ConsValid@1", "FatalVioCount_mean", "OtherFatalVioCount_mean", "ConsVioCount_mean"],
            ascending=[False, False, False, True, True, True]
        ).reset_index(drop=True)

    csv_out = os.path.join(out_dir, "stp_validity_fatal_v4_1.csv")
    df.to_csv(csv_out, index=False)

    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 260)
    print("\n=== STP Validity (fatal + consistency) v4.1 ===")
    print(df.to_string(index=False))

    if cfgs.get("save_txt", True):
        txt_out = os.path.join(out_dir, "stp_validity_fatal_v4_1.txt")
        with open(txt_out, "w", encoding="utf-8") as f:
            f.write(df.to_string(index=False))
            f.write("\n")
        print(f"\n[Saved] {csv_out}\n[Saved] {txt_out}")
    else:
        print(f"\n[Saved] {csv_out}")


def main():
    if not os.path.exists(CFG["csv_path"]):
        raise FileNotFoundError(f"CSV not found: {CFG['csv_path']}")

    artifact_dir = CFG["baseline"]["artifact_dir"]
    export_subdir = CFG["baseline"].get("export_subdir", "adv_npz")
    export_prefix = CFG["baseline"].get("export_prefix", "")

    manifest_path = find_manifest_path(
        artifact_dir=artifact_dir,
        export_subdir=export_subdir,
        export_prefix=export_prefix
    )
    print(f"[Manifest] Using: {manifest_path}")

    run_suite(CFG, manifest_path=manifest_path, extra_gen_sources=CFG.get("extra_gen_sources", None))


if __name__ == "__main__":
    main()
