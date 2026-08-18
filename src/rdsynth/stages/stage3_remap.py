from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rdsynth.models.mlp import MLP
from rdsynth.stages.stage3_targets import (
    CONTINUOUS_MOD_NAMES,
    MOD_NAMES,
    PORT_MOD_NAMES,
    build_remap_targets,
    build_rule_based_modifications,
    clip_modifications,
)
from rdsynth.stages.stage3_targets import (
    build_port_vocab as _build_port_vocab,
)
from rdsynth.stages.stage3_targets import (
    decode_ports as _decode_ports,
)
from rdsynth.stages.stage3_targets import (
    encode_ports as _encode_ports,
)


@dataclass
class RemapBundle:
    remapper: nn.Module
    feature_names: List[str]
    mod_names: List[str]
    mod_mean: np.ndarray
    mod_std: np.ndarray
    input_mean: np.ndarray
    input_std: np.ndarray
    continuous_names: List[str]
    port_values: np.ndarray
    src_port_values: np.ndarray | None = None
    train_log: List[Dict[str, float]] | None = None
    best_epoch: int | None = None
    best_score: float | None = None
    # OOD detection: Mahalanobis parameters from benign training distribution
    ood_maha_mean: np.ndarray | None = None
    ood_maha_inv_cov: np.ndarray | None = None
    # Information bottleneck: effective remapping degrees of freedom
    remap_effective_dim: int | None = None
    remap_subspace_explained_var: float | None = None


class RemapHead(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dims: List[int], continuous_dim: int, dst_port_classes: int, src_port_classes: int = 0
    ):
        super().__init__()
        self.backbone = MLP(in_dim, hidden_dims, hidden_dims[-1] if hidden_dims else in_dim)
        if not hasattr(self.backbone, "output") or not hasattr(self.backbone, "feature_net"):
            raise TypeError(
                f"RemapHead requires an MLP backbone with 'output' and 'feature_net' attributes. Got {type(self.backbone)}"
            )
        feature_dim = self.backbone.output.out_features
        self.continuous_head = nn.Linear(feature_dim, continuous_dim)
        self.dst_port_head = nn.Linear(feature_dim, dst_port_classes)
        self.has_src_port = src_port_classes > 0
        if self.has_src_port:
            self.src_port_head = nn.Linear(feature_dim, src_port_classes)

    def forward(self, x: torch.Tensor):
        feats = self.backbone.feature_net(x)
        cont = self.continuous_head(feats)
        dst_port = self.dst_port_head(feats)
        src_port = self.src_port_head(feats) if self.has_src_port else None
        return cont, dst_port, src_port


def _compute_r2_from_sse(sum_squared_error: float, target_var: float) -> float:
    return float(1.0 - (float(sum_squared_error) / (float(target_var) + 1.0e-12)))


def train_remapper(
    x_ben_norm: np.ndarray,
    x_ben_raw: np.ndarray,
    feature_names: List[str],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    loss: str = "huber",
    huber_beta: float = 1.0,
    grad_clip: float = 0.0,
    target_clip_sigma: float = 0.0,
    weight_decay: float = 0.0,
    train_objective: str = "identity",
    x_mal_norm: np.ndarray | None = None,
    x_mal_raw: np.ndarray | None = None,
) -> RemapBundle:
    use_cuda_loader = device.type == "cuda"

    # ── Select training data based on objective ──────────────────────
    if train_objective == "projection":
        if x_mal_norm is None or x_mal_raw is None:
            raise ValueError("train_objective='projection' requires x_mal_norm and x_mal_raw")
        x_in = np.asarray(x_mal_norm, dtype=np.float32)
        y = build_rule_based_modifications(x_adv_raw=x_mal_raw, x_ben_raw=x_ben_raw, feature_names=feature_names)
    else:
        x_in = np.asarray(x_ben_norm, dtype=np.float32)
        y = build_remap_targets(x_ben_raw, feature_names)

    input_mean = np.mean(x_in, axis=0).astype(np.float32)
    input_std = (np.std(x_in, axis=0) + 1.0e-6).astype(np.float32)
    x_in = ((x_in - input_mean) / input_std).astype(np.float32)

    continuous_idx = [MOD_NAMES.index(name) for name in CONTINUOUS_MOD_NAMES]
    dst_port_idx = MOD_NAMES.index(PORT_MOD_NAMES[0])
    src_port_idx = MOD_NAMES.index(PORT_MOD_NAMES[1])
    y_cont = y[:, continuous_idx]
    y_dst_port = y[:, dst_port_idx]
    y_src_port = y[:, src_port_idx]
    mod_mean = np.mean(y_cont, axis=0)
    mod_std = np.std(y_cont, axis=0) + 1.0e-6
    y_scaled = (y_cont - mod_mean) / mod_std
    if target_clip_sigma and target_clip_sigma > 0.0:
        y_scaled = np.clip(y_scaled, -target_clip_sigma, target_clip_sigma)
    dst_port_values = _build_port_vocab(y_dst_port)
    src_port_values = _build_port_vocab(y_src_port, max_classes=64)
    y_dst_port_cls = _encode_ports(y_dst_port, dst_port_values)
    y_src_port_cls = _encode_ports(y_src_port, src_port_values)

    ds = TensorDataset(
        torch.tensor(x_in, dtype=torch.float32),
        torch.tensor(y_scaled, dtype=torch.float32),
        torch.tensor(y_dst_port_cls, dtype=torch.long),
        torch.tensor(y_src_port_cls, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=use_cuda_loader)

    remapper = RemapHead(
        x_ben_norm.shape[1],
        [256, 256],
        y_scaled.shape[1],
        int(dst_port_values.size),
        int(src_port_values.size),
    ).to(device)
    opt = torch.optim.AdamW(remapper.parameters(), lr=lr, weight_decay=weight_decay)
    loss_name = str(loss).lower()
    if loss_name in ("huber", "smoothl1"):
        loss_fn_cont = nn.SmoothL1Loss(beta=huber_beta)
    elif loss_name == "l1":
        loss_fn_cont = nn.L1Loss()
    else:
        loss_fn_cont = nn.MSELoss()
    loss_fn_port = nn.CrossEntropyLoss()
    train_log: List[Dict[str, float]] = []
    target_var = float(np.sum((y_scaled - np.mean(y_scaled, axis=0)) ** 2)) + 1.0e-6
    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    has_src = remapper.has_src_port

    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        mae_sum = 0.0
        mse_sum = 0.0
        dst_acc_sum = 0.0
        src_acc_sum = 0.0
        count = 0
        for xb, yb_cont, yb_dst, yb_src in loader:
            xb = xb.to(device, non_blocking=use_cuda_loader)
            yb_cont = yb_cont.to(device, non_blocking=use_cuda_loader)
            yb_dst = yb_dst.to(device, non_blocking=use_cuda_loader)
            yb_src = yb_src.to(device, non_blocking=use_cuda_loader)
            pred_cont, pred_dst, pred_src = remapper(xb)
            loss_cont = loss_fn_cont(pred_cont, yb_cont)
            loss_dst = loss_fn_port(pred_dst, yb_dst)
            loss = loss_cont + 0.25 * loss_dst
            if has_src and pred_src is not None:
                loss_src = loss_fn_port(pred_src, yb_src)
                loss = loss + 0.25 * loss_src
            opt.zero_grad()
            loss.backward()
            if grad_clip and grad_clip > 0.0:
                nn.utils.clip_grad_norm_(remapper.parameters(), grad_clip)
            opt.step()
            batch = xb.shape[0]
            loss_sum += float(loss.item()) * batch
            mae_sum += float(torch.mean(torch.abs(pred_cont - yb_cont)).item()) * batch
            mse_sum += float(torch.mean((pred_cont - yb_cont) ** 2).item()) * batch
            dst_acc_sum += float((pred_dst.argmax(dim=1) == yb_dst).float().mean().item()) * batch
            if has_src and pred_src is not None:
                src_acc_sum += float((pred_src.argmax(dim=1) == yb_src).float().mean().item()) * batch
            count += batch
        loss_mean = loss_sum / max(count, 1)
        mae_mean = mae_sum / max(count, 1)
        rmse = float(np.sqrt(mse_sum / max(count, 1)))
        dst_acc = dst_acc_sum / max(count, 1)
        src_acc = src_acc_sum / max(count, 1) if has_src else 1.0
        port_acc = 0.6 * dst_acc + 0.4 * src_acc
        r2 = _compute_r2_from_sse(mse_sum, target_var)
        score = float(r2 if r2 is not None and np.isfinite(float(r2)) else float("nan"))
        train_log.append(
            {
                "epoch": float(epoch),
                "loss": loss_mean,
                "mae": mae_mean,
                "rmse": rmse,
                "port_acc": port_acc,
                "dst_port_acc": dst_acc,
                "src_port_acc": src_acc,
                "r2": r2,
                "selection_score": score,
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in remapper.state_dict().items()}
        extra = f" dst_acc={dst_acc:.4f} src_acc={src_acc:.4f}" if has_src else ""
        print(
            f"[Stage3] epoch={epoch} loss={loss_mean:.6f} mae={mae_mean:.6f} "
            f"rmse={rmse:.6f} port_acc={port_acc:.6f} score={score:.6f}{extra}"
        )

    if best_state is not None:
        remapper.load_state_dict(best_state)

    # ── OOD detection: Mahalanobis parameters on benign training inputs ──
    ood_maha_mean = None
    ood_maha_inv_cov = None
    try:
        x_centered = x_in - np.mean(x_in, axis=0)
        cov = np.cov(x_centered, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1.0e-6
        ood_maha_mean = np.mean(x_in, axis=0).astype(np.float32)
        ood_maha_inv_cov = np.linalg.inv(cov).astype(np.float32)
    except Exception:
        pass

    # ── Information bottleneck: effective remapping degrees of freedom ──
    remap_effective_dim = None
    remap_subspace_explained_var = None
    try:
        # How many linearly independent modification dimensions are reachable?
        remapper.eval()
        x_t = torch.tensor(x_in, dtype=torch.float32)
        with torch.no_grad():
            cont, _, _ = remapper(x_t)
        mods = cont.detach().cpu().numpy()
        # SVD on the output modifications to estimate reachable subspace rank
        u, s, _ = np.linalg.svd(mods - np.mean(mods, axis=0), full_matrices=False)
        explained_var = np.cumsum(s**2) / np.sum(s**2)
        remap_effective_dim = int(np.searchsorted(explained_var, 0.95) + 1)
        remap_subspace_explained_var = float(explained_var[min(remap_effective_dim - 1, len(explained_var) - 1)])
    except Exception:
        pass

    return RemapBundle(
        remapper=remapper,
        feature_names=feature_names,
        mod_names=list(MOD_NAMES),
        mod_mean=mod_mean.astype(np.float32),
        mod_std=mod_std.astype(np.float32),
        input_mean=input_mean,
        input_std=input_std,
        continuous_names=list(CONTINUOUS_MOD_NAMES),
        port_values=dst_port_values.astype(np.int64),
        src_port_values=src_port_values.astype(np.int64),
        train_log=train_log,
        best_epoch=best_epoch if best_epoch > 0 else None,
        best_score=best_score if np.isfinite(best_score) else None,
        ood_maha_mean=ood_maha_mean,
        ood_maha_inv_cov=ood_maha_inv_cov,
        remap_effective_dim=remap_effective_dim,
        remap_subspace_explained_var=remap_subspace_explained_var,
    )


def predict_modifications(
    bundle: RemapBundle,
    x_adv_norm: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_in = np.asarray(x_adv_norm, dtype=np.float32)
    x_in = (x_in - bundle.input_mean) / (bundle.input_std + 1.0e-8)
    with torch.no_grad():
        result = bundle.remapper(torch.tensor(x_in, dtype=torch.float32, device=device))
    if len(result) == 3:
        pred_cont, pred_dst_port, pred_src_port = result
    else:
        pred_cont, pred_dst_port = result
        pred_src_port = None
    pred_cont = pred_cont.cpu().numpy()
    pred_dst_port = pred_dst_port.cpu().numpy()
    cont = pred_cont * bundle.mod_std + bundle.mod_mean
    dst_port = _decode_ports(pred_dst_port, bundle.port_values).reshape(-1, 1)
    src_port = (
        _decode_ports(pred_src_port.cpu().numpy(), bundle.src_port_values).reshape(-1, 1)
        if pred_src_port is not None and bundle.src_port_values is not None and bundle.src_port_values.size > 0
        else np.full((cont.shape[0], 1), 1024.0, dtype=np.float32)
    )
    mods = np.zeros((cont.shape[0], len(bundle.mod_names)), dtype=np.float32)
    for index, name in enumerate(bundle.mod_names):
        if name == PORT_MOD_NAMES[0]:  # dst_port_new
            mods[:, index] = dst_port[:, 0]
        elif name == PORT_MOD_NAMES[1]:  # src_port_new
            mods[:, index] = src_port[:, 0]
        else:
            cont_index = bundle.continuous_names.index(name)
            mods[:, index] = cont[:, cont_index]
    return mods.astype(np.float32)


def build_random_remap_modifications(
    x_adv_raw: np.ndarray,
    x_ben_raw: np.ndarray,
    feature_names: List[str],
    *,
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = build_rule_based_modifications(x_adv_raw=x_adv_raw, x_ben_raw=x_ben_raw, feature_names=feature_names)
    benign_targets = clip_modifications(build_remap_targets(x_ben_raw, feature_names))
    out = np.asarray(base, dtype=np.float32).copy()
    if benign_targets.shape[0] == 0:
        return clip_modifications(out)
    for mod_idx, name in enumerate(MOD_NAMES):
        if name in PORT_MOD_NAMES:
            port_idx = MOD_NAMES.index(name)
            port_values = _build_port_vocab(benign_targets[:, port_idx])
            choices = port_values if port_values.size else np.array([80], dtype=np.int64)
            out[:, mod_idx] = rng.choice(choices, size=out.shape[0]).astype(np.float32)
            continue
        col = np.asarray(benign_targets[:, mod_idx], dtype=np.float32)
        lo = float(np.quantile(col, 0.10))
        hi = float(np.quantile(col, 0.90))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo = float(np.min(col))
            hi = float(np.max(col))
        if hi < lo:
            lo, hi = hi, lo
        if np.isclose(lo, hi):
            out[:, mod_idx] = lo
        else:
            out[:, mod_idx] = rng.uniform(lo, hi, size=out.shape[0]).astype(np.float32)
    return clip_modifications(out)


def _compute_timing_schedule(
    pkt_times: np.ndarray,
    mean_iat_s: float,
    std_iat_s: float,
    flow_scale: float,
    apply_mean_std: bool,
) -> np.ndarray:
    times = np.asarray(pkt_times, dtype=np.float64)
    if times.size == 0:
        return np.zeros((0,), dtype=np.float64)
    deltas = np.diff(times, prepend=times[0])
    deltas[0] = 0.0
    if deltas.size <= 1:
        return deltas

    out = deltas.copy()
    if apply_mean_std:
        base = deltas[1:]
        if base.size:
            base_mean = float(np.mean(base))
            base_std = float(np.std(base))
            if base_std > 1.0e-9 and std_iat_s > 0.0:
                standardized = (base - base_mean) / base_std
                remapped = mean_iat_s + std_iat_s * standardized
            else:
                remapped = np.full_like(base, max(mean_iat_s, 0.0))
            positive_base = base[base > 0.0]
            ref_q95 = float(np.quantile(positive_base, 0.95)) if positive_base.size else max(mean_iat_s, 0.0)
            upper = max(ref_q95 * 4.0, mean_iat_s + 3.0 * std_iat_s, 1.0e-6)
            out[1:] = np.clip(remapped, 0.0, upper)
    scale = max(float(flow_scale), 0.0)
    if scale != 1.0:
        out[1:] *= scale
    return np.maximum(out, 0.0)


def apply_mod_using_scapy(
    pkts,
    mod: np.ndarray,
    seed: int = 42,
    apply_fields: List[str] | None = None,
    tcp_fixup: bool = True,
    dst_port_policy: str = "keep",
    dst_port_allowlist: List[int] | None = None,
    protocol_auto_fix: bool = True,
):
    from scapy.all import IP, TCP, UDP, Raw

    apply_fields = set(MOD_NAMES) if apply_fields is None else set(apply_fields)
    apply_timing = any(name in apply_fields for name in ("mean_iat_ms", "std_iat_ms", "flow_scale",
                                                          "fwd_iat_mean_ms", "fwd_iat_std_ms",
                                                          "bwd_iat_mean_ms", "bwd_iat_std_ms"))
    apply_mean_std = any(name in apply_fields for name in ("mean_iat_ms", "std_iat_ms"))
    apply_pad = "pad_bytes" in apply_fields
    apply_dport = "dst_port_new" in apply_fields
    apply_sport = "src_port_new" in apply_fields
    apply_flag = "flag_ratio" in apply_fields
    apply_payload_scale = "payload_scale" in apply_fields or "fwd_payload_scale" in apply_fields or "bwd_payload_scale" in apply_fields
    apply_syn = "syn_flag_ratio" in apply_fields
    apply_fin = "fin_flag_ratio" in apply_fields
    apply_rst = "rst_flag_ratio" in apply_fields
    apply_init_win_fwd = "tcp_init_win_fwd" in apply_fields
    apply_init_win_bwd = "tcp_init_win_bwd" in apply_fields
    apply_fwd_scale = "fwd_pkt_scale" in apply_fields
    apply_bwd_scale = "bwd_pkt_scale" in apply_fields

    mean_iat = float(mod[0]) / 1000.0
    std_iat = float(mod[1]) / 1000.0
    pad_bytes = int(round(float(mod[2])))
    dst_port_new = int(round(float(mod[3])))
    flag_ratio = float(mod[4])
    flow_scale = float(mod[5])
    payload_scale = float(mod[6]) if len(mod) > 6 else 1.0
    src_port_new = int(round(float(mod[7]))) if len(mod) > 7 else 1024
    tcp_init_win_fwd = int(round(float(mod[8]))) if len(mod) > 8 else 65535
    tcp_init_win_bwd = int(round(float(mod[9]))) if len(mod) > 9 else 65535
    syn_flag_ratio = float(mod[10]) if len(mod) > 10 else 0.0
    fin_flag_ratio = float(mod[11]) if len(mod) > 11 else 0.0
    rst_flag_ratio = float(mod[12]) if len(mod) > 12 else 0.0
    fwd_pkt_scale = float(mod[13]) if len(mod) > 13 else 1.0
    # --- 扩展参数 (indices 14-20) ---
    bwd_pkt_scale = float(mod[14]) if len(mod) > 14 else 1.0
    fwd_payload_scale = float(mod[15]) if len(mod) > 15 else payload_scale
    bwd_payload_scale = float(mod[16]) if len(mod) > 16 else payload_scale
    fwd_iat_mean = float(mod[17]) / 1000.0 if len(mod) > 17 else mean_iat
    fwd_iat_std = float(mod[18]) / 1000.0 if len(mod) > 18 else std_iat
    bwd_iat_mean = float(mod[19]) / 1000.0 if len(mod) > 19 else mean_iat
    bwd_iat_std = float(mod[20]) / 1000.0 if len(mod) > 20 else std_iat

    def _wrap32(value: int) -> int:
        return int(value) % (1 << 32)

    def _seq_advance(flags: int, payload_len: int) -> int:
        advance = max(int(payload_len), 0)
        if flags & 0x02:
            advance += 1
        if flags & 0x01:
            advance += 1
        return advance

    allowlist = [int(p) for p in (dst_port_allowlist or []) if 1 <= int(p) <= 65535]
    allowlist = sorted(set(allowlist))
    tcp_port_counts = Counter(int(pkt[TCP].dport) for pkt in pkts if TCP in pkt and 1 <= int(pkt[TCP].dport) <= 65535)
    udp_port_counts = Counter(int(pkt[UDP].dport) for pkt in pkts if UDP in pkt and 1 <= int(pkt[UDP].dport) <= 65535)
    observed_ports = sorted(set(tcp_port_counts) | set(udp_port_counts))

    def _valid_port_or_fallback(port: int, fallback: int) -> int:
        port = int(port)
        if 1 <= port <= 65535:
            return port
        fallback = int(fallback)
        if 1 <= fallback <= 65535:
            return fallback
        return 80

    def _rank_flow_vocab_candidate(
        port: int,
        *,
        orig_port: int,
        target_port: int,
        observed_for_proto: set[int],
        proto_counts: Counter[int],
    ) -> float:
        score = float(abs(int(port) - int(target_port)))
        if int(port) in observed_for_proto:
            score -= 32.0
        if int(port) == int(orig_port):
            score -= 16.0
        if proto_counts:
            score -= 8.0 * (float(proto_counts.get(int(port), 0)) / float(max(proto_counts.values())))
        return score

    def _select_dport(orig_port: int, proto_name: str) -> int:
        if not apply_dport:
            return orig_port
        policy = str(dst_port_policy or "keep").lower()
        if policy == "keep":
            return _valid_port_or_fallback(orig_port, 80)
        if not allowlist:
            return _valid_port_or_fallback(dst_port_new, orig_port)
        if policy == "allowlist":
            if dst_port_new in allowlist:
                return _valid_port_or_fallback(dst_port_new, orig_port)
            if orig_port in allowlist:
                return _valid_port_or_fallback(orig_port, 80)
            return allowlist[0]
        if policy == "closest":
            return min(allowlist, key=lambda p: abs(p - dst_port_new))
        if policy == "flow_vocab_closest":
            proto_counts = tcp_port_counts if proto_name == "tcp" else udp_port_counts
            observed_for_proto = set(proto_counts)
            flow_vocab = sorted(set(allowlist) | set(observed_ports) | observed_for_proto | {int(orig_port)})
            if dst_port_new in flow_vocab:
                return dst_port_new
            if flow_vocab:
                best_port = min(
                    flow_vocab,
                    key=lambda p: (
                        _rank_flow_vocab_candidate(
                            int(p),
                            orig_port=int(orig_port),
                            target_port=dst_port_new,
                            observed_for_proto=observed_for_proto,
                            proto_counts=proto_counts,
                        ),
                        abs(int(p) - int(orig_port)),
                    ),
                )
                if int(orig_port) in flow_vocab:
                    orig_score = _rank_flow_vocab_candidate(
                        int(orig_port),
                        orig_port=int(orig_port),
                        target_port=dst_port_new,
                        observed_for_proto=observed_for_proto,
                        proto_counts=proto_counts,
                    )
                    best_score = _rank_flow_vocab_candidate(
                        int(best_port),
                        orig_port=int(orig_port),
                        target_port=dst_port_new,
                        observed_for_proto=observed_for_proto,
                        proto_counts=proto_counts,
                    )
                    if orig_score <= best_score + 8.0:
                        return _valid_port_or_fallback(orig_port, 80)
                return _valid_port_or_fallback(best_port, orig_port)
            return _valid_port_or_fallback(orig_port, 80)
        return _valid_port_or_fallback(dst_port_new, orig_port)

    def _select_sport(orig_port: int) -> int:
        if not apply_sport:
            return _valid_port_or_fallback(orig_port, 80)
        return _valid_port_or_fallback(src_port_new, orig_port)

    first_ip = None
    for p in pkts:
        if IP in p:
            first_ip = p
            break
    src0 = first_ip[IP].src if first_ip is not None else None
    dst0 = first_ip[IP].dst if first_ip is not None else None

    def is_fwd(pkt) -> bool:
        return src0 is not None and IP in pkt and pkt[IP].src == src0 and pkt[IP].dst == dst0

    def _raw_payload(pkt) -> bytes:
        if Raw not in pkt:
            return b""
        try:
            return bytes(pkt[Raw].load)
        except Exception as exc:
            print(f"[Stage3/Remap][Warn] Raw payload extraction failed: {exc}")
            return b""

    def _cap_raw_payload_for_ip(pkt) -> None:
        if IP not in pkt or Raw not in pkt:
            return
        # Keep enough room for IP + transport headers and options. The exact
        # serialized header size can vary after Scapy rebuilds checksums, so use
        # a conservative cap below the IPv4 total-length uint16 ceiling.
        max_payload_len = 60000
        payload = bytes(pkt[Raw].load)
        if len(payload) > max_payload_len:
            pkt[Raw].load = payload[:max_payload_len]

    def _eligible_payload_indices() -> list[int]:
        preferred: list[int] = []
        fallback: list[int] = []
        for idx, pkt in enumerate(pkts):
            payload = _raw_payload(pkt)
            if not payload:
                continue
            fallback.append(idx)
            if TCP in pkt and is_fwd(pkt):
                preferred.append(idx)
        return preferred or fallback

    def _distribute_total(target_total: int, base_lengths: list[int]) -> dict[int, int]:
        if not base_lengths:
            return {}
        total_base = int(sum(base_lengths))
        target_total = max(int(target_total), 0)
        if total_base <= 0:
            return {idx: 0 for idx in range(len(base_lengths))}
        scaled = [length * target_total / total_base for length in base_lengths]
        floors = [int(np.floor(value)) for value in scaled]
        remainder = target_total - int(sum(floors))
        if remainder > 0:
            order = sorted(
                range(len(base_lengths)),
                key=lambda idx: (scaled[idx] - floors[idx], base_lengths[idx]),
                reverse=True,
            )
            for idx in order[:remainder]:
                floors[idx] += 1
        return {idx: max(0, floors[idx]) for idx in range(len(base_lengths))}

    payload_indices = _eligible_payload_indices()
    payload_lengths = [_raw_payload(pkts[idx]).__len__() for idx in payload_indices]
    payload_length_targets: dict[int, int] = {}
    if apply_payload_scale and payload_indices:
        # ── Per-direction payload scaling ──────────────────────────
        has_per_dir_payload = ("fwd_payload_scale" in apply_fields or "bwd_payload_scale" in apply_fields)
        if has_per_dir_payload:
            fwd_pi = [i for i in payload_indices if is_fwd(pkts[i])]
            bwd_pi = [i for i in payload_indices if not is_fwd(pkts[i])]
            fwd_lengths = [_raw_payload(pkts[idx]).__len__() for idx in fwd_pi]
            bwd_lengths = [_raw_payload(pkts[idx]).__len__() for idx in bwd_pi]
            if fwd_pi:
                fwd_desired = int(round(sum(fwd_lengths) * max(fwd_payload_scale, 0.0)))
                fwd_dist = _distribute_total(fwd_desired, fwd_lengths)
                for local_idx, target_len in fwd_dist.items():
                    payload_length_targets[fwd_pi[local_idx]] = int(target_len)
            if bwd_pi:
                bwd_desired = int(round(sum(bwd_lengths) * max(bwd_payload_scale, 0.0)))
                bwd_dist = _distribute_total(bwd_desired, bwd_lengths)
                for local_idx, target_len in bwd_dist.items():
                    payload_length_targets[bwd_pi[local_idx]] = int(target_len)
        else:
            desired_total = int(round(sum(payload_lengths) * max(payload_scale, 0.0)))
            distributed = _distribute_total(desired_total, payload_lengths)
            payload_length_targets = {
                payload_indices[local_idx]: int(target_len) for local_idx, target_len in distributed.items()
            }

    pad_targets: dict[int, int] = {}
    if apply_pad and pad_bytes > 0 and payload_indices:
        topk = min(len(payload_indices), max(1, min(4, int(np.ceil(len(payload_indices) / 4.0)))))
        ranked = sorted(payload_indices, key=lambda idx: _raw_payload(pkts[idx]).__len__(), reverse=True)
        selected = ranked[:topk]
        total_pad_budget = int(pad_bytes) * topk
        per_pkt = total_pad_budget // topk
        remainder = total_pad_budget % topk
        for offset, pkt_idx in enumerate(selected):
            pad_targets[pkt_idx] = int(per_pkt + (1 if offset < remainder else 0))

    flag_target_indices: set[int] = set()
    if apply_flag:
        eligible_flags: list[int] = []
        for idx, pkt in enumerate(pkts):
            if TCP not in pkt or not is_fwd(pkt):
                continue
            flags = int(pkt[TCP].flags)
            if flags & 0x01 or flags & 0x02 or flags & 0x04:
                continue
            if len(_raw_payload(pkt)) <= 0:
                continue
            eligible_flags.append(idx)
        if eligible_flags:
            desired_count = int(round(np.clip(flag_ratio, 0.0, 1.0) * len(eligible_flags)))
            ranked = sorted(eligible_flags, key=lambda idx: _raw_payload(pkts[idx]).__len__(), reverse=True)
            flag_target_indices = set(ranked[:desired_count])

    # ── SYN / FIN / RST flag targets ────────────────────────────────
    syn_target_indices: set[int] = set()
    fin_target_indices: set[int] = set()
    rst_target_indices: set[int] = set()
    if apply_syn or apply_fin or apply_rst:
        for idx, pkt in enumerate(pkts):
            if TCP not in pkt or not is_fwd(pkt):
                continue
            flags = int(pkt[TCP].flags)
            if apply_syn and (flags & 0x02) and not (flags & 0x10):
                syn_target_indices.add(idx)
            if apply_fin and not (flags & 0x02) and not (flags & 0x04) and len(_raw_payload(pkt)) > 0:
                fin_target_indices.add(idx)
            if apply_rst and not (flags & 0x02):
                rst_target_indices.add(idx)
        if syn_target_indices and apply_syn:
            syn_list = sorted(syn_target_indices)
            syn_desired = max(1, int(round(np.clip(syn_flag_ratio, 0.0, 1.0) * len(syn_list))))
            syn_target_indices = set(syn_list[:syn_desired])
        if fin_target_indices and apply_fin:
            fin_list = sorted(fin_target_indices)
            fin_desired = int(round(np.clip(fin_flag_ratio, 0.0, 1.0) * len(fin_list)))
            fin_target_indices = set(fin_list[:fin_desired])
        if rst_target_indices and apply_rst:
            rst_list = sorted(rst_target_indices)
            rst_desired = int(round(np.clip(rst_flag_ratio, 0.0, 1.0) * len(rst_list)))
            rst_target_indices = set(rst_list[:rst_desired])

    # ── Forward packet drop targets ─────────────────────────────────
    drop_indices: set[int] = set()
    if apply_fwd_scale and fwd_pkt_scale < 1.0:
        droppable = []
        for idx, pkt in enumerate(pkts):
            if TCP in pkt and is_fwd(pkt):
                flags = int(pkt[TCP].flags)
                if not (flags & 0x02) and not (flags & 0x01) and not (flags & 0x04):
                    droppable.append(idx)
            elif UDP in pkt and is_fwd(pkt):
                droppable.append(idx)
        if droppable:
            keep_count = max(1, int(round(len(droppable) * fwd_pkt_scale)))
            rng = np.random.default_rng(seed + 999)
            keep_set = set(rng.choice(droppable, size=keep_count, replace=False))
            drop_indices = set(droppable) - keep_set

    # ── Backward packet drop targets (扩展参数) ──────────────────────
    if apply_bwd_scale and bwd_pkt_scale < 1.0:
        droppable_bwd = []
        for idx, pkt in enumerate(pkts):
            if TCP in pkt and not is_fwd(pkt):
                flags = int(pkt[TCP].flags)
                if not (flags & 0x02) and not (flags & 0x01) and not (flags & 0x04):
                    droppable_bwd.append(idx)
            elif UDP in pkt and not is_fwd(pkt):
                droppable_bwd.append(idx)
        if droppable_bwd:
            keep_count_bwd = max(1, int(round(len(droppable_bwd) * bwd_pkt_scale)))
            rng_bwd = np.random.default_rng(seed + 1999)
            keep_set_bwd = set(rng_bwd.choice(droppable_bwd, size=keep_count_bwd, replace=False))
            drop_indices |= set(droppable_bwd) - keep_set_bwd

    # ── Per-direction timing schedules (扩展参数) ────────────────────
    pkt_times = (
        np.array([float(p.time) for p in pkts], dtype=np.float64) if len(pkts) else np.zeros((0,), dtype=np.float64)
    )
    has_per_dir_iat = ("fwd_iat_mean_ms" in apply_fields or "bwd_iat_mean_ms" in apply_fields)
    if has_per_dir_iat:
        fwd_indices_iat = [i for i, p in enumerate(pkts) if is_fwd(p)]
        bwd_indices_iat = [i for i, p in enumerate(pkts) if not is_fwd(p)]
        fwd_times_arr = np.array([float(pkts[i].time) for i in fwd_indices_iat], dtype=np.float64)
        bwd_times_arr = np.array([float(pkts[i].time) for i in bwd_indices_iat], dtype=np.float64)
        fwd_schedule = _compute_timing_schedule(
            fwd_times_arr, fwd_iat_mean, fwd_iat_std,
            flow_scale=(flow_scale if "flow_scale" in apply_fields else 1.0),
            apply_mean_std=bool("fwd_iat_mean_ms" in apply_fields or "fwd_iat_std_ms" in apply_fields),
        )
        bwd_schedule = _compute_timing_schedule(
            bwd_times_arr, bwd_iat_mean, bwd_iat_std,
            flow_scale=(flow_scale if "flow_scale" in apply_fields else 1.0),
            apply_mean_std=bool("bwd_iat_mean_ms" in apply_fields or "bwd_iat_std_ms" in apply_fields),
        )
        delta_map: dict[int, float] = {}
        for i, idx in enumerate(fwd_indices_iat):
            delta_map[idx] = float(fwd_schedule[i]) if i < fwd_schedule.size else 0.0
        for i, idx in enumerate(bwd_indices_iat):
            delta_map[idx] = float(bwd_schedule[i]) if i < bwd_schedule.size else 0.0
        delta_schedule = np.zeros((len(pkts),), dtype=np.float64)
        for i in range(len(pkts)):
            delta_schedule[i] = delta_map.get(i, 0.0)
    else:
        fwd_indices_iat = []
        bwd_indices_iat = []
        delta_schedule = _compute_timing_schedule(
            pkt_times,
            mean_iat_s=mean_iat,
            std_iat_s=std_iat,
            flow_scale=flow_scale if "flow_scale" in apply_fields else 1.0,
            apply_mean_std=apply_mean_std,
        )
    out = []
    t = pkts[0].time if len(pkts) else 0.0
    flow_breakpoints: dict[tuple[str, str, int, int], list[tuple[int, int]]] = {}

    def _flow_key(pkt) -> tuple[str, str, int, int] | None:
        if IP not in pkt or TCP not in pkt:
            return None
        return (str(pkt[IP].src), str(pkt[IP].dst), int(pkt[TCP].sport), int(pkt[TCP].dport))

    def _reverse_flow_key(key: tuple[str, str, int, int] | None) -> tuple[str, str, int, int] | None:
        if key is None:
            return None
        return (key[1], key[0], key[3], key[2])

    def _delta_for_seq(key: tuple[str, str, int, int] | None, seq_value: int) -> int:
        if key is None:
            return 0
        delta = 0
        for breakpoint_seq, cumulative_delta in flow_breakpoints.get(key, []):
            if seq_value >= breakpoint_seq:
                delta = cumulative_delta
            else:
                break
        return delta

    def _record_seq_delta(key: tuple[str, str, int, int] | None, seq_end: int, delta_inc: int) -> None:
        if key is None or delta_inc == 0:
            return
        points = list(flow_breakpoints.get(key, []))
        cumulative = points[-1][1] if points else 0
        points.append((int(seq_end), int(cumulative + delta_inc)))
        points.sort(key=lambda item: item[0])
        collapsed: list[tuple[int, int]] = []
        for breakpoint_seq, cumulative_delta in points:
            if collapsed and collapsed[-1][0] == breakpoint_seq:
                collapsed[-1] = (breakpoint_seq, cumulative_delta)
            else:
                collapsed.append((breakpoint_seq, cumulative_delta))
        flow_breakpoints[key] = collapsed

    for index, p in enumerate(pkts):
        if index in drop_indices:
            # Record SEQ delta for dropped forward data packets
            if TCP in p and tcp_fixup:
                flags = int(p[TCP].flags)
                key = _flow_key(p)
                payload_len = len(_raw_payload(p))
                seq_end = int(p[TCP].seq) + _seq_advance(flags, payload_len)
                _record_seq_delta(key, seq_end, -payload_len)
            continue

        q = p.copy()
        if apply_timing:
            delta = float(delta_schedule[index]) if index < delta_schedule.size else 0.0
            t += delta
            q.time = t
        else:
            q.time = p.time

        if TCP in q and tcp_fixup:
            flags = int(q[TCP].flags)
            key = _flow_key(p)
            rev_key = _reverse_flow_key(key)
            orig_seq = int(p[TCP].seq)
            q[TCP].seq = _wrap32(orig_seq + _delta_for_seq(key, orig_seq))
            if flags & 0x10:
                orig_ack = int(p[TCP].ack)
                q[TCP].ack = _wrap32(orig_ack + _delta_for_seq(rev_key, orig_ack))
        else:
            key = None
            flags = int(q[TCP].flags) if TCP in q else 0

        payload_delta = 0
        payload_before_len = len(_raw_payload(p)) if TCP in p else 0
        if apply_payload_scale and Raw in q and index in payload_length_targets:
            payload = bytes(q[Raw].load)
            target_len = max(int(payload_length_targets[index]), 0)
            if target_len >= len(payload):
                new_payload = payload + bytes([0] * (target_len - len(payload)))
            else:
                new_payload = payload[:target_len]
            q[Raw].load = new_payload
            payload_delta += len(new_payload) - len(payload)

        if apply_pad and index in pad_targets:
            add_bytes = int(pad_targets[index])
            if Raw in q:
                payload = bytes(q[Raw].load)
                new_payload = payload + bytes([0] * add_bytes)
                q[Raw].load = new_payload
            else:
                q = q / Raw(bytes([0] * add_bytes))
            payload_delta += add_bytes

        if TCP in q and tcp_fixup and payload_delta != 0:
            seq_end = int(p[TCP].seq) + _seq_advance(flags, payload_before_len)
            _record_seq_delta(key, seq_end, payload_delta)

        if TCP in q or UDP in q:
            if TCP in q:
                q[TCP].sport = (
                    _select_sport(int(q[TCP].sport)) if apply_sport else _valid_port_or_fallback(int(q[TCP].sport), 80)
                )
                if apply_dport:
                    q[TCP].dport = _select_dport(int(q[TCP].dport), "tcp")
                else:
                    q[TCP].dport = _valid_port_or_fallback(int(q[TCP].dport), 80)
                # TCP initial window on SYN packets
                if apply_init_win_fwd and is_fwd(p) and (int(p[TCP].flags) & 0x02):
                    q[TCP].window = max(0, min(65535, tcp_init_win_fwd))
                elif apply_init_win_bwd and not is_fwd(p) and (int(p[TCP].flags) & 0x02):
                    q[TCP].window = max(0, min(65535, tcp_init_win_bwd))
                # Flag manipulation (PSH + SYN/FIN/RST)
                if apply_flag or apply_syn or apply_fin or apply_rst:
                    flags = int(q[TCP].flags)
                    if apply_flag and not (flags & 0x01 or flags & 0x02 or flags & 0x04) and len(_raw_payload(q)) > 0:
                        if index in flag_target_indices:
                            flags |= 0x08
                        else:
                            flags &= ~0x08
                    if apply_syn and index in syn_target_indices:
                        flags |= 0x02
                    if apply_fin and index in fin_target_indices:
                        flags |= 0x01
                    if apply_rst and index in rst_target_indices:
                        flags = 0x04 | 0x10  # RST|ACK
                    q[TCP].flags = flags
                if protocol_auto_fix:
                    del q[TCP].chksum
            elif UDP in q:
                q[UDP].sport = (
                    _select_sport(int(q[UDP].sport)) if apply_sport else _valid_port_or_fallback(int(q[UDP].sport), 80)
                )
                if apply_dport:
                    q[UDP].dport = _select_dport(int(q[UDP].dport), "udp")
                else:
                    q[UDP].dport = _valid_port_or_fallback(int(q[UDP].dport), 80)
                if protocol_auto_fix:
                    if IP in q:
                        del q[UDP].len
                    del q[UDP].chksum
            if IP in q and protocol_auto_fix:
                _cap_raw_payload_for_ip(q)
                q[IP].ttl = int(np.clip(int(getattr(q[IP], "ttl", 64)), 1, 255))
                del q[IP].len
                del q[IP].chksum
        out.append(q)
    return out


# ── Remapping diagnostics (P2-1, P2-2) ──────────────────────────────────────


def detect_remap_ood(
    bundle: RemapBundle,
    x: np.ndarray,
    *,
    threshold_percentile: float = 95.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect OOD inputs using Mahalanobis distance from benign training distribution.

    Returns (is_ood: bool array, distances: float array).
    """
    if bundle.ood_maha_mean is None or bundle.ood_maha_inv_cov is None:
        return np.zeros(x.shape[0], dtype=bool), np.zeros(x.shape[0], dtype=np.float32)

    x_norm = (np.asarray(x, dtype=np.float32) - bundle.input_mean) / (bundle.input_std + 1.0e-6)
    x_norm_centered = x_norm - bundle.ood_maha_mean
    dists = np.sqrt(np.sum((x_norm_centered @ bundle.ood_maha_inv_cov) * x_norm_centered, axis=1))
    threshold = float(np.percentile(dists, threshold_percentile)) if x_norm.shape[0] > 1 else float(np.max(dists))
    return dists > threshold, dists.astype(np.float32)


def compute_remap_info_loss(
    stage2_target: np.ndarray,
    remapped_features: np.ndarray,
    feature_names: list[str],
    remap_mod_names: list[str],
) -> dict[str, float]:
    """Quantify how much of the Stage2 adversarial target shift is captured by remapping.

    stage2_target: adversarial feature vectors from Stage2 (d-dimensional)
    remapped_features: features extracted from remapped PCAP (d-dimensional)
    feature_names: names of the d features
    remap_mod_names: names of the remappable modification parameters (m-dimensional)
    """
    d = stage2_target.shape[1]
    m = len(remap_mod_names)

    # Per-feature target error
    per_feature_l2 = np.mean((stage2_target - remapped_features) ** 2, axis=0)
    total_l2 = float(np.mean(np.linalg.norm(stage2_target - remapped_features, axis=1)))

    # Information loss: how much of the target shift is preserved?
    target_shift = stage2_target - remapped_features
    shift_var = float(np.var(np.linalg.norm(target_shift, axis=1)))

    # Effective feature coverage: how many of the d target dimensions are within tolerance?
    feature_std = np.std(remapped_features, axis=0) + 1.0e-8
    aligned_count = int(np.sum(per_feature_l2 < (0.1 * feature_std)))

    return {
        "remap_info_total_l2": total_l2,
        "remap_info_per_feature_l2_mean": float(np.mean(per_feature_l2)),
        "remap_info_target_dim": d,
        "remap_info_mod_dim": m,
        "remap_info_dim_ratio": float(m) / float(d) if d > 0 else 0.0,
        "remap_info_aligned_features": aligned_count,
        "remap_info_aligned_ratio": float(aligned_count) / float(d) if d > 0 else 0.0,
        "remap_info_shift_variance": shift_var,
    }
