"""Quick scan of all malicious PCAPs using the trained oracle to find strong candidates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from rdsynth.pipeline.data import load_data_context
from rdsynth.pipeline.preprocessing import DatasetPreprocessor
from rdsynth.pipeline.runtime import load_stage_runtime
from rdsynth.pipeline.stage3_calibration import CalibratedPreprocessor, compute_pcap_calibration
from rdsynth.pipeline.stage3_pcap import PcapFeatureExtractor
from rdsynth.pipeline.stage3_ops import load_stage3_artifacts
from rdsynth.utils.feature_align import build_statistical_feature_aliases, load_feature_aliases


def main():
    config_path = "configs/paper_main.yaml"
    runtime = load_stage_runtime(config_path, "stage3")
    cfg = runtime.cfg
    seed = runtime.seed
    device = runtime.device
    stage3_cfg = runtime.stage_cfg

    data_ctx = load_data_context(cfg, seed)
    bundle = data_ctx.bundle
    preprocessor = DatasetPreprocessor.from_bundle(bundle)

    x_train = bundle.x_train
    y_train = bundle.y_train
    x_ben = x_train[y_train == 0]
    x_ben_raw = preprocessor.inverse_transform(x_ben)
    raw_feature_mean = preprocessor.feature_mean(x_train)

    oracle_name = stage3_cfg.get("oracle_name", "mlp_small")
    artifacts = load_stage3_artifacts(
        cfg=cfg, oracle_name=oracle_name,
        x_train=x_train, y_train=y_train,
        x_val=bundle.x_val, y_val=bundle.y_val,
        feature_names=list(bundle.feature_names),
        device=device, seed=seed,
    )
    oracle = artifacts.oracle
    surrogate = artifacts.surrogate if artifacts.checkpoint_path.exists() else None

    # Build alias map + extractor
    alias_path = stage3_cfg.get("feature_aliases_path", "")
    alias_map = build_statistical_feature_aliases(
        bundle.feature_names,
        dataset_name=str(cfg.get("data", {}).get("dataset", "")),
        base_alias_map=load_feature_aliases(alias_path),
    )

    feature_names = list(bundle.feature_names)
    feature_backend = stage3_cfg.get("feature_backend", "auto")
    align_min_cov = float(stage3_cfg.get("pcap_align_min_coverage", 0.70))

    # Try calibration with benign PCAP
    benign_path = Path(stage3_cfg.get("pcap_ids_benign_path", "data/PCAPs/benign/benign.pcap"))
    calibration_active = False
    calibrated_preprocessor = CalibratedPreprocessor(preprocessor, compute_pcap_calibration(
        dataset_benign_raw=np.zeros((0, x_train.shape[1])), pcap_benign_raw=np.zeros((0, x_train.shape[1])), min_samples=999_999,
    ))

    pcap_eval_model = oracle
    pcap_eval_model_name = "oracle"

    pcap_features = PcapFeatureExtractor(
        feature_backend=feature_backend,
        feature_names=feature_names,
        raw_feature_mean=raw_feature_mean,
        alias_map=alias_map,
        align_min_cov=align_min_cov,
        scapy_available=True, nfstream_available=True,
        fail_closed=False, fail_on_partial_alignment=False,
        preprocessor=preprocessor,
        pcap_eval_model=pcap_eval_model,
        pcap_eval_model_name=pcap_eval_model_name,
        ids=None, oracle=oracle, surrogate=surrogate,
        pcap_eval_batch_size=256, seed=seed, device=device,
    )

    # Try to compute calibration from benign PCAP
    calibration = None
    if benign_path.exists():
        print(f"[scan] Loading benign PCAP: {benign_path}")
        try:
            bfeat, _, bmeta = pcap_features.extract(str(benign_path))
            bfeat = np.asarray(bfeat, dtype=np.float64)
            if bfeat.ndim == 2 and bfeat.shape[0] >= 10 and bfeat.shape[1] == x_train.shape[1]:
                calibration = compute_pcap_calibration(
                    dataset_benign_raw=x_ben_raw,
                    pcap_benign_raw=bfeat,
                    min_samples=10,
                )
                calibrated_preprocessor = CalibratedPreprocessor(preprocessor, calibration)
                calibration_active = calibration.is_active
                print(f"[scan] Calibration active: {calibration_active}, pcap_samples={calibration.pcap_sample_count}")
        except Exception as exc:
            print(f"[scan] Calibration failed: {exc}")

    if calibration_active:
        pcap_eval_model = oracle
        pcap_eval_model_name = "oracle"
        pcap_features = PcapFeatureExtractor(
            feature_backend=feature_backend, feature_names=feature_names,
            raw_feature_mean=raw_feature_mean, alias_map=alias_map,
            align_min_cov=align_min_cov,
            scapy_available=True, nfstream_available=True,
            fail_closed=False, fail_on_partial_alignment=False,
            preprocessor=calibrated_preprocessor,
            pcap_eval_model=pcap_eval_model,
            pcap_eval_model_name=pcap_eval_model_name,
            ids=None, oracle=oracle, surrogate=surrogate,
            pcap_eval_batch_size=256, seed=seed, device=device,
        )
        print("[scan] Using calibrated preprocessor + oracle evaluator")

    # Scan all malicious PCAPs
    scan_dir = Path("data/PCAPs/malicious")
    if not scan_dir.exists():
        print("[scan] No malicious PCAP dir found")
        return

    ranked = pcap_features.rank_pcaps(
        scan_dir, "*.pcap", limit=200,
        pmal_weight=0.80, target_fit_weight=0.10, target_mod_fit_weight=0.10,
    )

    print(f"\n[scan] === Top 20 PCAPs ranked by oracle malicious probability ===\n")
    for i, row in enumerate(ranked[:20]):
        pmal = row.get("prob_malicious")
        pmal_str = f"{pmal:.4f}" if isinstance(pmal, (int, float)) and np.isfinite(pmal) else str(pmal)
        print(
            f"  {i+1:2d}. pmal={pmal_str}  "
            f"pred={row.get('pred_label', '?')}  "
            f"flows={row.get('flow_count', '?')}  "
            f"size={row.get('pcap_size_bytes', 0)}  "
            f"status={row.get('status', '?')}  "
            f"name={row.get('name', '?')}"
        )

    # Save full ranking
    out_path = Path("outputs/paper_main/pcap_scan_ranking.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = []
    for row in ranked:
        serializable.append({
            k: (float(v) if isinstance(v, (np.floating,)) else str(v) if isinstance(v, Path) else v)
            for k, v in row.items()
        })
    out_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")
    print(f"\n[scan] Full ranking saved to {out_path}")

    # Count strong candidates
    strong = [r for r in ranked if r.get("prob_malicious") is not None and float(r["prob_malicious"]) >= 0.5]
    print(f"[scan] Strong candidates (pmal >= 0.5): {len(strong)}")
    detected = [r for r in ranked if r.get("pred_label") == 1]
    print(f"[scan] Detected as malicious (pred=1): {len(detected)}")


if __name__ == "__main__":
    main()
