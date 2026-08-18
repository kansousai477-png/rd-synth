from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401
import numpy as np
import torch
from sklearn.metrics import f1_score

from rdsynth.pipeline.data import load_data_context
from rdsynth.pipeline.reviewer_suite import load_json
from rdsynth.stages.oracle import predict_sklearn_probs, train_oracle_from_config
from rdsynth.utils.config import load_yaml
from rdsynth.utils.pipeline_config import prepare_pipeline_config

TRANSFER_IDS_PRESETS: dict[str, dict[str, Any]] = {
    "logistic_small": {
        "type": "logistic",
        "class_weight": "balanced",
    },
    "random_forest_small": {
        "type": "random_forest",
        "class_weight": "balanced",
    },
    "linear_svm_small": {
        "type": "linear_svm",
        "class_weight": "balanced",
    },
}


def _predict_probs(bundle: Any, x: np.ndarray, device: torch.device) -> np.ndarray:
    model_type = str(bundle.model_type)
    if model_type in {"mlp", "cnn", "rnn", "lstm", "gru", "transformer"}:
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32, device=device)
            logits = bundle.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return np.asarray(probs, dtype=np.float32)
    probs = predict_sklearn_probs(bundle.model, x)
    if probs is None:
        raise RuntimeError(f"Unable to compute probabilities for oracle type '{model_type}'.")
    return np.asarray(probs, dtype=np.float32)


def _load_adv_pre(run_dir: Path) -> np.ndarray:
    adv_path = run_dir / "stage2" / "adv_samples.npz"
    if not adv_path.exists():
        raise FileNotFoundError(f"adv samples not found: {adv_path}")
    payload = np.load(adv_path)
    adv = payload.get("adv_pre")
    if adv is None:
        adv = payload.get("adv")
    if adv is None:
        raise RuntimeError(f"adv samples missing adv/adv_pre array: {adv_path}")
    return np.asarray(adv, dtype=np.float32)


def _evaluate_transfer_ids_model(
    *,
    oracle_name: str,
    seed: int,
    bundle: Any,
    adv_pre: np.ndarray,
    device: torch.device,
    main_asr: object,
) -> dict[str, str]:
    ids_cfg = dict(TRANSFER_IDS_PRESETS[oracle_name])
    ids_cfg["name"] = oracle_name
    oracle_bundle, val_acc = train_oracle_from_config(
        name=oracle_name,
        cfg=ids_cfg,
        x_train=bundle.x_train,
        y_train=bundle.y_train,
        x_val=bundle.x_val,
        y_val=bundle.y_val,
        device=device,
        seed=seed,
    )

    y_test = np.asarray(bundle.y_test, dtype=np.int64)
    mal_test = bundle.x_test[bundle.y_test == 1]

    test_probs = _predict_probs(oracle_bundle, bundle.x_test, device)
    test_pred = np.argmax(test_probs, axis=1)
    test_acc = float(np.mean(test_pred == y_test)) if len(y_test) else float("nan")
    test_f1 = float(f1_score(y_test, test_pred, zero_division=0)) if len(y_test) else float("nan")

    adv_probs = _predict_probs(oracle_bundle, adv_pre, device)
    adv_pred = np.argmax(adv_probs, axis=1)
    adv_asr = float(np.mean(adv_pred == 0)) if len(adv_pred) else float("nan")
    adv_prob_malicious = float(np.mean(adv_probs[:, 1])) if adv_probs.size else float("nan")

    mal_probs = _predict_probs(oracle_bundle, mal_test, device) if len(mal_test) else np.zeros((0, 2), dtype=np.float32)
    mal_pred = np.argmax(mal_probs, axis=1) if mal_probs.size else np.zeros((0,), dtype=np.int64)
    mal_detect_rate = float(np.mean(mal_pred == 1)) if len(mal_pred) else float("nan")
    mal_prob_malicious = float(np.mean(mal_probs[:, 1])) if mal_probs.size else float("nan")

    delta_vs_main = None
    try:
        delta_vs_main = float(adv_asr - float(main_asr))
    except (TypeError, ValueError):
        delta_vs_main = None

    return {
        "ids_name": oracle_name,
        "model_type": str(oracle_bundle.model_type),
        "val_acc": f"{float(val_acc):.6f}",
        "test_acc": f"{test_acc:.6f}",
        "test_f1": f"{test_f1:.6f}",
        "adv_asr": f"{adv_asr:.6f}",
        "adv_prob_malicious_mean": f"{adv_prob_malicious:.6f}",
        "mal_test_detect_rate": f"{mal_detect_rate:.6f}",
        "mal_test_prob_malicious_mean": f"{mal_prob_malicious:.6f}",
        "delta_asr_vs_main_ids": "" if delta_vs_main is None else f"{delta_vs_main:.6f}",
        "adv_count": str(int(adv_pre.shape[0])),
        "test_count": str(int(bundle.x_test.shape[0])),
        "mal_test_count": str(int(mal_test.shape[0])),
    }


def run_transfer_ids_eval(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    ids_names: list[str],
    out_path: str | Path | None = None,
    jobs: int = 0,
) -> Path:
    cfg = prepare_pipeline_config(load_yaml(config_path), config_path)
    run_dir = Path(run_dir).resolve()
    out_path = Path(out_path).resolve() if out_path else (run_dir / "pipeline" / "transfer_ids.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = [token.strip() for token in ids_names if token.strip()]
    unknown = [name for name in selected if name not in TRANSFER_IDS_PRESETS]
    if unknown:
        raise SystemExit(f"Unknown transfer IDS presets: {', '.join(unknown)}")

    seed = int(cfg["project"].get("seed", 42))
    data_ctx = load_data_context(cfg, seed)
    bundle = data_ctx.bundle
    adv_pre = _load_adv_pre(run_dir)
    if adv_pre.shape[1] != bundle.x_train.shape[1]:
        raise RuntimeError(
            f"adv feature dim mismatch: adv={adv_pre.shape[1]} expected={bundle.x_train.shape[1]}"
        )

    device = torch.device("cpu")
    stage2_metrics = load_json(run_dir / "stage2" / "metrics.json")
    main_asr = stage2_metrics.get("asr_oracle")

    worker_count = jobs if jobs and jobs > 0 else min(len(selected), 3)
    worker_count = max(1, min(worker_count, len(selected)))
    rows: list[dict[str, str]] = []
    if worker_count == 1:
        for oracle_name in selected:
            rows.append(
                _evaluate_transfer_ids_model(
                    oracle_name=oracle_name,
                    seed=seed,
                    bundle=bundle,
                    adv_pre=adv_pre,
                    device=device,
                    main_asr=main_asr,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {
                pool.submit(
                    _evaluate_transfer_ids_model,
                    oracle_name=oracle_name,
                    seed=seed,
                    bundle=bundle,
                    adv_pre=adv_pre,
                    device=device,
                    main_asr=main_asr,
                ): oracle_name
                for oracle_name in selected
            }
            for future in as_completed(futures):
                rows.append(future.result())
    order = {name: idx for idx, name in enumerate(selected)}
    rows.sort(key=lambda row: order.get(str(row.get("ids_name", "")), 999))

    fieldnames = list(rows[0].keys()) if rows else ["ids_name"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[TransferIDS] saved {out_path}")
    return out_path


def run_transfer_oracle_eval(
    *,
    config_path: str | Path,
    run_dir: str | Path,
    oracle_names: list[str],
    out_path: str | Path | None = None,
    jobs: int = 0,
) -> Path:
    return run_transfer_ids_eval(
        config_path=config_path,
        run_dir=run_dir,
        ids_names=oracle_names,
        out_path=out_path,
        jobs=jobs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage2 adversarial samples on transfer IDS models.")
    parser.add_argument("--config", required=True, help="Pipeline config path used for this run.")
    parser.add_argument("--run-dir", required=True, help="Pipeline output directory for a single run.")
    parser.add_argument(
        "--ids",
        default="logistic_small,random_forest_small",
        help="Comma-separated transfer IDS preset names.",
    )
    parser.add_argument("--oracles", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--out",
        default="",
        help="Output CSV path. Defaults to <run-dir>/pipeline/transfer_ids.csv",
    )
    parser.add_argument("--jobs", type=int, default=0, help="Parallel transfer-IDS workers. 0 means auto.")
    args = parser.parse_args()
    names_arg = args.ids or args.oracles
    run_transfer_ids_eval(
        config_path=args.config,
        run_dir=args.run_dir,
        ids_names=[token.strip() for token in names_arg.split(",") if token.strip()],
        out_path=args.out or None,
        jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
