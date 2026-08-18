from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from rdsynth.models.mlp import MLP
from rdsynth.models.tabular import (
    TabularCNN,
    TabularGRU,
    TabularLSTM,
    TabularRNN,
    TabularTransformer,
)


@dataclass
class OracleBundle:
    name: str
    model: Any
    n_classes: int
    model_type: str


SAFE_LINEAR_ORACLE_TYPES = {"logistic", "linear_svm"}


def build_oracle_model(model_type: str, feature_dim: int, cfg: Dict[str, Any], n_classes: int) -> Any:
    if model_type == "mlp":
        return MLP(feature_dim, cfg["hidden_dims"], n_classes)
    if model_type == "cnn":
        return TabularCNN(in_dim=feature_dim, channels=cfg.get("channels", 32), out_dim=n_classes)
    if model_type == "rnn":
        return TabularRNN(in_dim=feature_dim, hidden_dim=cfg.get("hidden_dim", 64), out_dim=n_classes)
    if model_type == "gru":
        return TabularGRU(in_dim=feature_dim, hidden_dim=cfg.get("hidden_dim", 64), out_dim=n_classes)
    if model_type == "lstm":
        return TabularLSTM(in_dim=feature_dim, hidden_dim=cfg.get("hidden_dim", 64), out_dim=n_classes)
    if model_type == "transformer":
        return TabularTransformer(
            in_dim=feature_dim,
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            out_dim=n_classes,
        )
    raise ValueError(f"Unknown oracle type: {model_type}")


def _class_weight_tensor(
    y_train: np.ndarray,
    class_weight: str | Dict[int, float] | list[float] | None,
    device: torch.device,
) -> torch.Tensor | None:
    if class_weight is None:
        return None
    n_classes = int(np.max(y_train)) + 1
    if class_weight == "balanced":
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
        weights = counts.sum() / (counts + 1.0e-6)
        return torch.tensor(weights, dtype=torch.float32, device=device)
    if isinstance(class_weight, dict):
        weights = np.ones(n_classes, dtype=np.float32)
        for k, v in class_weight.items():
            weights[int(k)] = float(v)
        return torch.tensor(weights, dtype=torch.float32, device=device)
    if isinstance(class_weight, (list, tuple, np.ndarray)):
        return torch.tensor(class_weight, dtype=torch.float32, device=device)
    return None


def _class_weight_sklearn(
    y_train: np.ndarray,
    class_weight: str | Dict[int, float] | list[float] | None,
) -> str | Dict[int, float] | None:
    if class_weight is None:
        return None
    if class_weight == "balanced":
        return "balanced"
    if isinstance(class_weight, dict):
        return {int(k): float(v) for k, v in class_weight.items()}
    if isinstance(class_weight, (list, tuple, np.ndarray)):
        return {idx: float(w) for idx, w in enumerate(class_weight)}
    return None


def _train_mlp_oracle(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden_dims: Tuple[int, ...],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    class_weight: str | Dict[int, float] | list[float] | None,
    sample_strategy: str | None,
) -> Tuple[nn.Module, float]:
    n_classes = int(np.max(y_train)) + 1
    use_cuda_loader = device.type == "cuda"
    model = MLP(x_train.shape[1], hidden_dims, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight_tensor = _class_weight_tensor(y_train, class_weight, device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    sampler = None
    if sample_strategy == "balanced":
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float32)
        weights = counts.sum() / (counts + 1.0e-6)
        sample_weights = torch.tensor(weights[y_train], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.long, device=device)

    model.train()
    for _ in range(epochs):
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=use_cuda_loader,
        )
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=use_cuda_loader)
            yb = yb.to(device, non_blocking=use_cuda_loader)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        acc = (logits.argmax(dim=1) == val_y).float().mean().item()
    return model, acc


def _train_torch_oracle(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    max_batches_per_epoch: int | None,
    device: torch.device,
    class_weight: str | Dict[int, float] | list[float] | None,
    sample_strategy: str | None,
) -> float:
    use_cuda_loader = device.type == "cuda"
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    weight_tensor = _class_weight_tensor(y_train, class_weight, device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    sampler = None
    if sample_strategy == "balanced":
        counts = np.bincount(y_train, minlength=int(np.max(y_train)) + 1).astype(np.float32)
        weights = counts.sum() / (counts + 1.0e-6)
        sample_weights = torch.tensor(weights[y_train], dtype=torch.float32)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val, dtype=torch.long, device=device)

    model.train()
    for epoch in range(1, epochs + 1):
        loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=use_cuda_loader,
        )
        for batch_idx, (xb, yb) in enumerate(loader, start=1):
            xb = xb.to(device, non_blocking=use_cuda_loader)
            yb = yb.to(device, non_blocking=use_cuda_loader)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if max_batches_per_epoch is not None and batch_idx >= max_batches_per_epoch:
                break
        print(f"[Oracle:{model.__class__.__name__}] epoch={epoch}/{epochs} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        logits = model(val_x)
        acc = (logits.argmax(dim=1) == val_y).float().mean().item()
    return acc


def _train_logistic_oracle(
    x_train: np.ndarray, y_train: np.ndarray, class_weight: str | Dict[int, float] | None, random_state: int
) -> LogisticRegression:
    model = LogisticRegression(max_iter=1000, n_jobs=None, class_weight=class_weight, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def _train_rf_oracle(
    x_train: np.ndarray, y_train: np.ndarray, class_weight: str | Dict[int, float] | None, random_state: int
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        n_jobs=1,
        class_weight=class_weight,
        random_state=random_state,
    )
    model.fit(x_train, y_train)
    return model


def _train_extra_trees_oracle(
    x_train: np.ndarray, y_train: np.ndarray, class_weight: str | Dict[int, float] | None, random_state: int
) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=1,
        random_state=random_state,
        class_weight=class_weight,
    )
    model.fit(x_train, y_train)
    return model


def _train_linear_svm_oracle(
    x_train: np.ndarray, y_train: np.ndarray, class_weight: str | Dict[int, float] | None, random_state: int
) -> LinearSVC:
    model = LinearSVC(C=1.0, max_iter=5000, class_weight=class_weight, random_state=random_state)
    model.fit(x_train, y_train)
    return model


def train_oracle_from_config(
    name: str,
    cfg: Dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    seed: int = 0,
) -> Tuple[OracleBundle, float]:
    model_type = cfg["type"]
    sample_strategy = cfg.get("sample_strategy")
    class_weight_cfg = cfg.get("class_weight")
    class_weight_sklearn = _class_weight_sklearn(y_train, class_weight_cfg)
    n_classes = int(np.max(y_train)) + 1
    print(f"[Oracle] start training {name} ({model_type})")
    if model_type == "mlp":
        model, val_acc = _train_mlp_oracle(
            x_train,
            y_train,
            x_val,
            y_val,
            hidden_dims=tuple(cfg["hidden_dims"]),
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            device=device,
            class_weight=class_weight_cfg,
            sample_strategy=sample_strategy,
        )
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "cnn":
        model = TabularCNN(in_dim=x_train.shape[1], channels=cfg.get("channels", 32), out_dim=n_classes)
        val_acc = _train_torch_oracle(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            max_batches_per_epoch=cfg.get("max_batches_per_epoch"),
            device=device,
            class_weight=class_weight_cfg,
            sample_strategy=sample_strategy,
        )
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "rnn":
        model = TabularRNN(in_dim=x_train.shape[1], hidden_dim=cfg.get("hidden_dim", 64), out_dim=n_classes)
        val_acc = _train_torch_oracle(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            max_batches_per_epoch=cfg.get("max_batches_per_epoch"),
            device=device,
            class_weight=class_weight_cfg,
            sample_strategy=sample_strategy,
        )
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "lstm":
        model = TabularLSTM(in_dim=x_train.shape[1], hidden_dim=cfg.get("hidden_dim", 64), out_dim=n_classes)
        val_acc = _train_torch_oracle(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            max_batches_per_epoch=cfg.get("max_batches_per_epoch"),
            device=device,
            class_weight=class_weight_cfg,
            sample_strategy=sample_strategy,
        )
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "gru":
        model = TabularGRU(in_dim=x_train.shape[1], hidden_dim=cfg.get("hidden_dim", 64), out_dim=n_classes)
        val_acc = _train_torch_oracle(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            max_batches_per_epoch=cfg.get("max_batches_per_epoch"),
            device=device,
            class_weight=class_weight_cfg,
            sample_strategy=sample_strategy,
        )
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "transformer":
        model = TabularTransformer(
            in_dim=x_train.shape[1],
            d_model=cfg.get("d_model", 64),
            nhead=cfg.get("nhead", 4),
            num_layers=cfg.get("num_layers", 2),
            out_dim=n_classes,
        )
        val_acc = _train_torch_oracle(
            model,
            x_train,
            y_train,
            x_val,
            y_val,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            max_batches_per_epoch=cfg.get("max_batches_per_epoch"),
            device=device,
            class_weight=class_weight_cfg,
            sample_strategy=sample_strategy,
        )
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "logistic":
        model = _train_logistic_oracle(x_train, y_train, class_weight_sklearn, random_state=int(seed))
        val_pred = model.predict(x_val)
        val_acc = float(np.mean(val_pred == y_val))
        print(f"[Oracle] finished {name} ({model_type}) val_acc={val_acc:.4f}")
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "random_forest":
        model = _train_rf_oracle(x_train, y_train, class_weight_sklearn, random_state=int(seed))
        val_pred = model.predict(x_val)
        val_acc = float(np.mean(val_pred == y_val))
        print(f"[Oracle] finished {name} ({model_type}) val_acc={val_acc:.4f}")
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "extra_trees":
        model = _train_extra_trees_oracle(x_train, y_train, class_weight_sklearn, random_state=int(seed))
        val_pred = model.predict(x_val)
        val_acc = float(np.mean(val_pred == y_val))
        print(f"[Oracle] finished {name} ({model_type}) val_acc={val_acc:.4f}")
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    if model_type == "linear_svm":
        model = _train_linear_svm_oracle(x_train, y_train, class_weight_sklearn, random_state=int(seed))
        val_pred = model.predict(x_val)
        val_acc = float(np.mean(val_pred == y_val))
        print(f"[Oracle] finished {name} ({model_type}) val_acc={val_acc:.4f}")
        return OracleBundle(name=name, model=model, n_classes=n_classes, model_type=model_type), val_acc
    raise ValueError(f"Unknown oracle type: {model_type}")


class SafeLinearOracleModel:
    def __init__(
        self,
        *,
        model_type: str,
        coef: np.ndarray,
        intercept: np.ndarray,
        classes: np.ndarray,
    ) -> None:
        self.model_type = str(model_type)
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = np.asarray(intercept, dtype=np.float64)
        self.classes_ = np.asarray(classes)
        self.n_features_in_ = int(self.coef_.shape[1])

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x_np = np.asarray(x, dtype=np.float64)
        scores = x_np @ self.coef_.T + self.intercept_
        if scores.ndim == 2 and scores.shape[1] == 1:
            return scores[:, 0]
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        if np.asarray(scores).ndim == 1:
            positive_idx = 1 if self.classes_.size > 1 else 0
            negative_idx = 0
            return np.where(np.asarray(scores) >= 0.0, self.classes_[positive_idx], self.classes_[negative_idx])
        pred_idx = np.argmax(np.asarray(scores), axis=1)
        return self.classes_[pred_idx]

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        if np.asarray(scores).ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-np.asarray(scores, dtype=np.float64)))
            probs = np.stack([1.0 - probs_pos, probs_pos], axis=1)
            return probs.astype(np.float32)
        scores_np = np.asarray(scores, dtype=np.float64)
        scores_np = scores_np - np.max(scores_np, axis=1, keepdims=True)
        exp_scores = np.exp(scores_np)
        denom = np.sum(exp_scores, axis=1, keepdims=True) + 1.0e-12
        return (exp_scores / denom).astype(np.float32)


def serialize_safe_oracle_model(model: Any, model_type: str) -> dict[str, Any] | None:
    if model_type not in SAFE_LINEAR_ORACLE_TYPES:
        return None
    coef = getattr(model, "coef_", None)
    intercept = getattr(model, "intercept_", None)
    classes = getattr(model, "classes_", None)
    if coef is None or intercept is None or classes is None:
        raise ValueError(f"Oracle type '{model_type}' is missing fitted linear-model attributes.")
    return {
        "model_type": str(model_type),
        "coef": np.asarray(coef, dtype=np.float64).tolist(),
        "intercept": np.asarray(intercept, dtype=np.float64).tolist(),
        "classes": np.asarray(classes).tolist(),
    }


def restore_safe_oracle_model(payload: Dict[str, Any], model_type: str) -> SafeLinearOracleModel:
    payload_type = str(payload.get("model_type", model_type))
    return SafeLinearOracleModel(
        model_type=payload_type,
        coef=np.asarray(payload["coef"], dtype=np.float64),
        intercept=np.asarray(payload["intercept"], dtype=np.float64),
        classes=np.asarray(payload["classes"]),
    )


class OracleWrapper:
    def __init__(self, model: Any, model_type: str, device: torch.device):
        self.model = model
        self.model_type = model_type
        self.device = device
        if hasattr(self.model, "eval"):
            self.model.eval()

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.model_type in {"mlp", "cnn", "rnn", "lstm", "gru", "transformer"}:
            logits = self.model(x)
            return logits.argmax(dim=1)
        x_np = x.detach().cpu().numpy()
        preds = self.model.predict(x_np)
        return torch.tensor(preds, dtype=torch.long, device=self.device)


def predict_sklearn_probs(model: Any, x: np.ndarray) -> np.ndarray | None:
    x_np = np.asarray(x, dtype=np.float64)
    if x_np.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_np)
        return np.asarray(probs, dtype=np.float32)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_np), dtype=np.float64)
        if scores.ndim == 1:
            probs_pos = 1.0 / (1.0 + np.exp(-scores))
            probs = np.stack([1.0 - probs_pos, probs_pos], axis=1)
            return probs.astype(np.float32)
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        denom = np.sum(exp_scores, axis=1, keepdims=True) + 1.0e-12
        return (exp_scores / denom).astype(np.float32)
    return None
