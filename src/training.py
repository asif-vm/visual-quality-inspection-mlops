from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.data import SyntheticDefectDataset
from src.model import DefectCNN


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    model.eval()
    correct = total = true_positive = false_positive = false_negative = 0
    losses = []
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            losses.append(float(loss_fn(logits, labels)))
            predicted = logits.argmax(1)
            correct += int((predicted == labels).sum())
            total += len(labels)
            true_positive += int(((predicted == 1) & (labels == 1)).sum())
            false_positive += int(((predicted == 1) & (labels == 0)).sum())
            false_negative += int(((predicted == 0) & (labels == 1)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "loss": round(float(np.mean(losses)), 4),
        "accuracy": round(correct / max(total, 1), 4),
        "defect_precision": round(precision, 4),
        "defect_recall": round(recall, 4),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
    }


def train(output_dir: Path, samples: int = 2_000, epochs: int = 5, seed: int = 42) -> dict[str, float]:
    set_seed(seed)
    dataset = SyntheticDefectDataset(samples=samples, seed=seed)
    train_size = int(len(dataset) * 0.8)
    train_set, test_set = random_split(dataset, [train_size, len(dataset) - train_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128)
    model = DefectCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
    metrics = evaluate(model, test_loader)
    metrics.update({"train_samples": train_size, "test_samples": len(test_set), "epochs": epochs})
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "image_size": 32, "classes": ["normal", "defect"]}, output_dir / "model.pt")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    np.savez(output_dir / "reference_stats.npz", mean=float(dataset.images.mean()), std=float(dataset.images.std()))
    return metrics


def log_mlflow(metrics: dict[str, float], artifact_dir: Path) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise SystemExit("Install requirements-mlops.txt to enable MLflow") from exc
    mlflow.set_tracking_uri(f"sqlite:///{(artifact_dir.parent / 'mlflow.db').resolve().as_posix()}")
    mlflow.set_experiment("visual-quality-inspection")
    with mlflow.start_run(run_name="defect-cnn"):
        mlflow.log_params({"architecture": "two-block-cnn", "seed": 42})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifacts(str(artifact_dir), artifact_path="model_outputs")

