from pathlib import Path

import torch
from fastapi.testclient import TestClient

from src.data import SyntheticDefectDataset
from src.model import DefectCNN
from src.training import train
from api import app


def test_dataset_and_model_shapes() -> None:
    dataset = SyntheticDefectDataset(samples=12, seed=7)
    image, label = dataset[0]
    assert image.shape == (1, 32, 32)
    assert int(label) in {0, 1}
    logits = DefectCNN()(image.unsqueeze(0))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_training_writes_loadable_artifacts(tmp_path: Path) -> None:
    metrics = train(tmp_path, samples=96, epochs=1)
    assert 0 <= metrics["accuracy"] <= 1
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "metrics.json").exists()
    bundle = torch.load(tmp_path / "model.pt", map_location="cpu", weights_only=True)
    assert bundle["classes"] == ["normal", "defect"]


def test_homepage_has_human_friendly_upload() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert "Drop a PNG or JPEG here" in response.text
    assert "Inspect image" in response.text

