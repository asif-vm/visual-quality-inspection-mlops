# Visual Quality Inspection MLOps

A reproducible PyTorch computer-vision system for classifying manufactured
surfaces as normal or defective. It adds deep learning and computer vision to
the portfolio without repeating the existing classical RUL model or FinSight
RAG application.

## What it demonstrates

- Deterministic, privacy-safe synthetic image generation
- An authored convolutional neural network in PyTorch
- Leakage-safe seeded train/test splitting
- Accuracy, precision, recall, F1, and loss reporting
- Optional MLflow experiment and artifact tracking
- FastAPI image inference with confidence and human-review routing
- Input-distribution drift monitoring
- Docker, Kubernetes manifests, pytest, and GitHub Actions CI

## Architecture

```text
Synthetic/public images -> PyTorch CNN -> model + metrics
                                |              |
                             MLflow         FastAPI
                                                |
                                      drift + review policy
```

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python train.py --samples 2000 --epochs 5
pytest -q
uvicorn api:app --reload
```

Open `http://127.0.0.1:8000` for the drag-and-drop inspection interface. Upload
a grayscale or color PNG/JPEG; the service converts it to a normalized 32x32
grayscale tensor and returns the predicted class, defect probability, and
whether a human review is required. The developer API remains available at
`http://127.0.0.1:8000/docs`.

## MLflow

Install `requirements-mlops.txt`, then run:

```bash
python train.py --samples 2000 --epochs 5 --mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Evidence boundaries

The bundled training workflow uses synthetic surfaces so it can run without
licenses, accounts, or private factory data. It proves the engineering and
evaluation workflow, not production accuracy on a real manufacturing line.
The next extension would benchmark against a licensed public dataset such as
MVTec AD without committing that dataset to Git.

## Repository map

- `src/data.py`: deterministic surface generator and Dataset
- `src/model.py`: CNN architecture
- `src/training.py`: training, evaluation, artifacts, and MLflow
- `api.py`: model-serving API
- `monitor.py`: distribution-shift report
- `k8s/`: deployment and readiness configuration
- `tests/`: shape, training, and artifact tests

