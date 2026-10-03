import argparse
from pathlib import Path

from src.training import log_mlflow, train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()
    artifacts = Path(__file__).parent / "artifacts"
    metrics = train(artifacts, args.samples, args.epochs)
    if args.mlflow:
        log_mlflow(metrics, artifacts)
    print(metrics)

