from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def drift_report(reference_path: Path, image_paths: list[Path], threshold: float = 0.20) -> dict:
    reference = np.load(reference_path)
    pixels = []
    for path in image_paths:
        image = Image.open(path).convert("L").resize((32, 32))
        pixels.append(np.asarray(image, dtype=np.float32) / 255.0)
    if not pixels:
        raise ValueError("At least one image is required")
    current_mean = float(np.mean(pixels))
    current_std = float(np.std(pixels))
    reference_std = max(float(reference["std"]), 1e-6)
    standardized_mean_shift = abs(current_mean - float(reference["mean"])) / reference_std
    return {
        "reference_mean": round(float(reference["mean"]), 4),
        "current_mean": round(current_mean, 4),
        "standardized_mean_shift": round(standardized_mean_shift, 4),
        "threshold": threshold,
        "drift_detected": standardized_mean_shift >= threshold,
        "current_std": round(current_std, 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--threshold", type=float, default=0.20)
    args = parser.parse_args()
    report = drift_report(Path("artifacts/reference_stats.npz"), args.images, args.threshold)
    Path("artifacts/drift_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

