"""Teste funcional artificial, explicitamente separado dos resultados Kaggle."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config
from .pipeline import train


def make_synthetic_data(folder, rows=240, seed=2026):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = rng.normal(size=rows + 31)
    categories = rng.choice(["Eco", "Business", None], size=rows + 31)
    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(rows + 31)],
            "distance": x,
            "Class": categories,
            "rating": rng.integers(0, 6, size=rows + 31).astype(float),
        }
    )
    frame.loc[[4, 17, rows + 3], "distance"] = np.nan
    train_frame, test_frame = frame.iloc[:rows].copy(), frame.iloc[rows:].copy()
    train_frame["satisfaction"] = (x[:rows] + rng.normal(scale=0.6, size=rows) > 0).astype(int)
    sample = pd.DataFrame({"id": test_frame["id"].iloc[::-1], "satisfaction": 0.5})
    train_frame.to_csv(folder / "train.csv", index=False)
    test_frame.to_csv(folder / "test.csv", index=False)
    sample.to_csv(folder / "sample_submission.csv", index=False)


def run_smoke(folder="artifacts/smoke", model="catboost"):
    folder = Path(folder)
    make_synthetic_data(folder / "data")
    config = load_config(f"configs/{model}.json")
    iteration_key = "iterations" if model == "catboost" else "n_estimators"
    config.update(
        {
            "name": f"smoke_{model}",
            "hypothesis": "Verificar o pipeline em dados artificiais.",
            "data_dir": str(folder / "data"),
            "output_dir": str(folder),
            "reports_dir": str(folder / "reports"),
            "registry": str(folder / "experiments.csv"),
            "dataset_role": "synthetic_test",
            "n_splits": 3,
            "early_stopping_rounds": 8,
        }
    )
    config["params"][iteration_key] = 30
    config["params"]["thread_count" if model == "catboost" else "n_jobs"] = 2
    return train(config)
