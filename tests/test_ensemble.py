import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airline_ml.utils import sha256
from scripts.ensemble_campaign import combine


def members(tmp_path):
    sample = pd.DataFrame({"id": ["10", "11"], "satisfaction": [0.5, 0.5]})
    (tmp_path / "data/raw").mkdir(parents=True)
    sample.to_csv(tmp_path / "data/raw/sample_submission.csv", index=False)
    paths = []
    for index in range(2):
        folder = tmp_path / f"member{index}"
        folder.mkdir()
        oof = pd.DataFrame(
            {
                "id": ["1", "2", "3", "4"],
                "target": [0, 1, 0, 1],
                "fold": [0, 0, 1, 1],
                "prediction": [0.1, 0.8, 0.4, 0.9],
            }
        )
        if index:
            oof = oof.iloc[::-1]
        oof.to_csv(folder / "oof.csv", index=False, float_format="%.17g")
        sample.assign(satisfaction=[0.2, 0.7]).to_csv(
            folder / "submission.csv", index=False, float_format="%.17g"
        )
        metrics = {
            "experiment_id": folder.name,
            "evaluation_scope": "full_cv",
            "fold_key": "same-folds",
            "data_hashes": {"train": "same-data"},
            "train_sha256": "same-data",
            "oof_sha256": sha256(folder / "oof.csv"),
            "submission_sha256": sha256(folder / "submission.csv"),
        }
        (folder / "metrics.json").write_text(json.dumps(metrics))
        paths.append(str(folder))
    return paths


def test_ensemble_aligns_ids_and_preserves_equal_members(tmp_path, monkeypatch):
    paths = members(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = combine("camp_ensemble_test", paths, [0.5, 0.5])
    oof = pd.read_csv(Path(result["artifacts"]) / "oof.csv")
    np.testing.assert_allclose(oof.prediction, [0.1, 0.8, 0.4, 0.9])
    assert result["auc_oof"] == 1
    assert result["resources"]["peak_rss_bytes"] > 0


def test_ensemble_rejects_different_folds(tmp_path, monkeypatch):
    paths = members(tmp_path)
    path = Path(paths[1]) / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["fold_key"] = "other-folds"
    path.write_text(json.dumps(metrics))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="folds diferentes"):
        combine("camp_ensemble_bad", paths, [0.5, 0.5])


@pytest.mark.parametrize("weights", [[0.2, 0.2], [1.1, -0.1], [1.0]])
def test_bad_ensemble_weights_are_rejected(tmp_path, monkeypatch, weights):
    paths = members(tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        combine("camp_ensemble_bad_weights", paths, weights)
