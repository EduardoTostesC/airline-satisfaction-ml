import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Reutiliza a fixture de dados artificiais; nenhum CSV real entra nos testes.
from test_integrity import config  # noqa: F401

from airline_ml.data import load_data
from airline_ml.folds import get_folds
from scripts.audit_baseline import align_oof


def freeze(settings):
    bundle = load_data(settings)
    values, meta = get_folds(bundle, settings)
    manifest = Path(settings["output_dir"]) / "test-lock.json"
    manifest.write_text(
        json.dumps({k: meta[k] for k in ("settings", "key", "folds_sha256")}), encoding="utf-8"
    )
    return bundle, values, meta, {**settings, "folds_lock": str(manifest)}


def test_frozen_folds_are_reused(config):  # noqa: F811
    bundle, original, _, locked = freeze(config)
    other, _ = get_folds(bundle, {**locked, "model": "xgboost"})
    np.testing.assert_array_equal(original, other)


def test_frozen_folds_reject_new_seed_without_creating_split(config):  # noqa: F811
    bundle, _, _, locked = freeze(config)
    before = set(Path(config["output_dir"]).rglob("folds.csv"))
    with pytest.raises(ValueError, match="congelados"):
        get_folds(bundle, {**locked, "seed": config["seed"] + 1})
    assert set(Path(config["output_dir"]).rglob("folds.csv")) == before


def test_frozen_folds_reject_missing_or_modified_csv(config):  # noqa: F811
    bundle, _, meta, locked = freeze(config)
    path = Path(meta["path"])
    original = path.read_bytes()
    path.write_bytes(original + b"999,0\n")
    with pytest.raises(ValueError, match="alterado"):
        get_folds(bundle, locked)
    path.unlink()
    with pytest.raises(ValueError, match="ausentes"):
        get_folds(bundle, locked)
    assert not path.exists()


def test_fold_metadata_key_tampering_is_rejected(config):  # noqa: F811
    bundle, _, meta, _ = freeze(config)
    path = Path(meta["path"]).with_name("metadata.json")
    value = json.loads(path.read_text())
    value["key"] = "wrong-key"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="alterado"):
        get_folds(bundle, config)


def oof_sample():
    ids = pd.Series(["1", "2", "3", "4"], dtype="string", name="id")
    y, folds = np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])
    frame = pd.DataFrame(
        {"id": ids, "target": y, "fold": folds, "prediction": [0.1, 0.9, 0.3, 0.8]}
    )
    return ids, y, folds, frame


def test_oof_alignment_uses_ids():
    ids, y, folds, frame = oof_sample()
    result = align_oof(frame.iloc[::-1], ids, y, folds)
    np.testing.assert_array_equal(result.prediction, frame.prediction)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "id", "target", "fold", "nan", "range"])
def test_oof_audit_rejects_corruption(fault):
    ids, y, folds, frame = oof_sample()
    if fault == "missing":
        frame = frame.iloc[:-1]
    elif fault == "duplicate":
        frame.loc[1, "id"] = "1"
    elif fault == "id":
        frame.loc[1, "id"] = "999"
    elif fault == "target":
        frame.loc[1, "target"] = 0
    elif fault == "fold":
        frame.loc[1, "fold"] = 0
    elif fault == "nan":
        frame.loc[1, "prediction"] = np.nan
    else:
        frame.loc[1, "prediction"] = 1.1
    with pytest.raises(ValueError):
        align_oof(frame, ids, y, folds)
