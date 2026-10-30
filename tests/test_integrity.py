import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airline_ml.config import load_config, validate_config
from airline_ml.data import encode_target, load_data
from airline_ml.folds import get_folds, validate_folds
from airline_ml.models import categorical_strings, prepare_fold
from airline_ml.pipeline import audit, train
from airline_ml.smoke import make_synthetic_data
from airline_ml.submission import validate_submission, write_submission
from airline_ml.utils import fingerprint


@pytest.fixture
def config(tmp_path):
    make_synthetic_data(tmp_path / "data")
    settings = load_config("configs/catboost.json")
    settings.update(
        {
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "artifacts"),
            "reports_dir": str(tmp_path / "reports"),
            "registry": str(tmp_path / "registry.csv"),
            "dataset_role": "synthetic_test",
            "folds_lock": None,
            "n_splits": 3,
            "early_stopping_rounds": 5,
        }
    )
    settings["params"].update(iterations=15, depth=3, thread_count=2)
    return settings


def test_submission_aligns_ids_and_checks_saved_file(tmp_path):
    ids = pd.Series(["001", "002", "003"], name="id", dtype="string")
    sample = pd.DataFrame({"id": ids.iloc[::-1].reset_index(drop=True), "satisfaction": 0.5})
    path = write_submission(ids, [0.1, 0.2, 0.9], sample, tmp_path / "submission.csv")
    saved = pd.read_csv(path, dtype={"id": "string"})
    assert saved.id.tolist() == ["003", "002", "001"]
    np.testing.assert_allclose(saved.satisfaction, [0.9, 0.2, 0.1])
    validate_submission(saved, sample, ids)


@pytest.mark.parametrize("values", [[np.nan, 0.1], [np.inf, 0.1], [-0.1, 1.1], [0.1]])
def test_submission_rejects_bad_probabilities(tmp_path, values):
    sample = pd.DataFrame({"id": pd.Series(["1", "2"], dtype="string"), "satisfaction": 0.5})
    with pytest.raises(ValueError):
        write_submission(sample.id, values, sample, tmp_path / "bad.csv")


@pytest.mark.parametrize("fault", ["duplicate_id", "wrong_id", "wrong_order", "extra_index"])
def test_submission_rejects_id_and_header_errors(fault):
    sample = pd.DataFrame({"id": pd.Series(["1", "2"], dtype="string"), "satisfaction": 0.5})
    bad = sample.copy()
    if fault == "duplicate_id":
        bad.loc[1, "id"] = "1"
    elif fault == "wrong_id":
        bad.loc[1, "id"] = "3"
    elif fault == "wrong_order":
        bad = bad.iloc[::-1]
    else:
        bad["Unnamed: 0"] = [0, 1]
    with pytest.raises(ValueError):
        validate_submission(bad, sample, sample.id)


def test_target_needs_explicit_text_mapping():
    labels = pd.Series(["satisfied", "neutral or dissatisfied"])
    with pytest.raises(ValueError):
        encode_target(labels)
    assert encode_target(labels, {"satisfied": 1, "neutral or dissatisfied": 0}).tolist() == [1, 0]
    with pytest.raises(ValueError):
        encode_target(pd.Series([0, 0]))


@pytest.mark.parametrize(
    "change",
    [
        {"n_splits": 1},
        {"seed": -1},
        {"validation": "temporal"},
        {"unknown": 1},
        {"model": "magic"},
        {"params": {"random_seed": 17}},
        {"group_column": "satisfaction"},
        {"categorical_columns": "Class"},
    ],
)
def test_config_rejects_incompatible_options(change):
    with pytest.raises(ValueError):
        validate_config(change)


def test_incompatible_data_and_missing_files(config):
    path = Path(config["data_dir"]) / "train.csv"
    frame = pd.read_csv(path)
    frame.rename(columns={"id": "item_id"}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="incompatível"):
        load_data(config)
    path.unlink()
    with pytest.raises(ValueError, match="ausente"):
        load_data(config)


def test_train_test_id_mismatch_is_rejected(config):
    path = Path(config["data_dir"]) / "sample_submission.csv"
    sample = pd.read_csv(path)
    sample.loc[0, "id"] = 999999
    sample.to_csv(path, index=False)
    with pytest.raises(ValueError, match="não correspondem"):
        load_data(config)


def test_folds_reused_across_models_and_row_order(config):
    bundle = load_data(config)
    folds, metadata = get_folds(bundle, config)
    changed_model = {**config, "model": "xgboost"}
    other, other_metadata = get_folds(bundle, changed_model)
    np.testing.assert_array_equal(folds, other)
    assert metadata["key"] == other_metadata["key"]
    table = pd.read_csv(metadata["path"], dtype={"id": "string"}).iloc[::-1]
    aligned = validate_folds(table, bundle["train"].id, bundle["y"].to_numpy(), 3)
    np.testing.assert_array_equal(aligned, folds)


def test_fold_cache_rejects_tampering(config):
    bundle = load_data(config)
    _, metadata = get_folds(bundle, config)
    path = Path(metadata["path"])
    path.write_text(path.read_text() + "999,0\n")
    with pytest.raises(ValueError, match="alterado"):
        get_folds(bundle, config)


def test_duplicates_stay_together(config):
    path = Path(config["data_dir"]) / "train.csv"
    frame = pd.read_csv(path)
    duplicate = frame.iloc[[0]].copy()
    duplicate["id"] = 99999
    frame = pd.concat([frame, duplicate], ignore_index=True)
    frame.to_csv(path, index=False)
    bundle = load_data(config)
    folds, metadata = get_folds(bundle, config)
    assert metadata["settings"]["strategy"] == "stratified_group"
    assert folds[0] == folds[-1]
    with pytest.raises(ValueError, match="Duplicatas"):
        get_folds(bundle, {**config, "validation": "stratified"})


def test_folds_reject_missing_oof_rows(config):
    bundle = load_data(config)
    _, metadata = get_folds(bundle, config)
    table = pd.read_csv(metadata["path"], dtype={"id": "string"}).iloc[1:]
    with pytest.raises(ValueError, match="incompleta"):
        validate_folds(table, bundle["train"].id, bundle["y"].to_numpy(), 3)


def test_categories_are_learned_from_training_only():
    fit = pd.DataFrame({"class": ["Eco", None], "x": [0, 1]})
    valid = pd.DataFrame({"class": ["Business"], "x": [2]})
    prepared = prepare_fold(fit, valid, valid, ["class"], "lightgbm")
    assert "V:Business" not in prepared[0].f0.cat.categories
    assert prepared[1].f0.isna().all()


def test_numeric_categories_have_consistent_representations():
    assert categorical_strings(pd.Series([1, 2])).tolist() == ["V:1", "V:2"]
    assert categorical_strings(pd.Series([1.0, np.nan])).tolist() == ["V:1", "M:"]
    assert categorical_strings(pd.Series(["M:", None])).tolist() == ["V:M:", "M:"]


def test_feature_changes_preserve_raw_data_folds(config):
    first_bundle = load_data(config)
    _, first = get_folds(first_bundle, config)
    changed = {**config, "drop_columns": ["rating"], "categorical_columns": ["Class"]}
    second_bundle = load_data(changed)
    _, second = get_folds(second_bundle, changed)
    assert first["key"] == second["key"]


def test_audit_writes_aggregates_and_plot(config):
    result = audit(config)
    folder = Path(result["report_path"]).parent
    assert result["train_rows"] == 240
    assert (folder / "audit.png").stat().st_size > 0
    assert (folder / "columns.csv").exists()


@pytest.mark.parametrize("model", ["catboost", "lightgbm", "xgboost"])
def test_pipeline_end_to_end(config, model):
    pytest.importorskip(model)
    original = config
    config = {
        **load_config(f"configs/{model}.json"),
        **{
            key: original[key]
            for key in (
                "data_dir",
                "output_dir",
                "reports_dir",
                "registry",
                "dataset_role",
                "n_splits",
                "folds_lock",
            )
        },
    }
    key = "iterations" if model == "catboost" else "n_estimators"
    config["params"][key] = 15
    config["params"]["thread_count" if model == "catboost" else "n_jobs"] = 2
    result = train(config)
    directory = Path(result["artifacts"])
    assert result["submission_state"] == "diagnostic_only"
    assert not (directory / "submission.csv").exists()
    assert result["public_score"] is None
    assert len(result["folds"]) == 3
    oof = pd.read_csv(directory / "oof.csv")
    assert len(oof) == 240 and oof.id.is_unique
    assert oof.prediction.notna().all()
    for fold in range(3):
        membership = json.loads((directory / f"preprocessing_fold_{fold}.json").read_text())
        fit_ids = oof.loc[oof.fold != fold, "id"].astype(str).tolist()
        valid_ids = oof.loc[oof.fold == fold, "id"].astype(str).tolist()
        assert not set(fit_ids) & set(valid_ids)
        assert membership["training_ids_sha256"] == fingerprint(fit_ids)
        assert membership["validation_ids_sha256"] == fingerprint(valid_ids)
    assert json.loads((directory / "metrics.json").read_text())["dataset_role"] == "synthetic_test"
    # Uma falha de invariantes, e não a qualidade de uma base artificial, deve reprovar o teste.
    assert 0 <= result["auc_oof"] <= 1


def test_competition_submission_and_pilot_are_separate(config):
    competition = {**config, "dataset_role": "competition"}
    result = train(competition)
    assert result["submission_state"] == "generated"
    assert Path(result["submission_path"]).name == "submission.csv"
    pilot = train(competition, pilot_rows=90)
    assert pilot["submission_state"] == "diagnostic_only"
    assert pilot["fold_key"] != result["fold_key"]
    assert Path(pilot["submission_path"]).name == "diagnostic_predictions.csv"
