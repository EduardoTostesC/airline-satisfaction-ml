import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_integrity import config  # noqa: F401

from airline_ml.features import EXPERIENCE, SERVICE_COLUMNS, cross, transform
from airline_ml.pipeline import train


def test_counts_distinguish_zero_low_missing_and_baggage():
    frame = pd.DataFrame(np.nan, index=range(3), columns=SERVICE_COLUMNS)
    frame.loc[0, SERVICE_COLUMNS[:3]] = [0, 1, 5]
    frame.loc[0, "Baggage handling"] = 1
    frame.loc[1, "Baggage handling"] = 5
    result, _ = transform(frame, ["rating_counts"])
    assert result.loc[0, "fe_ratings_observed"] == 4
    assert result.loc[0, "fe_rating_count_zero"] == 1
    assert result.loc[0, "fe_rating_count_low_1_2"] == 2
    assert result.loc[0, "fe_rating_count_high_4_5"] == 1
    assert result.loc[1, "fe_rating_count_zero"] == 0
    assert pd.isna(result.loc[2, "fe_rating_count_zero"])


def test_delay_and_difference_missing_values_are_preserved():
    frame = pd.DataFrame(
        {"Departure Delay in Minutes": [0, 10, np.nan], "Arrival Delay in Minutes": [0, np.nan, 20]}
    )
    result, _ = transform(frame, ["delays"])
    assert result.loc[0, "fe_arrival_delayed"] == 0
    assert pd.isna(result.loc[1, "fe_arrival_delayed"])
    assert pd.isna(result.loc[1, "fe_arrival_minus_departure"])
    np.testing.assert_allclose(result.loc[1, "fe_departure_log1p"], np.log1p(10))


def test_summary_preserves_zero_and_empty_rows():
    frame = pd.DataFrame([[0, 2, np.nan, 4], [np.nan] * 4], columns=EXPERIENCE)
    result, _ = transform(frame, ["rating_summary", "rating_differences"])
    assert result.loc[0, "fe_experience_mean"] == 2
    assert result.loc[0, "fe_experience_min"] == 0
    assert pd.isna(result.loc[1, "fe_experience_mean"])
    assert pd.isna(result.loc[1, "fe_experience_agreements"])


def test_cross_is_row_deterministic_and_handles_missing():
    frame = pd.DataFrame({"a": ["x|y", "x", None], "b": ["z", "y|z", "z"]})
    first, _ = cross(frame, ["a", "b"]), cross(frame.iloc[::-1], ["a", "b"])
    assert first.iloc[0] != first.iloc[1]
    assert pd.isna(first.iloc[2])
    assert first.iloc[0] == cross(frame.iloc[[0]], ["a", "b"]).iloc[0]


def test_screening_uses_original_fold_and_never_promotes(config):  # noqa: F811
    full = train(config)
    screening = train(config, folds_to_run=[0])
    assert screening["fold_key"] == full["fold_key"]
    assert screening["auc_oof"] is None
    assert screening["evaluation_scope"] == "screening"
    assert screening["submission_state"] == "diagnostic_only"
    assert screening["oof_rows"] == 80
    assert screening["resources"]["peak_rss_bytes"] > 0
    oof = pd.read_csv(screening["oof_path"])
    assert set(oof.fold) == {0}
    assert not (Path(screening["artifacts"]) / "submission.csv").exists()
    assert (
        json.loads((Path(screening["artifacts"]) / "metrics.json").read_text())["auc_oof"] is None
    )


@pytest.mark.parametrize("folds", [[9], [], [0, 0]])
def test_bad_screen_fold_selection_is_rejected(config, folds):  # noqa: F811
    with pytest.raises(ValueError):
        train(config, folds_to_run=folds)
