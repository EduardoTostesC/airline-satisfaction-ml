"""Hipóteses pequenas e determinísticas, calculadas somente por linha."""

import json

import numpy as np
import pandas as pd

SERVICE_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
EXPERIENCE = ["Seat comfort", "Inflight entertainment", "Cleanliness", "On-board service"]
SEGMENTS = ["Customer Type", "Type of Travel", "Class"]
GROUPS = {
    "delays",
    "rating_counts",
    "rating_summary",
    "rating_differences",
    "segments",
    "service_segments",
}


def cross(frame, columns):
    # Factorização só evita repetir a serialização: o nome final depende dos valores,
    # não dos códigos/frequências. O vocabulário do modelo continua restrito ao fold.
    tuples = pd.MultiIndex.from_frame(frame[columns])
    codes, unique = pd.factorize(tuples)
    names = np.array(
        [
            json.dumps(list(values), ensure_ascii=False)
            if not any(pd.isna(v) for v in values)
            else np.nan
            for values in unique
        ],
        dtype=object,
    )
    values = names[codes]
    return pd.Series(pd.Categorical(values), index=frame.index)


def transform(frame, groups):
    frame = frame.copy()
    added_cats = []
    if "delays" in groups:
        for column, suffix in [
            ("Departure Delay in Minutes", "departure"),
            ("Arrival Delay in Minutes", "arrival"),
        ]:
            value = frame[column]
            if value.lt(0).any():
                raise ValueError("log1p de atrasos requer valores não negativos.")
            frame[f"fe_{suffix}_delayed"] = value.gt(0).astype(float).where(value.notna())
            frame[f"fe_{suffix}_log1p"] = np.log1p(value)
        frame["fe_arrival_minus_departure"] = (
            frame["Arrival Delay in Minutes"] - frame["Departure Delay in Minutes"]
        )
    if "rating_counts" in groups:
        values = frame[SERVICE_COLUMNS]
        count = values.notna().sum(axis=1)
        frame["fe_ratings_observed"] = count
        for name, condition in [
            ("zero", values.eq(0)),
            ("low_1_2", values.ge(1) & values.le(2)),
            ("high_4_5", values.ge(4) & values.le(5)),
        ]:
            frame[f"fe_rating_count_{name}"] = (
                condition.sum(axis=1).astype(float).where(count.gt(0))
            )
    if "rating_summary" in groups:
        values = frame[EXPERIENCE]
        frame["fe_experience_observed"] = values.notna().sum(axis=1)
        frame["fe_experience_mean"] = values.mean(axis=1)
        frame["fe_experience_min"] = values.min(axis=1)
        frame["fe_experience_std"] = values.std(axis=1, ddof=0)
    if "rating_differences" in groups:
        pairs = [
            ("Seat comfort", "Inflight entertainment"),
            ("Seat comfort", "Cleanliness"),
            ("On-board service", "Cleanliness"),
        ]
        observed, agreements = [], []
        for index, (left, right) in enumerate(pairs):
            frame[f"fe_experience_difference_{index}"] = frame[left] - frame[right]
            known = frame[left].notna() & frame[right].notna()
            observed.append(known)
            agreements.append(frame[left].eq(frame[right]) & known)
        count = pd.concat(observed, axis=1).sum(axis=1)
        frame["fe_experience_pairs_observed"] = count
        frame["fe_experience_agreements"] = (
            pd.concat(agreements, axis=1).sum(axis=1).astype(float).where(count.gt(0))
        )
    if "segments" in groups:
        for name, columns in [
            ("customer_travel", SEGMENTS[:2]),
            ("travel_class", SEGMENTS[1:]),
            ("customer_travel_class", SEGMENTS),
        ]:
            feature = f"fe_{name}"
            frame[feature] = cross(frame, columns)
            added_cats.append(feature)
    if "service_segments" in groups:
        for index, columns in enumerate(
            [
                ["Inflight wifi service", "Type of Travel"],
                ["Online boarding", "Customer Type"],
                ["Seat comfort", "Class"],
            ]
        ):
            feature = f"fe_service_segment_{index}"
            frame[feature] = cross(frame, columns)
            added_cats.append(feature)
    return frame, added_cats


def apply_features(bundle, config):
    groups = config["feature_groups"]
    if not groups and config["service_representation"] == "ordinal":
        return bundle
    if set(groups) - GROUPS:
        raise ValueError("Grupo de features desconhecido.")
    required = set(SERVICE_COLUMNS) if config["service_representation"] == "categorical" else set()
    if "rating_counts" in groups:
        required.update(SERVICE_COLUMNS)
    if set(groups) & {"rating_summary", "rating_differences"}:
        required.update(EXPERIENCE)
    if set(groups) & {"segments", "service_segments"}:
        required.update(SEGMENTS)
    if "service_segments" in groups:
        required.update(["Inflight wifi service", "Online boarding", "Seat comfort"])
    if "delays" in groups:
        required.update(["Departure Delay in Minutes", "Arrival Delay in Minutes"])
    if required - set(bundle["features"]):
        raise ValueError(
            f"Features necessárias ausentes: {sorted(required - set(bundle['features']))}"
        )
    result = {**bundle}
    result["train"], cats = transform(bundle["train"], groups)
    result["test"], test_cats = transform(bundle["test"], groups)
    if cats != test_cats:
        raise ValueError("Esquemas de engenharia divergentes.")
    additions = [c for c in result["train"] if c not in bundle["train"].columns]
    result["features"] = [*bundle["features"], *additions]
    result["categorical"] = list(
        dict.fromkeys(
            [
                *bundle["categorical"],
                *cats,
                *(SERVICE_COLUMNS if config["service_representation"] == "categorical" else []),
            ]
        )
    )
    return result
