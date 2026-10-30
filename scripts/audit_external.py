"""Audita a referência externa sem gerar previsões ou treinar modelos."""

import argparse

import pandas as pd

from airline_ml.utils import sha256, utc_now, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--output", default="reports/local/external_reference_audit.json")
    args = parser.parse_args()
    frame = pd.read_csv(args.path)
    if "satisfaction" not in frame:
        raise ValueError("Referência sem coluna satisfaction.")
    features = frame.drop(columns="satisfaction")
    groups = pd.util.hash_pandas_object(features, index=False)
    conflicts = pd.DataFrame({"group": groups, "target": frame.satisfaction})
    result = {
        "observed_at_utc": utc_now(),
        "dataset_role": "external_reference",
        "sha256": sha256(args.path),
        "rows": len(frame),
        "columns": len(frame.columns),
        "target_counts": {str(k): int(v) for k, v in frame.satisfaction.value_counts().items()},
        "missing_by_column": {k: int(v) for k, v in frame.isna().sum().items()},
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_feature_rows": int(features.duplicated().sum()),
        "conflicting_label_groups": int(conflicts.groupby("group").target.nunique().gt(1).sum()),
        "numeric_ranges": {
            column: {"min": float(frame[column].min()), "max": float(frame[column].max())}
            for column in features.select_dtypes(include="number")
        },
        "delay_columns_values_over_five": {
            column: int(frame[column].gt(5).sum())
            for column in features
            if "Delay in Minutes" in column
        },
        "conclusion": "Referência externa auditada; não usada como treino oficial da competição.",
    }
    write_json(args.output, result)
    print(args.output)


if __name__ == "__main__":
    main()
