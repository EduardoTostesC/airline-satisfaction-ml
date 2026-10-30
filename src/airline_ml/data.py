"""Leitura estrita, alinhamento e auditoria dos arquivos reais."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import sha256


def read_csv(path, id_column):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Arquivo ausente: {path}. Obtenha os CSVs oficiais da S6E10.")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"Cabeçalho vazio ou com colunas duplicadas: {path}")
    if id_column not in header:
        raise ValueError(f"{path}: coluna {id_column!r} ausente; dataset incompatível.")
    return pd.read_csv(path, dtype={id_column: "string"})


def validate_ids(frame, id_column, name):
    ids = frame[id_column]
    if frame.empty or ids.isna().any() or ids.str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError(f"{name}: IDs ausentes, repetidos ou arquivo vazio.")


def encode_target(series, mapping=None):
    if series.isna().any():
        raise ValueError("O alvo contém valores ausentes.")
    if mapping is not None:
        if set(mapping.values()) != {0, 1}:
            raise ValueError("target_mapping deve representar as classes 0 e 1.")
        encoded = series.astype(str).map(mapping)
    else:
        encoded = pd.to_numeric(series, errors="coerce")
    if encoded.isna().any() or set(encoded.unique()) != {0, 1}:
        raise ValueError("Alvo deve conter 0 e 1; configure target_mapping para rótulos textuais.")
    return encoded.astype("int8")


def load_data(config):
    folder = Path(config["data_dir"])
    ident, target = config["id_column"], config["target_column"]
    paths = {name: folder / f"{name}.csv" for name in ("train", "test", "sample_submission")}
    frames = {name: read_csv(path, ident) for name, path in paths.items()}
    train, test, sample = (frames[name] for name in ("train", "test", "sample_submission"))
    for name, frame in frames.items():
        validate_ids(frame, ident, name)
    if target not in train or target in test:
        raise ValueError("O alvo deve existir apenas no treino, além do sample.")
    if list(sample.columns) != [ident, target]:
        raise ValueError(f"Cabeçalho do sample deve ser exatamente {ident},{target}.")
    if set(train.columns) - {target} != set(test.columns):
        raise ValueError("Features de treino e teste são incompatíveis.")
    if set(test[ident]) != set(sample[ident]):
        raise ValueError("IDs do teste e do sample não correspondem.")
    if set(train[ident]) & set(test[ident]):
        raise ValueError("Há sobreposição de IDs entre treino e teste.")
    y = encode_target(train[target], config["target_mapping"])
    group_features = [column for column in train if column not in {ident, target}]
    # Identidade das linhas independe da seleção e transformação de features.
    raw_frames = [frame[group_features].copy() for frame in (train, test)]
    for column in group_features:
        if pd.api.types.is_numeric_dtype(train[column]):
            for frame in raw_frames:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
    raw_hashes = [pd.util.hash_pandas_object(frame, index=False).to_numpy() for frame in raw_frames]
    del raw_frames
    excluded = {ident, target, *config["drop_columns"]}
    group = config["group_column"]
    if group:
        if group not in train or train[group].isna().any():
            raise ValueError("group_column ausente ou com nulos.")
        excluded.add(group)
    missing_drop = set(config["drop_columns"]) - set(train.columns)
    if missing_drop or ident in config["drop_columns"] or target in config["drop_columns"]:
        raise ValueError("drop_columns contém colunas ausentes ou reservadas.")
    features = [column for column in train if column not in excluded]
    if not features:
        raise ValueError("Nenhuma feature disponível.")
    configured_cats = config["categorical_columns"]
    categorical = (
        list(train[features].select_dtypes(include=["object", "string", "category"]).columns)
        if configured_cats is None
        else list(configured_cats)
    )
    if set(categorical) - set(features) or len(set(categorical)) != len(categorical):
        raise ValueError("categorical_columns contém colunas ausentes, repetidas ou reservadas.")
    for column in set(features) - set(categorical):
        for frame in (train, test):
            try:
                numeric = pd.to_numeric(frame[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Feature {column}: tipo incompatível com numérica.") from exc
            if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
                raise ValueError(f"Feature {column}: infinito não permitido.")
            frame[column] = numeric.astype("float64")
    return {
        "train": train,
        "test": test,
        "sample": sample,
        "y": y,
        "features": features,
        "categorical": categorical,
        "group_features": group_features,
        "exact_groups": raw_hashes[0],
        "test_exact_groups": raw_hashes[1],
        "hashes": {name: sha256(path) for name, path in paths.items()},
    }


def feature_groups(bundle):
    return bundle["exact_groups"].astype(str)


def audit_data(bundle, config):
    train, test, y = bundle["train"], bundle["test"], bundle["y"]
    features, categorical = bundle["features"], bundle["categorical"]
    groups = feature_groups(bundle)
    test_hash = bundle["test_exact_groups"].astype(str)
    labels_by_group = pd.DataFrame({"group": groups, "y": y}).groupby("group")["y"].nunique()
    columns = []
    for column in features:
        left, right = train[column], test[column]
        row = {
            "column": column,
            "kind": "categorical" if column in categorical else "numeric_or_ordinal",
            "train_dtype": str(left.dtype),
            "test_dtype": str(right.dtype),
            "train_missing": int(left.isna().sum()),
            "test_missing": int(right.isna().sum()),
            "train_unique": int(left.nunique()),
            "test_unique": int(right.nunique()),
            "train_zero_count": int(left.eq(0).sum()),
            "test_zero_count": int(right.eq(0).sum()),
        }
        if column in categorical:
            known = set(left.dropna().astype(str))
            row["test_unseen_count"] = int((right.notna() & ~right.astype(str).isin(known)).sum())
            p = left.fillna("<MISSING>").astype(str).value_counts(normalize=True)
            q = right.fillna("<MISSING>").astype(str).value_counts(normalize=True)
            support = p.index.union(q.index)
            row["total_variation"] = float(
                (p.reindex(support, fill_value=0) - q.reindex(support, fill_value=0)).abs().sum()
                / 2
            )
        else:
            for name, series in (("train", left), ("test", right)):
                for quantile, value in series.quantile([0, 0.25, 0.5, 0.75, 1]).items():
                    row[f"{name}_q{quantile}"] = None if pd.isna(value) else float(value)
        columns.append(row)
    return {
        "hashes": bundle["hashes"],
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(features),
        "target_counts": {str(k): int(v) for k, v in y.value_counts().items()},
        "positive_rate": float(y.mean()),
        "duplicate_feature_rows": int(pd.Series(groups).duplicated().sum()),
        "conflicting_label_groups": int(labels_by_group.gt(1).sum()),
        "test_rows_matching_train_features": int(np.isin(test_hash, groups).sum()),
        "columns": columns,
        "decisions": ["id excluído", "zeros preservados", "sem remoção de outliers"],
        "limitations": [
            "Hashes identificam duplicatas exatas; grupos e tempo exigem revisão de domínio.",
            "Números ordinais não são inferidos semanticamente pelo tipo numérico.",
        ],
    }
