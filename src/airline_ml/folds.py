"""Folds persistidos por ID, vinculados ao hash do treino e à estratégia."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .data import feature_groups
from .utils import fingerprint, sha256, write_json


def validate_folds(table, ids, y, n_splits, groups=None):
    if list(table.columns) != [ids.name, "fold"]:
        raise ValueError("Colunas de folds incompatíveis.")
    if table[ids.name].isna().any() or table[ids.name].duplicated().any():
        raise ValueError("Folds contêm IDs ausentes ou repetidos.")
    if len(table) != len(ids) or set(table[ids.name]) != set(ids):
        raise ValueError("Cobertura de IDs dos folds incompleta.")
    aligned = table.set_index(ids.name).loc[ids, "fold"]
    values = pd.to_numeric(aligned, errors="coerce").to_numpy()
    if not np.isfinite(values).all() or not np.equal(values, values.astype(int)).all():
        raise ValueError("Folds devem ser inteiros finitos.")
    folds = values.astype(int)
    if set(folds) != set(range(n_splits)):
        raise ValueError("Folds fora do intervalo esperado.")
    for fold in range(n_splits):
        if len(np.unique(y[folds == fold])) != 2 or len(np.unique(y[folds != fold])) != 2:
            raise ValueError("Cada treino e validação precisa conter as duas classes.")
    if groups is not None:
        if (
            pd.DataFrame({"group": groups, "fold": folds})
            .groupby("group")
            .fold.nunique()
            .gt(1)
            .any()
        ):
            raise ValueError("Um grupo foi dividido entre folds: risco de vazamento.")
    return folds


def get_folds(bundle, config):
    train, y = bundle["train"], bundle["y"].to_numpy()
    ids = train[config["id_column"]]
    exact_groups = feature_groups(bundle)
    duplicate_rows = pd.Series(exact_groups).duplicated().any()
    group_column = config["group_column"]
    groups = train[group_column].astype(str).to_numpy() if group_column else exact_groups
    use_groups = bool(group_column) or bool(duplicate_rows)
    if config["validation"] == "stratified" and use_groups:
        raise ValueError("Duplicatas ou grupos encontrados: use auto ou stratified_group.")
    strategy = "stratified_group" if use_groups else "stratified"
    if config["validation"] == "stratified_group":
        strategy = "stratified_group"
    n_splits = config["n_splits"]
    if np.bincount(y).min() < n_splits:
        raise ValueError("Quantidade de folds maior que a classe minoritária.")
    settings = {
        "train_sha256": bundle["hashes"]["train"],
        "n_splits": n_splits,
        "seed": config["seed"],
        "strategy": strategy,
        "group_column": group_column,
        "target_mapping": config["target_mapping"],
        "id_column": config["id_column"],
        "target_column": config["target_column"],
    }
    key = fingerprint(settings)
    folder = Path(config["output_dir"]) / "folds" / key
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, metadata_path = folder / "folds.csv", folder / "metadata.json"
    checked_groups = groups if strategy == "stratified_group" else None
    if csv_path.exists() or metadata_path.exists():
        if not csv_path.exists() or not metadata_path.exists():
            raise ValueError("Cache de folds incompleto; investigue antes de recriar.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("settings") != settings or metadata.get("folds_sha256") != sha256(csv_path):
            raise ValueError("Cache de folds alterado ou incompatível.")
        table = pd.read_csv(csv_path, dtype={ids.name: "string"})
    else:
        folds = np.full(len(train), -1, dtype=int)
        splitter = (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=config["seed"])
            if strategy == "stratified_group"
            else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config["seed"])
        )
        for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y, checked_groups)):
            folds[valid] = fold
        table = pd.DataFrame({ids.name: ids, "fold": folds})
        validate_folds(table, ids, y, n_splits, checked_groups)
        table.to_csv(csv_path, index=False)
        metadata = {"settings": settings, "folds_sha256": sha256(csv_path), "key": key}
        write_json(metadata_path, metadata)
    folds = validate_folds(table, ids, y, n_splits, checked_groups)
    return folds, {**metadata, "path": str(csv_path)}
