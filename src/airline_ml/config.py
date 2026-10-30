"""Configurações explícitas e rejeição de opções ambíguas."""

import json
import re
from pathlib import Path

DEFAULTS = {
    "name": "experiment",
    "hypothesis": "Baseline",
    "data_dir": "data/raw",
    "output_dir": "artifacts",
    "reports_dir": "reports/local",
    "registry": "reports/experiments.csv",
    "dataset_role": "competition",
    "id_column": "id",
    "target_column": "satisfaction",
    "target_mapping": None,
    "categorical_columns": None,
    "drop_columns": [],
    "seed": 2026,
    "n_splits": 5,
    "validation": "auto",
    "group_column": None,
    "folds_lock": None,
    "model": "catboost",
    "early_stopping_rounds": 100,
    "params": {},
}


def validate_config(config):
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Opções desconhecidas: {sorted(unknown)}")
    result = {**DEFAULTS, **config}
    if not isinstance(result["name"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", result["name"]):
        raise ValueError("name deve conter apenas letras ASCII, números, _ e -.")
    for key in ("data_dir", "output_dir", "reports_dir", "registry", "id_column", "target_column"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"{key} deve ser texto não vazio.")
    if result["id_column"] == result["target_column"]:
        raise ValueError("ID e alvo devem ser colunas distintas.")
    for key in ("seed", "n_splits", "early_stopping_rounds"):
        if type(result[key]) is not int:
            raise ValueError(f"{key} deve ser inteiro.")
    if result["seed"] < 0 or result["n_splits"] < 2 or result["early_stopping_rounds"] < 1:
        raise ValueError("seed >= 0, n_splits >= 2 e early_stopping_rounds >= 1 são obrigatórios.")
    if result["validation"] not in {"auto", "stratified", "stratified_group"}:
        raise ValueError(
            "Estratégia de validação não implementada; tempo exige desenho específico."
        )
    if result["model"] not in {"catboost", "lightgbm", "xgboost"}:
        raise ValueError("Modelo não suportado.")
    if result["dataset_role"] not in {"competition", "synthetic_test", "external_reference"}:
        raise ValueError("dataset_role inválido.")
    for key in ("drop_columns", "categorical_columns"):
        value = result[key]
        if key == "categorical_columns" and value is None:
            continue
        if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
            raise ValueError(f"{key} deve ser lista de nomes de colunas.")
    if result["group_column"] is not None and not isinstance(result["group_column"], str):
        raise ValueError("group_column deve ser nome de coluna ou null.")
    if result["group_column"] in (result["id_column"], result["target_column"]):
        raise ValueError("group_column não pode ser ID ou alvo.")
    if result["folds_lock"] is not None and (
        not isinstance(result["folds_lock"], str) or not result["folds_lock"].strip()
    ):
        raise ValueError("folds_lock deve ser um caminho de manifesto ou null.")
    mapping = result["target_mapping"]
    if mapping is not None and (not isinstance(mapping, dict) or set(mapping.values()) != {0, 1}):
        raise ValueError("target_mapping deve mapear rótulos textuais para 0 e 1.")
    params = result["params"]
    if not isinstance(params, dict):
        raise ValueError("params deve ser um objeto JSON.")
    reserved = {
        "random_seed",
        "random_state",
        "seed",
        "cat_features",
        "early_stopping_rounds",
        "use_best_model",
        "allow_writing_files",
        "verbose",
        "enable_categorical",
    }
    if set(params) & reserved:
        raise ValueError("Seed, early stopping e opções de execução são controlados pelo pipeline.")
    return result


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8")))
