"""Auditoria, validação cruzada e artefatos identificáveis por execução."""

import csv
import gc
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import validate_config
from .data import audit_data, load_data
from .features import apply_features
from .folds import get_folds
from .models import fit_fold, prepare_fold
from .resources import profile_run
from .submission import validate_probabilities, write_submission
from .utils import fingerprint, provenance, sha256, utc_now, write_json

REGISTRY_FIELDS = [
    "experiment_id",
    "date_utc",
    "dataset_role",
    "model",
    "hypothesis",
    "commit",
    "dirty",
    "code_sha256",
    "train_sha256",
    "fold_key",
    "seed",
    "auc_oof",
    "auc_fold_mean",
    "auc_fold_std",
    "duration_seconds",
    "submission_state",
    "artifacts",
    "public_score",
]


def audit(config):
    config = validate_config(config)
    bundle = load_data(config)
    report = audit_data(bundle, config)
    folder = Path(config["reports_dir"]) / bundle["hashes"]["train"][:16]
    write_json(folder / "audit.json", report)
    pd.DataFrame(report["columns"]).to_csv(folder / "columns.csv", index=False)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    counts = bundle["y"].value_counts().sort_index()
    axes[0].bar(counts.index.astype(str), counts.values)
    axes[0].set(title="Distribuição do alvo no treino", xlabel="Classe", ylabel="Registros")
    missing = pd.DataFrame(report["columns"]).set_index("column")
    missing[["train_missing", "test_missing"]].div([report["train_rows"], report["test_rows"]]).max(
        axis=1
    ).nlargest(8).sort_values().plot.barh(ax=axes[1])
    axes[1].set(title="Maiores proporções de nulos", xlabel="Máximo entre treino e teste")
    fig.tight_layout()
    fig.savefig(folder / "audit.png", dpi=150)
    plt.close(fig)
    return {"report_path": str(folder / "audit.json"), **report}


def register_run(config, summary):
    path = Path(config["registry"])
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle), None) != REGISTRY_FIELDS:
                raise ValueError("Cabeçalho do registro de experimentos incompatível.")
    row = {key: summary.get(key, "") for key in REGISTRY_FIELDS}
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


@profile_run
def train(config, pilot_rows=None, folds_to_run=None):
    config = validate_config(config)
    started = time.perf_counter()
    bundle = load_data(config)
    original_hash = bundle["hashes"]["train"]
    if pilot_rows is not None:
        if type(pilot_rows) is not int or pilot_rows < config["n_splits"] * 2:
            raise ValueError("pilot_rows deve permitir pelo menos duas classes por fold.")
        if pilot_rows >= len(bundle["train"]):
            raise ValueError("pilot_rows deve ser menor que o treino para identificar um piloto.")
        selected, _ = train_test_split(
            np.arange(len(bundle["train"])),
            train_size=pilot_rows,
            stratify=bundle["y"],
            random_state=config["seed"],
        )
        selected.sort()
        bundle["train"] = bundle["train"].iloc[selected].reset_index(drop=True)
        bundle["y"] = bundle["y"].iloc[selected].reset_index(drop=True)
        bundle["exact_groups"] = bundle["exact_groups"][selected]
        bundle["hashes"] = {
            **bundle["hashes"],
            "train": fingerprint({"source": original_hash, "row_positions": selected.tolist()}),
        }
        config = {**config, "name": config["name"] + "_pilot"}
        # O piloto tem sua própria amostra e não substitui os folds oficiais congelados.
        config["folds_lock"] = None
        config["output_dir"] = str(Path(config["output_dir"]) / "pilots")
        config["registry"] = str(Path(config["reports_dir"]) / "pilot-experiments.csv")
    report = audit_data(bundle, config)
    folds, fold_metadata = get_folds(bundle, config)
    requested_folds = list(range(config["n_splits"])) if folds_to_run is None else folds_to_run
    if (
        not requested_folds
        or len(set(requested_folds)) != len(requested_folds)
        or any(type(f) is not int or not 0 <= f < config["n_splits"] for f in requested_folds)
    ):
        raise ValueError("folds_to_run deve selecionar folds únicos e válidos da divisão original.")
    requested_folds = sorted(requested_folds)
    screening = len(requested_folds) != config["n_splits"]
    bundle = apply_features(bundle, config)
    now = utc_now()
    experiment_id = (
        f"{now[:19].replace(':', '').replace('-', '')}_{config['name']}_{uuid.uuid4().hex[:6]}"
    )
    run_dir = Path(config["output_dir"]) / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = provenance()
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "environment.json", environment)
    write_json(run_dir / "audit.json", report)
    write_json(run_dir / "folds.json", fold_metadata)
    train_frame, test_frame = bundle["train"], bundle["test"]
    features, categorical, y = bundle["features"], bundle["categorical"], bundle["y"].to_numpy()
    categorical_indices = [features.index(column) for column in categorical]
    write_json(
        run_dir / "schema.json",
        {
            "features": features,
            "categorical_columns": categorical,
            "internal_names": [f"f{index}" for index in range(len(features))],
            "categorical_encoding": "missing=M:; observed=V:+canonical_value",
            "id_column": config["id_column"],
            "target_column": config["target_column"],
            "target_mapping": config["target_mapping"],
        },
    )
    oof = np.full(len(y), np.nan)
    oof_count = np.zeros(len(y), dtype=int)
    test_sum = np.zeros(len(test_frame), dtype=float)
    fold_results, importances = [], []
    try:
        for fold in requested_folds:
            fold_start = time.perf_counter()
            fit_index, valid_index = np.where(folds != fold)[0], np.where(folds == fold)[0]
            x_train, x_valid, x_test = prepare_fold(
                train_frame.iloc[fit_index][features],
                train_frame.iloc[valid_index][features],
                test_frame[features],
                categorical,
                config["model"],
            )
            write_json(
                run_dir / f"preprocessing_fold_{fold}.json",
                {
                    "training_rows": len(fit_index),
                    "validation_rows": len(valid_index),
                    "training_ids_sha256": fingerprint(
                        train_frame.iloc[fit_index][config["id_column"]].tolist()
                    ),
                    "validation_ids_sha256": fingerprint(
                        train_frame.iloc[valid_index][config["id_column"]].tolist()
                    ),
                    "train_sha256": bundle["hashes"]["train"],
                    "folds_sha256": fold_metadata["folds_sha256"],
                    "categories": {
                        str(index): x_train[f"f{index}"].cat.categories.tolist()
                        for index in categorical_indices
                    }
                    if config["model"] != "catboost"
                    else {},
                    "unseen_category_policy": "native hashing"
                    if config["model"] == "catboost"
                    else "missing category",
                },
            )
            model, best, importance = fit_fold(
                x_train,
                y[fit_index],
                x_valid,
                y[valid_index],
                categorical_indices,
                config,
                run_dir / f"model_fold_{fold}",
            )
            if config["model"] == "catboost":
                effective = model.get_all_params()
                history = model.get_evals_result()
            elif config["model"] == "lightgbm":
                effective = model.booster_.params
                history = model.evals_result_
            else:
                effective = json.loads(model.get_booster().save_config())
                history = model.evals_result()
            write_json(run_dir / f"effective_parameters_fold_{fold}.json", effective)
            write_json(run_dir / f"learning_history_fold_{fold}.json", history)
            iterations_evaluated = max(
                len(values) for dataset in history.values() for values in dataset.values()
            )
            valid_pred = validate_probabilities(
                model.predict_proba(x_valid)[:, 1], len(valid_index)
            )
            test_pred = validate_probabilities(model.predict_proba(x_test)[:, 1], len(test_frame))
            oof[valid_index] = valid_pred
            oof_count[valid_index] += 1
            test_sum += test_pred
            result = {
                "fold": fold,
                "auc": float(roc_auc_score(y[valid_index], valid_pred)),
                "best_iterations": best,
                "iterations_evaluated": iterations_evaluated,
                "train_rows": len(fit_index),
                "valid_rows": len(valid_index),
                "seconds": time.perf_counter() - fold_start,
            }
            fold_results.append(result)
            importances.append(importance)
            write_json(run_dir / "fold_metrics.json", fold_results)
            print(json.dumps(result), flush=True)
            del model, x_train, x_valid, x_test
            gc.collect()
        covered = np.isin(folds, requested_folds)
        if not np.all(oof_count[covered] == 1) or np.any(oof_count[~covered] != 0):
            raise ValueError("Cobertura OOF deve ser exatamente uma previsão por registro.")
        validate_probabilities(oof[covered], int(covered.sum()))
        ident, target = config["id_column"], config["target_column"]
        oof_path = run_dir / ("oof_screening.csv" if screening else "oof.csv")
        pd.DataFrame(
            {ident: train_frame[ident], "target": y, "fold": folds, "prediction": oof}
        ).loc[covered].to_csv(oof_path, index=False, float_format="%.17g")
        test_predictions = test_sum / len(requested_folds)
        # Pilotos e testes artificiais não recebem nome de submissão final.
        submission_name = (
            "submission.csv"
            if not screening and pilot_rows is None and config["dataset_role"] == "competition"
            else "diagnostic_predictions.csv"
        )
        submission = write_submission(
            test_frame[ident],
            test_predictions,
            bundle["sample"],
            run_dir / submission_name,
            ident,
            target,
        )
        pd.DataFrame({"feature": features, "importance": np.mean(importances, axis=0)}).to_csv(
            run_dir / "feature_importance.csv", index=False
        )
        aucs = [fold["auc"] for fold in fold_results]
        summary = {
            "experiment_id": experiment_id,
            "date_utc": now,
            "dataset_role": config["dataset_role"],
            "model": config["model"],
            "hypothesis": config["hypothesis"],
            "commit": environment["commit"],
            "dirty": environment["dirty"],
            "code_sha256": environment["code_sha256"],
            "train_sha256": bundle["hashes"]["train"],
            "source_train_sha256": original_hash,
            "data_hashes": bundle["hashes"],
            "fold_key": fold_metadata["key"],
            "seed": config["seed"],
            "auc_oof": None if screening else float(roc_auc_score(y, oof)),
            "validation_auc": float(roc_auc_score(y[covered], oof[covered])),
            "evaluation_scope": "screening" if screening else "full_cv",
            "folds_completed": requested_folds,
            "oof_rows": int(covered.sum()),
            "auc_fold_mean": float(np.mean(aucs)),
            "auc_fold_std": float(np.std(aucs)),
            "duration_seconds": time.perf_counter() - started,
            "folds": fold_results,
            "pilot_rows": pilot_rows,
            "submission_state": "generated"
            if submission_name == "submission.csv"
            else "diagnostic_only",
            "artifacts": str(run_dir),
            "submission_path": str(submission),
            "submission_sha256": sha256(submission),
            "oof_sha256": sha256(oof_path),
            "oof_path": str(oof_path),
            "model_hashes": {
                path.name: sha256(path) for path in sorted(run_dir.glob("model_fold_*"))
            },
            "public_score": None,
            "decision": "Avaliar auditoria e comparar OOF antes de escolher o modelo.",
            "hardware_requested": {
                key: value
                for key, value in config["params"].items()
                if key
                in {"task_type", "device", "device_type", "thread_count", "n_jobs", "devices"}
            },
        }
        write_json(run_dir / "metrics.json", summary)
        register_run(config, summary)
        return summary
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {
                "date_utc": utc_now(),
                "exception": type(exc).__name__,
                "message": str(exc),
                "completed_folds": len(fold_results),
                "status": "failed",
            },
        )
        raise
