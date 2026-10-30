"""Reproduz a OOF e a submissão a partir dos modelos salvos, sem treinar nem sobrescrever."""

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from airline_ml.config import load_config
from airline_ml.data import load_data
from airline_ml.folds import validate_folds
from airline_ml.models import prepare_fold
from airline_ml.submission import validate_submission, write_submission
from airline_ml.utils import fingerprint, sha256, utc_now, write_json


def check(condition, message):
    if not condition:
        raise ValueError(message)


def align_oof(oof, ids, y, folds):
    """Alinha por chave, rejeitando falta, duplicata, alvo/fold trocados e previsão inválida."""
    check(list(oof.columns) == [ids.name, "target", "fold", "prediction"], "Esquema OOF inválido.")
    check(not oof[ids.name].isna().any(), "OOF com ID ausente.")
    check(oof[ids.name].is_unique, "OOF com IDs repetidos.")
    check(
        len(oof) == len(ids) and set(oof[ids.name]) == set(ids), "OOF incompleta ou com IDs extras."
    )
    aligned = oof.set_index(ids.name).loc[ids].reset_index()
    check(np.array_equal(aligned.target.to_numpy(), y), "Alvos da OOF desalinhados.")
    check(np.array_equal(aligned.fold.to_numpy(), folds), "Folds da OOF desalinhados.")
    pred = aligned.prediction.to_numpy(dtype=float)
    check(np.isfinite(pred).all() and ((pred >= 0) & (pred <= 1)).all(), "OOF inválida.")
    return aligned


def predict_chunks(model, frame, features, categorical, rows):
    result = np.empty(len(frame), dtype=float)
    for start in range(0, len(frame), rows):
        chunk = frame.iloc[start : start + rows][features]
        prepared = prepare_fold(chunk, chunk.iloc[:0], chunk.iloc[:0], categorical, "catboost")[0]
        result[start : start + len(chunk)] = model.predict_proba(prepared, thread_count=4)[:, 1]
    return result


def audit(run, submitted, output, chunk_rows=25000):
    run, submitted, output = Path(run), Path(submitted), Path(output)
    check(not output.exists(), f"Pasta de auditoria já existe: {output}; escolha outra.")
    config = load_config(run / "config.json")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    environment = json.loads((run / "environment.json").read_text(encoding="utf-8"))
    schema = json.loads((run / "schema.json").read_text(encoding="utf-8"))
    fold_meta = json.loads((run / "folds.json").read_text(encoding="utf-8"))
    original_hash = sha256(submitted)
    check(original_hash == metrics["submission_sha256"], "CSV informado não é o registrado.")
    check(sha256(run / "submission.csv") == original_hash, "CSV arquivado divergente.")
    check(sha256(run / "oof.csv") == metrics["oof_sha256"], "Hash da OOF divergente.")

    # Confere o código histórico pelo Git; a auditoria não requer reescrever o checkout.
    source_checks = {}
    paths = [
        f"src/airline_ml/{name}.py"
        for name in ("config", "data", "folds", "models", "pipeline", "submission", "utils")
    ] + ["configs/catboost-first.json"]
    for path in paths:
        blob = subprocess.check_output(["git", "show", f"{environment['commit']}:{path}"])
        expected = environment["source_files"][path]
        historical_hash = hashlib.sha256(blob).hexdigest()
        check(historical_hash == expected, f"Código histórico não confere: {path}")
        source_checks[path] = {
            "recorded_sha256": expected,
            "historical_git_sha256": historical_hash,
            "current_file_sha256": sha256(path),
        }
    for path in (
        "src/airline_ml/data.py",
        "src/airline_ml/models.py",
        "src/airline_ml/submission.py",
    ):
        check(
            source_checks[path]["current_file_sha256"] == source_checks[path]["recorded_sha256"],
            f"Inferência mudou: use o checkout original: {path}",
        )

    bundle = load_data(config)
    check(bundle["hashes"] == metrics["data_hashes"], "Os dados não são os da baseline.")
    features, cats = schema["features"], schema["categorical_columns"]
    ident, target = config["id_column"], config["target_column"]
    expected_features = [c for c in bundle["train"] if c not in {ident, target}]
    check(
        features == bundle["features"] == expected_features, "Features descartadas ou diferentes."
    )
    check(cats == bundle["categorical"], "Categorias diferentes das registradas.")
    check(not {ident, target} & set(features), "ID/alvo entre as features.")
    check(
        not config["drop_columns"] and config["dataset_role"] == "competition", "Escopo inesperado."
    )

    fold_dir = Path(config["output_dir"]) / "folds" / fold_meta["key"]
    fold_csv = fold_dir / "folds.csv"
    check(sha256(fold_csv) == fold_meta["folds_sha256"], "Associação ID/fold foi alterada.")
    check(fingerprint(fold_meta["settings"]) == fold_meta["key"], "Chave dos folds inválida.")
    cache_meta = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    check(
        all(cache_meta[k] == fold_meta[k] for k in ("settings", "key", "folds_sha256")),
        "Metadados do cache divergentes.",
    )
    table = pd.read_csv(fold_csv, dtype={ident: "string"})
    ids, y = bundle["train"][ident], bundle["y"].to_numpy()
    folds = validate_folds(table, ids, y, config["n_splits"])
    expected_folds = np.full(len(y), -1)
    splitter = StratifiedKFold(config["n_splits"], shuffle=True, random_state=config["seed"])
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(y)), y)):
        expected_folds[valid] = fold
    check(np.array_equal(folds, expected_folds), "Folds não reproduzem a divisão registrada.")

    raw_oof = pd.read_csv(run / "oof.csv", dtype={ident: "string"}, float_precision="round_trip")
    oof_order_matches = raw_oof[ident].tolist() == ids.tolist()
    oof = align_oof(raw_oof, ids, y, folds)
    del raw_oof
    check(set(bundle["train"][target].unique()) == {False, True}, "Alvo booleano não confirmado.")
    check(np.array_equal(bundle["train"][target].astype("int8"), y), "Classe positiva invertida.")
    submitted_frame = pd.read_csv(submitted, dtype={ident: "string"}, float_precision="round_trip")
    validate_submission(submitted_frame, bundle["sample"], bundle["test"][ident], ident, target)
    test_sum = np.zeros(len(bundle["test"]))
    replay_oof = np.full(len(y), np.nan)
    coverage = np.zeros(len(y), dtype="uint8")
    results = []
    for fold in range(config["n_splits"]):
        model_path = run / f"model_fold_{fold}.cbm"
        check(sha256(model_path) == metrics["model_hashes"][model_path.name], "Modelo alterado.")
        model = CatBoostClassifier()
        model.load_model(str(model_path))
        check(model.classes_.tolist() == [0, 1], "Ordem de classes invertida.")
        check(model.feature_names_ == schema["internal_names"], "Esquema do modelo divergente.")
        check(
            model.get_cat_feature_indices() == [features.index(c) for c in cats],
            "Posições categóricas divergentes.",
        )
        meta = dict(model.get_metadata())
        effective = model.get_all_params()
        native = json.loads(meta["params"])
        history = json.loads(meta["training"])["metrics"]["test_metrics_history"]
        check(effective["task_type"] == "CPU", "Dispositivo diferente do documentado.")
        check(effective["random_seed"] == config["seed"], "Seed do modelo divergente.")
        check(not native["data_processing_options"]["ignored_features"], "Features ignoradas.")
        valid_idx, train_idx = np.where(folds == fold)[0], np.where(folds != fold)[0]
        check(not np.intersect1d(valid_idx, train_idx).size, "Treino e validação se sobrepõem.")
        valid = bundle["train"].iloc[valid_idx]
        pred = predict_chunks(model, valid, features, cats, chunk_rows)
        error = float(np.max(np.abs(pred - oof.prediction.to_numpy()[valid_idx])))
        check(error <= 1e-12, "OOF não corresponde ao modelo do fold designado.")
        replay_oof[valid_idx] = pred
        coverage[valid_idx] += 1
        test_sum += predict_chunks(model, bundle["test"], features, cats, chunk_rows)
        best = int(model.get_best_iteration())
        check(
            best + 1 == model.tree_count_ == metrics["folds"][fold]["best_iterations"],
            "Melhor iteração/quantidade de árvores inconsistente.",
        )
        auc = float(roc_auc_score(y[valid_idx], pred))
        check(abs(auc - metrics["folds"][fold]["auc"]) <= 1e-12, "AUC de fold divergente.")
        result = {
            "fold": fold,
            "model_sha256": sha256(model_path),
            "auc_replayed": auc,
            "oof_max_absolute_error": error,
            "validation_rows": len(valid_idx),
            "training_rows_reconstructed_from_split": len(train_idx),
            "split_train_validation_overlap": 0,
            "classes": model.classes_.tolist(),
            "best_iteration_zero_based": best,
            "retained_trees": int(model.tree_count_),
            "iterations_evaluated": len(history),
            "configured_iteration_cap": effective["iterations"],
            "auc_at_500_trees": history[499][0]["AUC"] if len(history) >= 500 else None,
            "auc_at_last_evaluated_iteration": history[-1][0]["AUC"],
            "effective_parameters": effective,
            "explicit_native_parameters": native["flat_params"],
            "system_parameters": native["system_options"],
            "ignored_features": native["data_processing_options"]["ignored_features"],
            "zero_importance_features": [
                features[i] for i, v in enumerate(model.get_feature_importance()) if v == 0
            ],
            "train_finish_time": meta["train_finish_time"],
        }
        results.append(result)
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "fold",
                        "auc_replayed",
                        "oof_max_absolute_error",
                        "best_iteration_zero_based",
                    )
                }
            ),
            flush=True,
        )
        del model, valid, pred
        gc.collect()

    check(np.all(coverage == 1), "Cobertura OOF diferente de uma previsão por linha.")
    auc = float(roc_auc_score(y, replay_oof))
    check(abs(auc - metrics["auc_oof"]) <= 1e-12, "AUC OOF global divergente.")
    mean = pd.Series(test_sum / config["n_splits"], index=bundle["test"][ident].to_numpy())
    test_error = float(
        np.max(
            np.abs(mean.loc[submitted_frame[ident]].to_numpy() - submitted_frame[target].to_numpy())
        )
    )
    check(test_error <= 1e-12, "Submissão não corresponde à média dos folds.")
    output.mkdir(parents=True, exist_ok=False)
    replay_file = write_submission(
        bundle["test"][ident],
        test_sum / config["n_splits"],
        bundle["sample"],
        output / "submission_replayed.csv",
        ident,
        target,
    )
    check(sha256(replay_file) == original_hash, "Reprodução não é idêntica byte a byte.")
    check(sha256(submitted) == original_hash, "CSV original foi alterado durante auditoria.")
    report = {
        "audited_at_utc": utc_now(),
        "experiment_id": metrics["experiment_id"],
        "submission_sha256": original_hash,
        "oof_sha256": sha256(run / "oof.csv"),
        "source_commit": environment["commit"],
        "source_code_sha256": environment["code_sha256"],
        "source_checks": source_checks,
        "data_hashes": bundle["hashes"],
        "fold_lock": {k: fold_meta[k] for k in ("settings", "key", "folds_sha256")},
        "oof_rows": len(oof),
        "oof_ids_unique": True,
        "oof_file_order_matches_train": oof_order_matches,
        "oof_coverage_min": int(coverage.min()),
        "oof_coverage_max": int(coverage.max()),
        "auc_oof_replayed": auc,
        "auc_if_positive_class_inverted": float(roc_auc_score(y, 1 - replay_oof)),
        "test_rows": len(submitted_frame),
        "test_mean_max_absolute_error": test_error,
        "submission_replayed_byte_identical": True,
        "features": features,
        "categorical_columns": cats,
        "folds": results,
        "membership_evidence_limit": (
            "Original fit calls did not save training IDs inside CBM files. Membership is inferred "
            "from the audited historical code and persisted split, with all held-out predictions "
            "replayed. This is not an independent contemporaneous proof of training membership."
        ),
    }
    write_json(output / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--submission", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = audit(args.run_dir, args.submission, args.output_dir)
    print(
        json.dumps(
            {
                "audit": str(Path(args.output_dir) / "audit.json"),
                "auc_oof": result["auc_oof_replayed"],
                "byte_identical": True,
            }
        )
    )
