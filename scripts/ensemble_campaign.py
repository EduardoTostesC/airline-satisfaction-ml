"""Combina poucas OOF completas com pesos declarados, sem otimização de limiar."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from airline_ml.pipeline import register_run
from airline_ml.resources import profile_run
from airline_ml.submission import validate_probabilities, write_submission
from airline_ml.utils import provenance, sha256, utc_now, write_json


@profile_run
def combine(name, members, weights):
    started = time.perf_counter()
    if not name.startswith("camp_ensemble_"):
        raise ValueError("Use um nome novo camp_ensemble_ para manter os artefatos anteriores.")
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(members) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError("Pesos devem ser não negativos, somar um e corresponder aos membros.")
    if not np.isfinite(weights).all():
        raise ValueError("Pesos inválidos.")
    weights /= weights.sum()
    reference = None
    oof_sum = test_sum = None
    evidence = []
    for member, weight in zip(members, weights):
        folder = Path(member)
        metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        if (
            sha256(folder / "oof.csv") != metrics["oof_sha256"]
            or sha256(folder / "submission.csv") != metrics["submission_sha256"]
        ):
            raise ValueError("Hash de previsão de um membro não confere.")
        if metrics.get("evaluation_scope", "full_cv") != "full_cv":
            raise ValueError("Ensemble requer membros confirmados com OOF completa.")
        oof = pd.read_csv(folder / "oof.csv", dtype={"id": "string"}, float_precision="round_trip")
        test = pd.read_csv(
            folder / "submission.csv", dtype={"id": "string"}, float_precision="round_trip"
        )
        if not oof.id.is_unique or not test.id.is_unique:
            raise ValueError("IDs duplicados em um membro.")
        if reference is None:
            reference = {
                "ids": oof.id.copy(),
                "y": oof.target.to_numpy(),
                "folds": oof.fold.to_numpy(),
                "test_ids": test.id.copy(),
                "metrics": metrics,
            }
            oof_sum, test_sum = np.zeros(len(oof)), np.zeros(len(test))
        else:
            if metrics["fold_key"] != reference["metrics"]["fold_key"]:
                raise ValueError("Membros usam folds diferentes.")
            if metrics["data_hashes"] != reference["metrics"]["data_hashes"]:
                raise ValueError("Membros usam dados diferentes.")
            if set(oof.id) != set(reference["ids"]) or set(test.id) != set(reference["test_ids"]):
                raise ValueError("Cobertura de IDs diferente.")
            oof = oof.set_index("id").loc[reference["ids"]].reset_index()
            test = test.set_index("id").loc[reference["test_ids"]].reset_index()
            if not np.array_equal(oof.target, reference["y"]) or not np.array_equal(
                oof.fold, reference["folds"]
            ):
                raise ValueError("Alvos ou folds desalinhados.")
        oof_sum += weight * validate_probabilities(oof.prediction, len(oof))
        test_sum += weight * validate_probabilities(test.satisfaction, len(test))
        evidence.append(
            {
                "experiment_id": metrics["experiment_id"],
                "weight": float(weight),
                "oof_sha256": sha256(folder / "oof.csv"),
                "submission_sha256": sha256(folder / "submission.csv"),
            }
        )
    cfg = {
        "name": name,
        "model": "ensemble",
        "dataset_role": "competition",
        "registry": "reports/experiments.csv",
        "hypothesis": (
            "Média de probabilidades de membros com OOF completas, pesos fixos e IDs alinhados."
        ),
        "params": {"members": evidence},
        "seed": 2026,
    }
    now = utc_now()
    run_id = now[:19].replace(":", "").replace("-", "") + "_" + name
    run = Path("artifacts/runs") / run_id
    run.mkdir(parents=True, exist_ok=False)
    environment = provenance()
    write_json(run / "config.json", cfg)
    write_json(run / "environment.json", environment)
    pd.DataFrame(
        {
            "id": reference["ids"],
            "target": reference["y"],
            "fold": reference["folds"],
            "prediction": oof_sum,
        }
    ).to_csv(run / "oof.csv", index=False, float_format="%.17g")
    sample = pd.read_csv("data/raw/sample_submission.csv", dtype={"id": "string"})
    submission = write_submission(reference["test_ids"], test_sum, sample, run / "submission.csv")
    folds = []
    for fold in sorted(np.unique(reference["folds"])):
        mask = reference["folds"] == fold
        folds.append(
            {"fold": int(fold), "auc": float(roc_auc_score(reference["y"][mask], oof_sum[mask]))}
        )
    scores = [fold["auc"] for fold in folds]
    result = {
        "experiment_id": run_id,
        "date_utc": now,
        "dataset_role": "competition",
        "model": "ensemble",
        "hypothesis": cfg["hypothesis"],
        "commit": environment["commit"],
        "dirty": environment["dirty"],
        "code_sha256": environment["code_sha256"],
        "train_sha256": reference["metrics"]["train_sha256"],
        "data_hashes": reference["metrics"]["data_hashes"],
        "fold_key": reference["metrics"]["fold_key"],
        "seed": 2026,
        "auc_oof": float(roc_auc_score(reference["y"], oof_sum)),
        "auc_fold_mean": float(np.mean(scores)),
        "auc_fold_std": float(np.std(scores)),
        "folds": folds,
        "folds_completed": [fold["fold"] for fold in folds],
        "evaluation_scope": "full_cv",
        "validation_auc": float(roc_auc_score(reference["y"], oof_sum)),
        "oof_rows": len(oof_sum),
        "oof_path": str(run / "oof.csv"),
        "duration_seconds": time.perf_counter() - started,
        "submission_state": "generated",
        "artifacts": str(run),
        "submission_path": str(submission),
        "submission_sha256": sha256(submission),
        "oof_sha256": sha256(run / "oof.csv"),
        "members": evidence,
        "public_score": None,
    }
    write_json(run / "metrics.json", result)
    register_run(cfg, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(combine(args.name, args.members, args.weights), ensure_ascii=True, indent=2))
