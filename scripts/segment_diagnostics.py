"""Compara erros OOF em segmentos definidos antes dos resultados, sem criar regras."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from airline_ml.submission import validate_probabilities
from airline_ml.utils import sha256, utc_now, write_json


def read_prediction(run, ids, y, folds):
    run = Path(run)
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("evaluation_scope", "full_cv") != "full_cv":
        raise ValueError("Diagnóstico de confirmação requer OOF completa.")
    frame = pd.read_csv(run / "oof.csv", dtype={"id": "string"}, float_precision="round_trip")
    if not frame.id.is_unique or len(frame) != len(ids) or set(frame.id) != set(ids):
        raise ValueError("Cobertura de IDs divergente.")
    frame = frame.set_index("id").loc[ids]
    if not np.array_equal(frame.target, y) or not np.array_equal(frame.fold, folds):
        raise ValueError("Alvos ou folds divergentes.")
    return validate_probabilities(frame.prediction, len(ids))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    base = Path("artifacts/runs/20261030T184827_catboost_first_submission_06d4f0")
    cols = [
        "id",
        "satisfaction",
        "Age",
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
        "Arrival Delay in Minutes",
        "Inflight wifi service",
        "Online boarding",
    ]
    train = pd.read_csv("data/raw/train.csv", usecols=cols, dtype={"id": "string"})
    ids, y = train.id, train.satisfaction.astype("int8").to_numpy()
    oof = pd.read_csv(base / "oof.csv", dtype={"id": "string"}).set_index("id").loc[ids]
    folds = oof.fold.to_numpy()
    base_pred = read_prediction(base, ids, y, folds)
    pred = read_prediction(args.candidate, ids, y, folds)
    baseline_metrics = json.loads((base / "metrics.json").read_text())
    if sha256("data/raw/train.csv") != baseline_metrics["train_sha256"]:
        raise ValueError("Dados diferentes da baseline.")
    segments = {
        name: train[name].fillna("<MISSING>").astype(str)
        for name in (
            "Gender",
            "Customer Type",
            "Type of Travel",
            "Class",
            "Inflight wifi service",
            "Online boarding",
        )
    }
    segments["Age group"] = pd.cut(
        train.Age, [-np.inf, 24, 44, 64, np.inf], labels=["<=24", "25-44", "45-64", ">=65"]
    ).astype(str)
    delay = train["Arrival Delay in Minutes"]
    segments["Arrival delay group"] = (
        pd.cut(delay, [-np.inf, 0, 15, np.inf], labels=["zero", "1-15", ">15"])
        .astype("string")
        .fillna("missing")
    )
    segments["Class x Travel"] = train.Class + " / " + train["Type of Travel"]
    segments["Customer x Travel"] = train["Customer Type"] + " / " + train["Type of Travel"]
    rows = []
    for axis, values in segments.items():
        for label in sorted(values.unique()):
            mask = values.eq(label).to_numpy(dtype=bool)
            truth = y[mask]
            eligible = len(truth) >= 500 and truth.sum() >= 50 and (1 - truth).sum() >= 50
            base_auc = float(roc_auc_score(truth, base_pred[mask])) if eligible else None
            auc = float(roc_auc_score(truth, pred[mask])) if eligible else None
            rows.append(
                {
                    "axis": axis,
                    "segment": str(label),
                    "n": int(mask.sum()),
                    "positive_rate": float(truth.mean()),
                    "baseline_auc": base_auc,
                    "candidate_auc": auc,
                    "auc_delta": auc - base_auc if eligible else None,
                    "baseline_brier": float(np.mean((base_pred[mask] - truth) ** 2)),
                    "candidate_brier": float(np.mean((pred[mask] - truth) ** 2)),
                    "mean_prediction_minus_target": float(np.mean(pred[mask] - truth)),
                    "auc_note": "computed" if eligible else "small segment or insufficient classes",
                }
            )
    metadata = json.loads((Path(args.candidate) / "metrics.json").read_text())
    if metadata["fold_key"] != baseline_metrics["fold_key"]:
        raise ValueError("Candidato usa outra associação de folds.")
    report = {
        "updated_at_utc": utc_now(),
        "candidate": metadata["experiment_id"],
        "train_sha256": baseline_metrics["train_sha256"],
        "fold_key": metadata["fold_key"],
        "rows": rows,
        "interpretation": "Overlapping predeclared segments; descriptive, "
        "not causal or an independent confidence assessment. No thresholds or rules fitted.",
    }
    write_json("reports/campaign_segments.json", report)
    write_json(Path("reports/segments") / (metadata["experiment_id"] + ".json"), report)
    lines = [
        "# Diagnóstico agregado de erros OOF",
        "",
        f"Candidato: `{metadata['experiment_id']}`. Comparação com a baseline auditada, "
        "nas mesmas linhas e folds. Nenhum limiar ou regra foi ajustado aos rótulos.",
        "",
        "AUC por segmento exige pelo menos 500 linhas e 50 exemplos de cada classe. "
        "Brier e resíduo médio são diagnósticos, não objetivos de threshold. Segmentos "
        "se sobrepõem; diferenças são descritivas e não provam causa ou significância.",
        "",
        "## Exemplos agregados de segmentos com menor AUC",
        "",
        "| Eixo | Segmento | Linhas | AUC baseline | AUC candidato | Delta |",
        "|---|---|---:|---:|---:|---:|",
    ]
    eligible_rows = [row for row in rows if row["candidate_auc"] is not None and row["n"] >= 5000]
    for row in sorted(eligible_rows, key=lambda r: r["candidate_auc"])[:12]:
        lines.append(
            f"| {row['axis']} | {row['segment']} | {row['n']} | "
            f"{row['baseline_auc']:.6f} | {row['candidate_auc']:.6f} | {row['auc_delta']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Uso das observações",
            "",
            "Os agregados orientam novas comparações gerais de representação e interações "
            "de serviço/segmento. Não transformamos um segmento ou uma resposta errada "
            "em regra de predição. Hipóteses de features permanecem declaradas antes do teste.",
            "",
            "Todos os segmentos, inclusive os que pioraram, estão em "
            "[campaign_segments.json](../reports/campaign_segments.json). "
            "Não são publicados exemplos individuais.",
            "",
        ]
    )
    Path("docs/error-analysis.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"candidate": metadata["experiment_id"], "segments": len(rows)}))


if __name__ == "__main__":
    main()
