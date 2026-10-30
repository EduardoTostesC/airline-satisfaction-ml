"""Executa uma tentativa isolada e atualiza o placar agregado da campanha."""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

from airline_ml.utils import utc_now, write_json

RESULTS = Path("reports/campaign_results.json")
DOC = Path("docs/campaign.md")


def collect():
    plan = json.loads(Path("reports/campaign_plan.json").read_text(encoding="utf-8"))
    baseline = plan["baseline"]
    results = []
    for path in Path("artifacts/runs").glob("*/metrics.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        cfg = json.loads(path.with_name("config.json").read_text(encoding="utf-8"))
        if not cfg["name"].startswith(("camp_", "cost_")):
            continue
        full = value.get("evaluation_scope") == "full_cv" and value.get("pilot_rows") is None
        if full and (
            value["folds_completed"] != [0, 1, 2, 3, 4]
            or value["oof_rows"] != 699635
            or value["fold_key"] != plan["fold_key"]
            or cfg["dataset_role"] != "competition"
        ):
            raise ValueError("Candidato completo não cobre os cinco folds/dados da campanha.")
        matched = {x["fold"]: x["auc"] for x in baseline["folds"]}
        deltas = [x["auc"] - matched[x["fold"]] for x in value["folds"]]
        consistent = (
            full
            and value["auc_oof"] > baseline["auc_oof"]
            and sum(delta > 0 for delta in deltas) >= 4
            and min(deltas) >= -0.0002
        )
        item = {
            "name": cfg["name"],
            "experiment_id": value["experiment_id"],
            "model": cfg["model"],
            "scope": value.get("evaluation_scope"),
            "folds_completed": value["folds_completed"],
            "auc_oof": value["auc_oof"],
            "validation_auc": value["validation_auc"],
            "duration_seconds": value["duration_seconds"],
            "resources": value.get("resources"),
            "fold_aucs": [x["auc"] for x in value["folds"]],
            "fold_deltas_vs_baseline": deltas,
            "global_oof_delta_vs_baseline": value["auc_oof"] - baseline["auc_oof"]
            if full
            else None,
            "consistent_improvement_vs_baseline": consistent,
            "hypothesis": cfg["hypothesis"],
            "params": cfg["params"],
            "feature_groups": cfg.get("feature_groups", []),
            "service_representation": cfg.get("service_representation", "ordinal"),
            "artifacts": str(path.parent),
            "submission_state": value["submission_state"],
            "fold_key": value["fold_key"],
        }
        results.append(item)
    results.sort(key=lambda row: row["experiment_id"])
    by_name = {row["name"]: row for row in results}
    for row in results:
        comparison = plan.get("comparisons", {}).get(row["name"], {})
        control = by_name.get(comparison.get("control_name"))
        row["control_name"] = comparison.get("control_name")
        if row["scope"] == "screening":
            reference = comparison.get("screen_reference_auc")
            row["delta_vs_control"] = (
                row["validation_auc"] - reference if reference is not None else None
            )
            row["confirmed_improvement_vs_control"] = False
        elif control is not None and control["scope"] == "full_cv":
            row["delta_vs_control"] = row["auc_oof"] - control["auc_oof"]
            deltas = [a - b for a, b in zip(row["fold_aucs"], control["fold_aucs"])]
            row["fold_deltas_vs_control"] = deltas
            row["confirmed_improvement_vs_control"] = (
                row["delta_vs_control"] > 0
                and sum(x > 0 for x in deltas) >= 4
                and min(deltas) >= -0.0002
            )
        else:
            row["delta_vs_control"] = row["global_oof_delta_vs_baseline"]
            row["confirmed_improvement_vs_control"] = row["consistent_improvement_vs_baseline"]
    write_json(RESULTS, {"updated_at_utc": utc_now(), "baseline": baseline, "results": results})
    lines = [
        "# Campanha controlada de melhoria",
        "",
        "## Referência e orçamento",
        "",
        f"Baseline auditada: OOF {baseline['auc_oof']:.8f}; pública informada 0,95770. "
        f"Duração {baseline['duration_seconds'] / 60:.2f} min. "
        "Pico de memória histórico não medido.",
        "",
        "Teto autorizado: quatro horas de execução local. Os folds originais permanecem "
        "congelados. Triagens usam só o fold 0 e não confirmam nem promovem candidatos.",
        "",
        "Critério prático de consistência: cinco folds completos, OOF global acima da "
        "baseline, ganho em pelo menos quatro folds e pior delta de fold ≥ -0,0002. "
        "Isso não é teste de significância nem previsão de leaderboard.",
        "",
        "Amostragem de memória a cada 0,5 s: RSS é memória residente; privada é comprometida "
        "no Windows e pode incluir páginas fora da RAM. Picos são amostrais, não limites exatos.",
        "",
        "## Custo e prioridade antes das confirmações",
        "",
        "Projeções lineares dos perfis de 200 árvores, usadas apenas para alocar orçamento. "
        "Overhead, early stopping, profundidade e paginação impedem tratá-las como previsão exata.",
        "",
        "| Modelo | Fold 0, 200 árvores (s) | Projeção de CV completa (min) | "
        "Pico RSS/privado do perfil (GiB) |",
        "|---|---:|---:|---:|",
    ]
    for model in plan.get("execution_order", [])[:3]:
        family = next((name for name in ("lightgbm", "catboost", "xgboost") if name in model), None)
        estimate = plan.get("cost_profiles_and_estimates", {}).get(family)
        if estimate:
            lines.append(
                f"| {family} | {estimate['measured_seconds_200_fold0']:.1f} | "
                f"{estimate['rough_full_cv_projection_seconds'] / 60:.1f} | "
                f"{estimate['peak_rss_measured_bytes'] / 2**30:.2f} / "
                f"{estimate['peak_private_measured_bytes'] / 2**30:.2f} |"
            )
    lines.extend(
        [
            "",
            "Ordem de confirmação: LightGBM, CatBoost com 2000 árvores e XGBoost com 1500. "
            "Ela considera o custo medido, sem supor superioridade de score. "
            "Execuções são sequenciais.",
            "",
            "## Resultados medidos",
            "",
            "| Tentativa | Escopo | AUC OOF completa | AUC validação coberta | "
            "Delta OOF | Delta controle | "
            "Minutos | Pico RSS GiB | Consistente |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    def number(value):
        return "—" if value is None else f"{value:.8f}"

    lines.append(
        f"| baseline_catboost_1000 | full_cv [0,1,2,3,4] | {baseline['auc_oof']:.8f} | "
        f"{baseline['auc_oof']:.8f} | 0 | 0 | {baseline['duration_seconds'] / 60:.2f} | "
        "— | referência |"
    )
    for item in results:
        resource = item["resources"] or {}
        rss = resource.get("peak_rss_bytes")
        lines.append(
            f"| {item['name']} | {item['scope']} {item['folds_completed']} | "
            f"{number(item['auc_oof'])} | {item['validation_auc']:.8f} | "
            f"{number(item['global_oof_delta_vs_baseline'])} | "
            f"{number(item['delta_vs_control'])} | "
            f"{item['duration_seconds'] / 60:.2f} | "
            f"{f'{rss / 2**30:.2f}' if rss is not None else '—'} | "
            f"{'sim' if item['confirmed_improvement_vs_control'] else 'não/triagem'} |"
        )
    lines.extend(["", "## Hipóteses, inclusive resultados sem ganho", ""])
    for item in results:
        lines.append(
            f"- **{item['name']}**: {item['hypothesis']} "
            f"Controle direto: {item['control_name'] or 'baseline auditada'}. "
            f"Deltas nos folds avaliados: {[round(x, 8) for x in item['fold_deltas_vs_baseline']]}."
        )
    lines.extend(
        [
            "",
            "A primeira submissão é preservada. Não há nova nota pública registrada "
            "para estes candidatos. Previsões individuais e modelos ficam fora do Git.",
            "",
            "Fontes: [LightGBM](https://lightgbm.readthedocs.io/en/stable/Advanced-Topics.html), "
            "[XGBoost](https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html), "
            "[psutil](https://psutil.io/).",
            "",
        ]
    )
    failures = Path("reports/campaign_failures.json")
    if failures.exists():
        lines.extend(["", "## Falhas ou tentativas interrompidas", ""])
        for failure in json.loads(failures.read_text(encoding="utf-8")):
            lines.append(f"- {failure['name']}: {failure['status']}; log local {failure['log']}.")
    DOC.write_text("\n".join(lines), encoding="utf-8")
    return results


def terminate_owned_tree(process):
    try:
        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            child.terminate()
        root.terminate()
        _, alive = psutil.wait_procs([*children, root], timeout=5)
        for child in alive:
            child.kill()
    except psutil.NoSuchProcess:
        pass


def run(config, folds, timeout):
    cfg = json.loads(Path(config).read_text(encoding="utf-8"))
    plan = json.loads(Path("reports/campaign_plan.json").read_text(encoding="utf-8"))
    campaign_start = datetime.fromisoformat(plan["campaign_started_utc"])
    remaining = (
        plan["execution_budget_seconds"]
        - (datetime.now(timezone.utc) - campaign_start).total_seconds()
    )
    if remaining <= 0:
        raise TimeoutError(
            "Orçamento global da campanha encerrado; nenhuma tentativa nova iniciada."
        )
    timeout = min(timeout, remaining)
    folder = Path("artifacts/campaign")
    folder.mkdir(parents=True, exist_ok=True)
    name = cfg["name"] + ("_full" if folds is None else "_screen")
    pointer = folder / (name + "_result.json")
    if pointer.exists():
        raise ValueError("Essa tentativa já tem resultado; preserve-a e use outro nome.")
    log = Path("reports/local") / (name + ".log")
    program = (
        "import json,sys;from airline_ml.config import load_config;"
        "from airline_ml.pipeline import train;from airline_ml.utils import write_json;"
        "r=train(load_config(sys.argv[1]),folds_to_run=json.loads(sys.argv[2]));"
        "write_json(sys.argv[3],r)"
    )
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", program, config, json.dumps(folds), str(pointer)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        try:
            while process.poll() is None:
                if time.monotonic() - started >= timeout:
                    terminate_owned_tree(process)
                    raise TimeoutError(
                        "Orçamento da tentativa encerrado; processo próprio terminado."
                    )
                time.sleep(1)
        except BaseException:
            terminate_owned_tree(process)
            failures = Path("reports/campaign_failures.json")
            old = json.loads(failures.read_text(encoding="utf-8")) if failures.exists() else []
            old.append(
                {
                    "name": name,
                    "date_utc": utc_now(),
                    "status": "interrupted",
                    "log": str(log),
                    "wall_seconds": time.monotonic() - started,
                }
            )
            write_json(failures, old)
            raise
    if process.returncode:
        failures = Path("reports/campaign_failures.json")
        old = json.loads(failures.read_text(encoding="utf-8")) if failures.exists() else []
        old.append(
            {
                "name": name,
                "date_utc": utc_now(),
                "status": "failed",
                "log": str(log),
                "returncode": process.returncode,
            }
        )
        write_json(failures, old)
        raise RuntimeError(f"Tentativa falhou; veja {log}")
    results = collect()
    print(json.dumps(results[-1], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--folds", nargs="+", type=int)
    parser.add_argument("--timeout", type=int, default=3600)
    args = parser.parse_args()
    if args.config:
        run(args.config, args.folds, args.timeout)
    else:
        print(json.dumps(collect(), ensure_ascii=True, indent=2))
