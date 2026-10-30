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
        "## Resultados medidos",
        "",
        "| Tentativa | Escopo | AUC OOF completa | AUC validação coberta | Delta OOF | "
        "Minutos | Pico RSS GiB | Consistente |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]

    def number(value):
        return "—" if value is None else f"{value:.8f}"

    for item in results:
        resource = item["resources"] or {}
        rss = resource.get("peak_rss_bytes")
        lines.append(
            f"| {item['name']} | {item['scope']} {item['folds_completed']} | "
            f"{number(item['auc_oof'])} | {item['validation_auc']:.8f} | "
            f"{number(item['global_oof_delta_vs_baseline'])} | "
            f"{item['duration_seconds'] / 60:.2f} | "
            f"{f'{rss / 2**30:.2f}' if rss is not None else '—'} | "
            f"{'sim' if item['consistent_improvement_vs_baseline'] else 'não/triagem'} |"
        )
    lines.extend(["", "## Hipóteses, inclusive resultados sem ganho", ""])
    for item in results:
        lines.append(
            f"- **{item['name']}**: {item['hypothesis']} "
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
            raise
    if process.returncode:
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
