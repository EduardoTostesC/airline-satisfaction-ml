"""Gera o notebook fino e sem saídas; o treinamento continua nos módulos Python."""

from pathlib import Path

import nbformat as nbf


def build():
    book = nbf.v4.new_notebook()
    book.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    }
    book.cells = [
        nbf.v4.new_markdown_cell(
            "# Airline Satisfaction S6E10\n\n"
            "Autor: Adilson Eduardo Tostes Carmo. Notebook de execução do pipeline modular.\n\n"
            "No VS Code: **Select Kernel > Colab > Auto Connect**, após instalar a extensão "
            "oficial e entrar na conta. O terminal local e os arquivos locais não são o servidor "
            "Colab. Consulte `docs/colab.md`.\n\n"
            "Execute primeiro o smoke test. Treino completo permanece desativado até você "
            "colocar os dados oficiais no servidor e revisar a auditoria."
        ),
        nbf.v4.new_code_cell("""import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/EduardoTostesC/airline-satisfaction-ml.git"
CODE_REF = "main"  # Para repetir um experimento, use o SHA registrado na primeira execução.
IN_COLAB = "COLAB_RELEASE_TAG" in os.environ or "COLAB_BACKEND_VERSION" in os.environ
if not (3, 12) <= sys.version_info[:2] < (3, 14):
    raise RuntimeError("Use Python 3.12 ou 3.13; o ambiente testado é Python 3.12.")

if IN_COLAB:
    ROOT = Path("/content/airline-satisfaction-ml")
    if not ROOT.exists():
        subprocess.run(["git", "clone", "--no-checkout", REPO_URL, str(ROOT)], check=True)
    origin = subprocess.check_output(
        ["git", "-C", str(ROOT), "remote", "get-url", "origin"], text=True
    ).strip()
    if origin != REPO_URL:
        raise RuntimeError("A pasta remota pertence a outro repositório.")
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
    ).strip()
    # Clone sem checkout ainda não tem working tree. Em sessões reutilizadas, preserve edições.
    if (ROOT / "pyproject.toml").exists() and dirty:
        raise RuntimeError("Há edições remotas locais. Salve-as antes de atualizar o checkout.")
    subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", CODE_REF], check=True)
    subprocess.run(["git", "-C", str(ROOT), "checkout", "--detach", "FETCH_HEAD"], check=True)
else:
    ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                 if (p / "pyproject.toml").exists()), None)
    if ROOT is None:
        raise RuntimeError("Abra o notebook dentro do checkout do projeto.")
os.chdir(ROOT)
COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
print({"runtime": "colab" if IN_COLAB else "local", "python": sys.version, "commit": COMMIT})
if importlib.util.find_spec("pip") is None:
    subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-lock.txt"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."], check=True)
print("Se bibliotecas já importadas foram atualizadas, reinicie o kernel antes de continuar.")"""),
        nbf.v4.new_code_cell("""from airline_ml.smoke import run_smoke

smoke_result = run_smoke()
print({"status": "smoke concluído", "dataset_role": smoke_result["dataset_role"],
       "artifacts": smoke_result["artifacts"]})"""),
        nbf.v4.new_markdown_cell(
            "## Dados e persistência\n\n"
            "Baixe os três CSVs oficiais. No Colab, copie-os explicitamente para o servidor ou "
            "use uma pasta do Drive montada. Montar o Drive não sincroniza a pasta do computador. "
            "Preencha `DATA_DIR` abaixo com o caminho do servidor. Para montar o Drive, execute "
            "`from google.colab import drive; drive.mount('/content/drive')` em uma célula, "
            "usando a autenticação oficial. Nunca cole tokens neste notebook."
        ),
        nbf.v4.new_code_cell("""from airline_ml.config import load_config
from airline_ml.pipeline import audit, train
from airline_ml.utils import provenance

DATA_DIR = "/content/drive/MyDrive/airline-s6e10/raw" if IN_COLAB else "data/raw"
PERSIST_RESULTS = "/content/drive/MyDrive/airline-s6e10/results" if IN_COLAB else "artifacts"
RUN_AUDIT = False  # Ative após disponibilizar os CSVs no caminho acima.
RUN_PILOT = False
RUN_FULL_TRAIN = False
config = load_config("configs/catboost.json")
config["data_dir"] = DATA_DIR
config["output_dir"] = str(Path(PERSIST_RESULTS) / "artifacts")
config["reports_dir"] = str(Path(PERSIST_RESULTS) / "reports")
config["registry"] = str(Path(PERSIST_RESULTS) / "experiments.csv")
print({"commit": COMMIT, "code_sha256": provenance()["code_sha256"],
       "data_dir": DATA_DIR, "results_dir": PERSIST_RESULTS})
if IN_COLAB and (RUN_AUDIT or RUN_PILOT or RUN_FULL_TRAIN):
    if not Path("/content/drive/MyDrive").is_dir():
        raise RuntimeError("Monte o Drive antes de usar os caminhos padrão de persistência.")
if RUN_AUDIT:
    audit_result = audit(config)
    print({k: audit_result[k] for k in ["report_path", "train_rows", "test_rows",
                                      "positive_rate", "duplicate_feature_rows"]})"""),
        nbf.v4.new_code_cell("""if RUN_PILOT:
    pilot_result = train(config, pilot_rows=10000)
    print({"duration_seconds": pilot_result["duration_seconds"],
           "folds": pilot_result["folds"], "submission_state": pilot_result["submission_state"]})
if RUN_FULL_TRAIN:
    if not RUN_AUDIT:
        raise RuntimeError("Execute e revise a auditoria antes do treino completo.")
    result = train(config)
    print({"experiment_id": result["experiment_id"], "auc_oof": result["auc_oof"],
           "auc_fold_std": result["auc_fold_std"], "submission_path": result["submission_path"],
           "submission_state": result["submission_state"]})"""),
        nbf.v4.new_markdown_cell(
            "## Recuperar resultados\n\n"
            "Os caminhos acima apontam para o Drive quando executados no Colab. Confirme que "
            "`metrics.json`, `environment.json`, modelos, folds e submissão persistiram antes "
            "de encerrar o runtime. Para cópia manual de um arquivo, use "
            "`from google.colab import files; files.download(result['submission_path'])`. "
            "Isso baixa um arquivo para o computador, sem efetuar push ou envio ao Kaggle. "
            "Limpe as saídas do notebook antes de versionar."
        ),
    ]
    path = Path("notebooks/airline_satisfaction.ipynb")
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.validate(book)
    nbf.write(book, path)
    print(path)


if __name__ == "__main__":
    build()
