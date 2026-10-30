"""Confere somente arquivos destinados ao Git, sem ler credenciais locais."""

import json
import re
import subprocess
from pathlib import Path


def main():
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=True,
        capture_output=True,
    )
    paths = sorted(set(result.stdout.decode().rstrip("\0").split("\0")) - {""})
    errors = []
    secret_pattern = re.compile(
        r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
        r"AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
    )
    for name in paths:
        path = Path(name)
        if not path.is_file():
            continue
        if path.suffix.lower() in {".csv", ".parquet", ".zip", ".cbm", ".pkl", ".ubj", ".npy"}:
            if name != "reports/experiments.csv":
                errors.append(f"Dados ou artefato individual: {name}")
        if any(part in {".venv", "artifacts", "raw", "external"} for part in path.parts):
            errors.append(f"Pasta excluída: {name}")
        if path.name.startswith(".env") or path.name in {"kaggle.json", "access_token"}:
            errors.append(f"Credencial/configuração privada: {name}")
        if path.stat().st_size > 1024 * 1024:
            errors.append(f"Arquivo maior que 1 MiB: {name}")
        if path.suffix in {".py", ".md", ".json", ".toml", ".yml", ".ipynb", ".csv"}:
            content = path.read_text(encoding="utf-8")
            if secret_pattern.search(content):
                errors.append(f"Possível segredo: {name}")
            if path.suffix == ".ipynb":
                notebook = json.loads(content)
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code" and (
                        cell.get("outputs") or cell.get("execution_count") is not None
                    ):
                        errors.append(f"Notebook com saídas persistidas: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Publicação: {len(paths)} arquivos verificados.")
    print("Sem dados individuais ou segredos detectados pelos padrões configurados.")


if __name__ == "__main__":
    main()
