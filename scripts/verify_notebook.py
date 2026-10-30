"""Executa o notebook local com um kernelspec temporário dentro de artifacts/."""

import json
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient


def main():
    root = Path.cwd()
    folder = root / "artifacts" / "notebook-qa"
    kernel_name = "airline-local-qa"
    kernel_root = folder / "kernels"
    spec = kernel_root / kernel_name
    spec.mkdir(parents=True, exist_ok=True)
    (spec / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Airline local QA",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    manager = KernelManager(
        kernel_name=kernel_name,
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]),
    )
    notebook = nbformat.read(root / "notebooks/airline_satisfaction.ipynb", as_version=4)
    nbformat.validate(notebook)
    client = NotebookClient(
        notebook,
        km=manager,
        kernel_name=kernel_name,
        timeout=300,
        resources={"metadata": {"path": str(root)}},
    )
    executed = client.execute()
    nbformat.write(executed, folder / "executed-local.ipynb")
    print(f"Notebook executado localmente: {folder / 'executed-local.ipynb'}")


if __name__ == "__main__":
    main()
