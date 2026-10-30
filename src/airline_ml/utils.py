"""Rastreabilidade sem capturar credenciais ou conteúdo dos dados."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def git_output(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def provenance(root="."):
    root = Path(root)
    code = {}
    for folder in ("src", "configs", "scripts", "notebooks"):
        for path in sorted((root / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json", ".ipynb"}:
                code[path.relative_to(root).as_posix()] = sha256(path)
    for name in ("pyproject.toml", "requirements-lock.txt", "requirements-boosting-lock.txt"):
        if (root / name).is_file():
            code[name] = sha256(root / name)
    versions = {}
    for package in ("numpy", "pandas", "scikit-learn", "catboost", "lightgbm", "xgboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    hardware = {"logical_cpus": os.cpu_count(), "processor": platform.processor()}
    try:
        import psutil

        hardware.update(
            {
                "physical_cpus": psutil.cpu_count(logical=False),
                "ram_total_bytes": psutil.virtual_memory().total,
                "ram_available_bytes": psutil.virtual_memory().available,
            }
        )
    except ImportError:
        hardware["ram_probe"] = "psutil não instalado"
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        hardware["nvidia_smi"] = gpu.stdout.strip() if gpu.returncode == 0 else "indisponível"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        hardware["nvidia_smi"] = "indisponível"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "versions": versions,
        "commit": git_output("rev-parse", "HEAD"),
        "dirty": bool(git_output("status", "--porcelain")),
        "code_sha256": fingerprint(code),
        "source_files": code,
        "hardware_observed": hardware,
    }
