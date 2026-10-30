"""Medição amostral de memória do processo; RSS e memória privada são distintas."""

import functools
import threading
import time
from pathlib import Path

import psutil

from .utils import write_json


def profile_run(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        process = psutil.Process()
        stop = threading.Event()
        samples = {
            "peak_rss_bytes": 0,
            "peak_private_bytes": 0,
            "min_system_available_bytes": psutil.virtual_memory().available,
            "sample_interval_seconds": 0.5,
        }

        def sample():
            while True:
                memory = process.memory_info()
                samples["peak_rss_bytes"] = max(samples["peak_rss_bytes"], memory.rss)
                private = getattr(memory, "private", 0)
                samples["peak_private_bytes"] = max(samples["peak_private_bytes"], private)
                samples["min_system_available_bytes"] = min(
                    samples["min_system_available_bytes"], psutil.virtual_memory().available
                )
                if stop.wait(0.5):
                    return

        worker = threading.Thread(target=sample, daemon=True)
        worker.start()
        started = time.perf_counter()
        try:
            result = function(*args, **kwargs)
        finally:
            stop.set()
            worker.join()
        samples["wall_seconds_including_profiling"] = time.perf_counter() - started
        result["resources"] = samples
        write_json(Path(result["artifacts"]) / "metrics.json", result)
        return result

    return wrapped
