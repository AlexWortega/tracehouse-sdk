"""Environment snapshot + background resource monitor.

`capture_environment()` is a one-shot, best-effort dict of "what was this
process running on" — used by ``TrainingRun`` to attach an artifact named
``environment`` so a run is reproducible without the user having to write
it down. Every probe is wrapped in a try/except — missing tooling
(no ``nvidia-smi``, not in a git repo, etc.) silently degrades.

``SystemMonitor`` is a daemon thread that polls GPU / CPU / RAM / disk /
network at a configurable interval and pushes them as ``system/*`` metrics
into the run. Prefers ``psutil`` when available (cross-platform);
otherwise reads ``/proc`` directly on Linux; otherwise skips that metric.
GPU stats come from ``nvidia-smi --query-gpu=...`` so no python CUDA deps.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional

_log = logging.getLogger(__name__)

# Detect psutil once at import time. Without it we fall back to Linux /proc.
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:  # noqa: BLE001
    _HAS_PSUTIL = False


# --------------------------------------------------------------------------- #
# One-shot environment snapshot.
# --------------------------------------------------------------------------- #


def capture_environment() -> dict[str, Any]:
    """Best-effort: every probe lives inside try/except so failures degrade
    silently to ``null`` in the output dict."""
    return {
        "python": _python_info(),
        "os": _os_info(),
        "cpu": _cpu_info(),
        "memory_total_bytes": _mem_total(),
        "gpus": _gpu_info(),
        "cuda": _cuda_info(),
        "git": _git_info(),
        "packages": _ml_packages(),
        "env_vars": _selected_env(),
        "cwd": _safe(lambda: os.getcwd()),
        "argv": sys.argv,
        "hostname": platform.node(),
    }


def _safe(fn: Callable[[], Any], default: Any = None) -> Any:
    try:
        return fn()
    except Exception:  # noqa: BLE001
        return default


def _python_info() -> dict[str, Any]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _os_info() -> dict[str, Any]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def _cpu_info() -> dict[str, Any]:
    out: dict[str, Any] = {
        "count": _safe(os.cpu_count),
        "processor": platform.processor() or None,
    }
    # Try sysctl on macOS / /proc/cpuinfo on Linux for a model string.
    if platform.system() == "Linux":
        out["model"] = _safe(lambda: _grep_first("/proc/cpuinfo", "model name"))
    elif platform.system() == "Darwin":
        out["model"] = _safe(
            lambda: subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], timeout=2
            )
            .decode()
            .strip()
        )
    return out


def _grep_first(path: str, prefix: str) -> Optional[str]:
    with open(path) as f:
        for line in f:
            if line.startswith(prefix):
                _, _, val = line.partition(":")
                return val.strip()
    return None


def _mem_total() -> Optional[int]:
    if _HAS_PSUTIL:
        return _safe(lambda: psutil.virtual_memory().total)
    if platform.system() == "Linux":
        return _safe(lambda: int(_grep_first("/proc/meminfo", "MemTotal").split()[0]) * 1024)  # type: ignore[union-attr]
    return None


def _gpu_info() -> list[dict[str, Any]]:
    """List GPUs via ``nvidia-smi``. No python CUDA dep required.

    Returns an empty list when nvidia-smi isn't installed (e.g. CPU-only,
    AMD, Mac). For AMD/ROCm a future probe would shell out to ``rocm-smi``.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            timeout=3,
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:  # noqa: BLE001
        return []
    gpus = []
    for line in out.strip().splitlines():
        cells = [c.strip() for c in line.split(",")]
        if len(cells) < 3:
            continue
        gpus.append(
            {
                "index": int(cells[0]),
                "name": cells[1],
                "memory_total_mib": int(float(cells[2])),
                "driver_version": cells[3] if len(cells) > 3 else None,
                "compute_capability": cells[4] if len(cells) > 4 else None,
            }
        )
    return gpus


def _cuda_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # nvidia-smi headline includes driver + CUDA runtime versions.
    out["nvidia_smi"] = _safe(
        lambda: subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
        .decode()
        .strip()
        .splitlines()[0]
    )
    # Whatever the user's torch sees, if torch is importable. Doesn't import torch unless it's already loaded.
    torch = sys.modules.get("torch")
    if torch is not None:
        out["torch_cuda_version"] = _safe(lambda: torch.version.cuda)
        out["torch_cudnn_version"] = _safe(lambda: torch.backends.cudnn.version())
    return out


def _git_info() -> dict[str, Any]:
    def _run(args: list[str]) -> Optional[str]:
        try:
            return (
                subprocess.check_output(args, timeout=3, stderr=subprocess.DEVNULL)
                .decode()
                .strip()
            )
        except Exception:  # noqa: BLE001
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    if not sha:
        return {}
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty = _run(["git", "status", "--porcelain"])
    remote = _run(["git", "config", "--get", "remote.origin.url"])
    return {
        "commit": sha,
        "branch": branch,
        "dirty": bool(dirty) if dirty is not None else None,
        "remote": remote,
    }


# Names of installed packages relevant to ML training. We don't run `pip freeze`
# (slow + writes to stderr); just probe well-known imports already importable.
_ML_PROBES = (
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "datasets",
    "trl",
    "peft",
    "bitsandbytes",
    "deepspeed",
    "tensorflow",
    "jax",
    "flax",
    "numpy",
    "scipy",
    "scikit_learn",
    "lightning",
    "pytorch_lightning",
    "vllm",
    "xformers",
    "flash_attn",
)


def _ml_packages() -> dict[str, str]:
    """Versions of ML packages that ARE already imported in the running
    process. We don't force-import anything — that would change the user's
    process state, and `pip freeze`-style scans are too slow / noisy."""
    out: dict[str, str] = {}
    for name in _ML_PROBES:
        mod = sys.modules.get(name)
        if mod is None:
            continue
        v = getattr(mod, "__version__", None) or getattr(mod, "VERSION", None)
        if isinstance(v, str):
            out[name] = v
    return out


# Allow-list of env vars worth capturing: training topology / device choice /
# distributed coordination. NEVER capture secrets like *_TOKEN, *_KEY, etc.
_ENV_ALLOW = (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "NCCL_DEBUG",
    "NCCL_SOCKET_IFNAME",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "TOKENIZERS_PARALLELISM",
    "TRANSFORMERS_VERBOSITY",
    "HF_HOME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "ACCELERATE_USE_DEEPSPEED",
    "ACCELERATE_USE_FSDP",
    "PYTORCH_CUDA_ALLOC_CONF",
)


def _selected_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _ENV_ALLOW if k in os.environ}


# --------------------------------------------------------------------------- #
# Background resource monitor.
# --------------------------------------------------------------------------- #


class SystemMonitor:
    """Polls GPU/CPU/RAM/disk/net at ``interval`` seconds and emits scalar
    metrics through ``log_fn`` (usually ``TrainingRun.log`` bound with
    ``commit=True``).

    Metric naming follows wandb's ``system/*`` convention so dashboards
    stay readable next to user metrics:

      * ``system/cpu.util_pct``
      * ``system/ram.used_pct``  + ``system/ram.used_bytes``
      * ``system/disk.used_pct`` + ``system/disk.used_bytes``
      * ``system/gpu.<i>.util_pct`` + ``system/gpu.<i>.mem_used_pct``
      * ``system/gpu.<i>.mem_used_mib`` + ``system/gpu.<i>.temp_c``
      * ``system/net.bytes_sent`` + ``system/net.bytes_recv`` (cumulative)
    """

    def __init__(
        self,
        log_fn: Callable[[Mapping[str, Any]], None],
        *,
        interval: float = 15.0,
    ) -> None:
        self._log_fn = log_fn
        self._interval = max(2.0, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Network counters require deltas, but we report cumulative here to
        # match wandb. Disk path defaults to root.
        self._disk_path = "/"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="cm-system-monitor", daemon=True
        )
        self._thread.start()
        _log.debug("system monitor started (interval=%.1fs)", self._interval)

    def stop(self, timeout: float = 2.0) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._thread = None
        _log.debug("system monitor stopped")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                payload = self._sample()
                if payload:
                    try:
                        self._log_fn(payload)
                    except Exception as e:  # noqa: BLE001
                        _log.debug("system metric flush failed: %s", e)
            except Exception as e:  # noqa: BLE001
                # Don't let a monitor failure kill the training process.
                _log.debug("system sample failed: %s", e)
            self._stop.wait(self._interval)

    def _sample(self) -> dict[str, float]:
        out: dict[str, float] = {}
        # ---- CPU ----
        cpu = self._cpu_util()
        if cpu is not None:
            out["system/cpu.util_pct"] = cpu
        # ---- RAM ----
        ram = self._ram()
        if ram:
            out["system/ram.used_pct"] = ram["used_pct"]
            out["system/ram.used_bytes"] = ram["used_bytes"]
        # ---- Disk ----
        disk = self._disk()
        if disk:
            out["system/disk.used_pct"] = disk["used_pct"]
            out["system/disk.used_bytes"] = disk["used_bytes"]
        # ---- Net (cumulative bytes since boot) ----
        net = self._net()
        if net:
            out["system/net.bytes_sent"] = net["bytes_sent"]
            out["system/net.bytes_recv"] = net["bytes_recv"]
        # ---- GPU ----
        for gpu in self._gpu_sample():
            i = gpu["index"]
            out[f"system/gpu.{i}.util_pct"] = gpu["util_pct"]
            out[f"system/gpu.{i}.mem_used_pct"] = gpu["mem_used_pct"]
            out[f"system/gpu.{i}.mem_used_mib"] = gpu["mem_used_mib"]
            if gpu.get("temp_c") is not None:
                out[f"system/gpu.{i}.temp_c"] = gpu["temp_c"]
        return out

    # ----- per-probe helpers ----- #

    def _cpu_util(self) -> Optional[float]:
        if _HAS_PSUTIL:
            try:
                return float(psutil.cpu_percent(interval=None))
            except Exception:  # noqa: BLE001
                return None
        # No psutil → use load avg as a rough proxy. Linux + macOS support it.
        try:
            loads = os.getloadavg()
            cores = os.cpu_count() or 1
            return min(100.0, loads[0] / cores * 100.0)
        except Exception:  # noqa: BLE001
            return None

    def _ram(self) -> Optional[dict[str, float]]:
        if _HAS_PSUTIL:
            try:
                m = psutil.virtual_memory()
                return {"used_pct": float(m.percent), "used_bytes": float(m.used)}
            except Exception:  # noqa: BLE001
                return None
        if platform.system() == "Linux":
            try:
                total = avail = None
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            total = int(line.split()[1]) * 1024
                        elif line.startswith("MemAvailable:"):
                            avail = int(line.split()[1]) * 1024
                        if total is not None and avail is not None:
                            break
                if total and avail is not None:
                    used = total - avail
                    return {
                        "used_pct": used / total * 100.0,
                        "used_bytes": float(used),
                    }
            except Exception:  # noqa: BLE001
                pass
        return None

    def _disk(self) -> Optional[dict[str, float]]:
        try:
            usage = shutil.disk_usage(self._disk_path)
            return {
                "used_pct": usage.used / usage.total * 100.0,
                "used_bytes": float(usage.used),
            }
        except Exception:  # noqa: BLE001
            return None

    def _net(self) -> Optional[dict[str, float]]:
        if _HAS_PSUTIL:
            try:
                n = psutil.net_io_counters()
                return {"bytes_sent": float(n.bytes_sent), "bytes_recv": float(n.bytes_recv)}
            except Exception:  # noqa: BLE001
                return None
        if platform.system() == "Linux":
            try:
                sent = recv = 0
                with open("/proc/net/dev") as f:
                    # Skip 2 header lines.
                    next(f)
                    next(f)
                    for line in f:
                        iface, _, rest = line.partition(":")
                        cols = rest.split()
                        if iface.strip() == "lo":
                            continue
                        # /proc/net/dev: recv-bytes recv-packets ... 8 cols ... send-bytes ...
                        recv += int(cols[0])
                        sent += int(cols[8])
                return {"bytes_sent": float(sent), "bytes_recv": float(recv)}
            except Exception:  # noqa: BLE001
                return None
        return None

    def _gpu_sample(self) -> list[dict[str, float]]:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                timeout=3,
                stderr=subprocess.DEVNULL,
            ).decode()
        except Exception:  # noqa: BLE001
            return []
        gpus = []
        for line in out.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) < 4:
                continue
            try:
                i = int(cells[0])
                util = float(cells[1])
                mem_used = float(cells[2])
                mem_total = float(cells[3])
                temp = float(cells[4]) if len(cells) > 4 and cells[4] not in ("[N/A]", "") else None
            except ValueError:
                continue
            gpus.append(
                {
                    "index": i,
                    "util_pct": util,
                    "mem_used_mib": mem_used,
                    "mem_used_pct": (mem_used / mem_total * 100.0) if mem_total else 0.0,
                    "temp_c": temp,
                }
            )
        return gpus
