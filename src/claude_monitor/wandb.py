"""claude_monitor.wandb — wandb-compatible API surface, hosted under our
namespace (no separate package).

Drop-in for the wandb call sites most user code actually uses::

    from claude_monitor import wandb         # or: import claude_monitor.wandb as wandb

    run = wandb.init(project="demo", name="qwen-sft",
                     config={"lr": 1e-4, "batch": 32})
    for step in range(1000):
        wandb.log({"train/loss": loss, "eval/acc": acc}, step=step)
        # wandb.Histogram works for gradient/activation distributions:
        wandb.log({"grad/norm": wandb.Histogram(grad_norms)}, step=step)
    wandb.config.update({"warmup_ratio": 0.03})
    wandb.summary["best_loss"] = best_loss
    wandb.finish()

What's wired through to the claude-monitor backend:
  * init / log / finish / config / summary / run / Run / Histogram
  * Metric values land in the same /v1/runs/:id/metrics ingest path as
    the native ``claude_monitor.init_run`` flow.

What's accepted-and-ignored for source compatibility (won't blow up your
existing wandb code):
  * entity / tags / notes / dir / save_code / settings / job_type —
    captured into ``config`` so they're still inspectable, but have no
    semantic effect server-side.
  * Image / Video / Audio / Table — accepted by ``log``, stored as JSON
    placeholders. Real media support TBD.
  * watch() / save() / unwatch() / login() — no-ops.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Mapping, Optional

from claude_monitor.client import ApiError, ClaudeMonitorError
from claude_monitor.training import TrainingRun

__all__ = [
    # core
    "init",
    "log",
    "finish",
    "config",
    "summary",
    "run",
    "Run",
    "define_metric",
    # media / data wrappers
    "Histogram",
    "Image",
    "Video",
    "Audio",
    "Table",
    # noops for source compat
    "save",
    "watch",
    "unwatch",
    "login",
    "alert",
    # constants
    "ONLINE",
    "OFFLINE",
    "DISABLED",
    # exceptions
    "ApiError",
    "ClaudeMonitorError",
]

__version__ = "0.5.1"

ONLINE = "online"
OFFLINE = "offline"
DISABLED = "disabled"


class _MetricDef:
    """Lightweight return value for ``define_metric`` — wandb returns a Metric
    object that most code ignores; we mirror just its identity."""

    def __init__(self, name: str, spec: Mapping[str, Any]) -> None:
        self.name = name
        self.spec = dict(spec)

    def __repr__(self) -> str:
        return f"<MetricDef {self.name!r} {self.spec!r}>"


# --------------------------------------------------------------------------- #
# Run wrapper.
# --------------------------------------------------------------------------- #


class Run:
    """wandb-shaped Run. Thin wrapper around ``TrainingRun`` exposing the
    attributes wandb consumers expect (``id``, ``name``, ``project``,
    ``config``, ``summary``, ``url``, ``log``, ``finish``)."""

    def __init__(self, inner: TrainingRun) -> None:
        self._inner = inner

    # --- read-only identity ------------------------------------------------ #
    @property
    def id(self) -> str:
        return self._inner.run_id or ""

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def project(self) -> Optional[str]:
        return self._inner.project

    @property
    def config(self) -> "_ConfigProxy":
        return _config_proxy

    @property
    def summary(self) -> "_SummaryProxy":
        return _summary_proxy

    @property
    def url(self) -> str:
        """Best-effort link to /runs/<id> on the web frontend."""
        base = os.environ.get(
            "CLAUDE_MONITOR_WEB_URL",
            "https://clewe.ai",
        ).rstrip("/")
        return f"{base}/runs/{self.id}" if self.id else base

    # --- behaviour --------------------------------------------------------- #
    def log(
        self,
        values: Mapping[str, Any],
        step: Optional[int] = None,
        commit: bool = True,
        **_kwargs: Any,
    ) -> None:
        self._inner.log(_unwrap_media(values), step=step, commit=commit)

    def finish(self, exit_code: int = 0, quiet: Any = None) -> None:  # noqa: ARG002
        status = "finished" if not exit_code else "failed"
        self._inner.finish(status=status)

    def define_metric(
        self,
        name: str,
        *,
        step_metric: Optional[str] = None,
        summary: Optional[str] = None,
        goal: Optional[str] = None,
        hidden: Optional[bool] = None,
        **_kw: Any,
    ) -> "_MetricDef":
        """``wandb.define_metric`` parity. We record the definition into the
        run config (under ``_metric_defs``) so it's inspectable in the Config
        tab; server-side summary aggregation by it is best-effort/TBD."""
        spec = {
            k: v
            for k, v in {
                "step_metric": step_metric,
                "summary": summary,
                "goal": goal,
                "hidden": hidden,
            }.items()
            if v is not None
        }
        try:
            defs = dict(self._inner.config.get("_metric_defs", {}))
            defs[name] = spec
            self._inner.config["_metric_defs"] = defs
        except Exception:
            pass
        return _MetricDef(name, spec)

    # --- HF link parity with cm SDK ---------------------------------------- #
    def link_dataset(self, hf_slug: str, *, split: Optional[str] = None) -> None:
        self._inner.link_dataset(hf_slug, split=split)

    def link_model(self, hf_repo: str, *, revision: Optional[str] = None) -> None:
        self._inner.link_model(hf_repo, revision=revision)

    def __repr__(self) -> str:
        return f"<claude_monitor.Run id={self.id!r} name={self.name!r}>"


class _DisabledRun:
    """mode='disabled' stub. Every method swallows arguments silently —
    matches wandb's "I'm offline, don't crash my training loop" contract."""

    id = ""
    name = "disabled"
    project: Optional[str] = None
    url = ""

    @property
    def config(self) -> "_ConfigProxy":
        return _config_proxy

    @property
    def summary(self) -> "_SummaryProxy":
        return _summary_proxy

    def log(self, *_a: Any, **_k: Any) -> None: ...
    def finish(self, *_a: Any, **_k: Any) -> None: ...
    def link_dataset(self, *_a: Any, **_k: Any) -> None: ...
    def link_model(self, *_a: Any, **_k: Any) -> None: ...
    def define_metric(self, name: str, **_k: Any) -> "_MetricDef":
        return _MetricDef(name, {})


# --------------------------------------------------------------------------- #
# Config + summary proxies.
# --------------------------------------------------------------------------- #


class _ConfigProxy:
    """wandb.config-shaped: dict, attribute access, ``.update()``. Mutations
    PATCH the run server-side so the change persists."""

    def __init__(self) -> None:
        # All real attributes go through object.__setattr__ to avoid
        # recursing into our own __setattr__ override.
        object.__setattr__(self, "_run", None)

    def _bind(self, run: Optional[TrainingRun]) -> None:
        object.__setattr__(self, "_run", run)

    def _patch(self, values: Mapping[str, Any]) -> None:
        run: Optional[TrainingRun] = object.__getattribute__(self, "_run")
        if run is None or not run.run_id:
            return
        run.config.update(values)
        try:
            run._request(
                "PATCH",
                f"/v1/runs/{run.run_id}",
                {"config_patch": dict(values)},
            )
        except Exception:
            # Don't take down a training loop because the config write failed.
            pass

    # dict-like
    def update(self, values: Mapping[str, Any], allow_val_change: bool = True) -> None:  # noqa: ARG002
        self._patch(values)

    def __getitem__(self, key: str) -> Any:
        run: Optional[TrainingRun] = object.__getattribute__(self, "_run")
        if run is None:
            raise KeyError(key)
        return run.config[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._patch({key: value})

    def __contains__(self, key: str) -> bool:
        run: Optional[TrainingRun] = object.__getattribute__(self, "_run")
        return run is not None and key in run.config

    def as_dict(self) -> dict[str, Any]:
        run: Optional[TrainingRun] = object.__getattribute__(self, "_run")
        return dict(run.config) if run is not None else {}

    def keys(self) -> Iterable[str]:
        return self.as_dict().keys()

    # attribute access
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._patch({name: value})

    def __repr__(self) -> str:
        return f"<claude_monitor.config {self.as_dict()!r}>"


class _SummaryProxy:
    """wandb.run.summary-shaped: writes land in ``runs.summary`` via a
    metric ingest call (the server keeps a last-value-per-key cache there).
    Reads aren't supported (no client-side mirror)."""

    def __init__(self) -> None:
        object.__setattr__(self, "_run", None)

    def _bind(self, run: Optional[TrainingRun]) -> None:
        object.__setattr__(self, "_run", run)

    def update(self, values: Mapping[str, Any]) -> None:
        run: Optional[TrainingRun] = object.__getattribute__(self, "_run")
        if run is None:
            return
        run.log(dict(values))

    def __setitem__(self, key: str, value: Any) -> None:
        self.update({key: value})

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self.update({name: value})


_config_proxy = _ConfigProxy()
_summary_proxy = _SummaryProxy()


# Always-current `run` attribute via module __getattr__.
_current: Optional[Any] = None  # Run | _DisabledRun | None


def __getattr__(name: str) -> Any:
    if name == "run":
        return _current
    if name == "config":
        return _config_proxy
    if name == "summary":
        return _summary_proxy
    raise AttributeError(f"module 'claude_monitor.wandb' has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# Media wrappers.
# --------------------------------------------------------------------------- #


class Histogram:
    """Compatibility shim for ``wandb.Histogram``. When logged, ships as a
    JSON object {bins, counts} that the dashboard renders as a bar chart."""

    def __init__(self, sequence: Any = None, *, num_bins: int = 64, np_histogram: Any = None) -> None:
        if np_histogram is not None:
            counts, bins = np_histogram
            self.counts = list(counts)
            self.bins = list(bins)
            return
        if sequence is None:
            self.counts = []
            self.bins = []
            return
        try:
            import numpy as np  # type: ignore

            counts, bins = np.histogram(np.asarray(sequence), bins=num_bins)
            self.counts = counts.tolist()
            self.bins = bins.tolist()
        except Exception:
            seq = list(sequence) if hasattr(sequence, "__iter__") else []
            self.counts = seq
            self.bins = []

    def __json__(self) -> dict[str, Any]:
        return {"_type": "histogram", "counts": self.counts, "bins": self.bins}


class _MediaStub:
    """Base for Image / Video / Audio — accepted by log(), no host yet."""

    _type = "media"

    def __init__(self, data_or_path: Any = None, *_a: Any, caption: Optional[str] = None, **_kw: Any) -> None:
        self.path = (
            str(data_or_path)
            if isinstance(data_or_path, (str, os.PathLike))
            else None
        )
        self.caption = caption

    def __json__(self) -> dict[str, Any]:
        return {
            "_type": self._type,
            "path": self.path,
            "caption": self.caption,
            "note": "media hosting not yet implemented in claude-monitor",
        }


class Image(_MediaStub):
    _type = "image"


class Video(_MediaStub):
    _type = "video"


class Audio(_MediaStub):
    _type = "audio"


class Table:
    """``wandb.Table`` shim. Stored as JSON: {columns, data}."""

    def __init__(self, columns: Optional[list[str]] = None, data: Optional[list[list[Any]]] = None, **_kw: Any) -> None:
        self.columns = list(columns or [])
        self.data: list[list[Any]] = [list(r) for r in (data or [])]

    def add_data(self, *row: Any) -> None:
        self.data.append(list(row))

    def __json__(self) -> dict[str, Any]:
        return {"_type": "table", "columns": self.columns, "data": self.data}


def _unwrap_media(values: Mapping[str, Any]) -> dict[str, Any]:
    """Convert media wrappers to their JSON shape for transport."""
    out: dict[str, Any] = {}
    for k, v in values.items():
        if hasattr(v, "__json__") and callable(v.__json__):
            out[k] = v.__json__()
        else:
            out[k] = v
    return out


# --------------------------------------------------------------------------- #
# Top-level API.
# --------------------------------------------------------------------------- #


# Keys that wandb users pass to init() but we accept-and-stash into config so
# their code keeps working. Each is captured under wandb_<key>.
_PASSTHROUGH_KEYS = ("entity", "tags", "notes", "job_type", "group")


def init(
    project: Optional[str] = None,
    *,
    entity: Optional[str] = None,
    name: Optional[str] = None,
    id: Optional[str] = None,  # noqa: A002 — wandb signature
    config: Optional[Mapping[str, Any]] = None,
    resume: Any = None,  # noqa: ARG001 — accepted for source compat
    mode: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    notes: Optional[str] = None,
    dir: Any = None,  # noqa: A002, ARG001
    save_code: Any = None,  # noqa: ARG001
    settings: Any = None,  # noqa: ARG001
    job_type: Optional[str] = None,
    group: Optional[str] = None,
    reinit: Any = None,  # noqa: ARG001
    anonymous: Any = None,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """``wandb.init`` drop-in. Most kwargs are accepted; only the ones with
    a sensible mapping (project / name / config / id / mode) actually do
    anything. ``mode='disabled'`` returns a no-op run."""
    global _current

    if mode == DISABLED:
        _current = _DisabledRun()
        _config_proxy._bind(None)
        _summary_proxy._bind(None)
        return _current

    # Honor the env vars wandb users set in CI.
    project = project or os.environ.get("WANDB_PROJECT") or os.environ.get("CLAUDE_MONITOR_PROJECT")
    name = name or os.environ.get("WANDB_NAME")
    id_ = id or os.environ.get("WANDB_RUN_ID")

    # Re-initialising while a run is open mirrors wandb: finish the old one first.
    if _current is not None and isinstance(_current, Run):
        try:
            _current.finish()
        except Exception:
            pass

    # Stash the ignored-but-accepted kwargs into config so they show up in
    # the dashboard's Config tab instead of getting silently dropped.
    cfg: dict[str, Any] = dict(config or {})
    if entity is not None:
        cfg.setdefault("wandb_entity", entity)
    if tags is not None:
        cfg.setdefault("wandb_tags", list(tags))
    if notes is not None:
        cfg.setdefault("wandb_notes", notes)
    if job_type is not None:
        cfg.setdefault("wandb_job_type", job_type)
    if group is not None:
        cfg.setdefault("wandb_group", group)

    inner = TrainingRun(
        project=project,
        name=name,
        config=cfg,
        client_run_id=id_,
    )
    _current = Run(inner)
    _config_proxy._bind(inner)
    _summary_proxy._bind(inner)
    return _current


def log(values: Mapping[str, Any], step: Optional[int] = None, commit: bool = True, **_kw: Any) -> None:
    if _current is None:
        raise ClaudeMonitorError("no active run — call claude_monitor.wandb.init() first")
    _current.log(values, step=step, commit=commit)


def finish(exit_code: int = 0, quiet: Any = None) -> None:  # noqa: ARG001
    global _current
    if _current is None:
        return
    _current.finish(exit_code=exit_code)
    _current = None
    _config_proxy._bind(None)
    _summary_proxy._bind(None)


def define_metric(name: str, **kwargs: Any) -> Any:
    """Module-level ``wandb.define_metric`` — delegates to the active run, or
    is a harmless no-op if no run is open yet."""
    if _current is None:
        return _MetricDef(name, {})
    return _current.define_metric(name, **kwargs)


# --------------------------------------------------------------------------- #
# Compat no-ops — accept-and-ignore.
# --------------------------------------------------------------------------- #


def save(*_a: Any, **_kw: Any) -> None:
    """No file hosting yet; accept the call so user code doesn't crash."""


def watch(*_a: Any, **_kw: Any) -> None:
    """``wandb.watch`` is PyTorch-specific gradient logging; out of scope."""


def unwatch(*_a: Any, **_kw: Any) -> None: ...


def login(*_a: Any, **_kw: Any) -> bool:
    """Always-true: claude-monitor auth uses env var / api_key argument."""
    return True


def alert(*_a: Any, **_kw: Any) -> None:
    """No alert plumbing yet."""
