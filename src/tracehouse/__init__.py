"""tracehouse — push traces and spans from Python.

Two surfaces:

* **Module-level (wandb-style)**: ``init``, ``log_user``, ``log_assistant``,
  ``log_tool_use``, ``log_tool_result``, ``log_thinking``, ``finish``.
  One implicit ``Run`` lives on the module; great for scripts.

* **Class-based**: ``Run`` for explicit lifetimes, multiple concurrent runs,
  or library code where module globals are a footgun.

Both speak the same wire protocol (``POST /v1/traces`` and ``POST /v1/spans``)
and accept the same ``api_key`` / ``api_base`` / ``machine_id`` config.

Example::

    import tracehouse as cm

    cm.init(project="my-bot", session_id="run-001")
    cm.log_user("hello")
    cm.log_assistant("hi")
    cm.log_tool_use("Read", {"file_path": "x.py"})
    cm.log_tool_result("file contents")
    cm.finish(outcome="good", metadata={"k": "v"})

Logging
-------

The SDK uses ``logging.getLogger("tracehouse")`` (and sub-loggers
``tracehouse.client`` / ``tracehouse.training``). It does not call
``logging.basicConfig`` itself — that's the application's job. Typical
setup::

    import logging
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    logging.getLogger("tracehouse").setLevel(logging.DEBUG)  # verbose HTTP

INFO covers lifecycle events (run/trace created, finished, linked, pushed
to HF, artifacts stored). DEBUG adds every HTTP request/response.
WARNING fires for API errors and dropped metric points (NaN/Inf).
"""

from . import wandb  # noqa: F401 — re-exported so `from tracehouse import wandb` works
from .client import (
    ApiError,
    ClaudeMonitorError,
    Run,
    Span,
)
from .training import TrainingRun
from .media import Image, Video
from ._global import (
    init,
    log_user,
    log_assistant,
    log_tool_use,
    log_tool_result,
    log_thinking,
    log_attachment,
    log,
    finish,
    current,
    init_run,
    run_log,
    run_finish,
    current_run,
)

__all__ = [
    "ApiError",
    "ClaudeMonitorError",
    "Run",
    "Span",
    "TrainingRun",
    "Image",
    "Video",
    "wandb",
    "init",
    "log_user",
    "log_assistant",
    "log_tool_use",
    "log_tool_result",
    "log_thinking",
    "log_attachment",
    "log",
    "finish",
    "current",
    "init_run",
    "run_log",
    "run_finish",
    "current_run",
]

__version__ = "0.5.0"
