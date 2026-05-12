"""claude-monitor — push traces and spans from Python.

Two surfaces:

* **Module-level (wandb-style)**: ``init``, ``log_user``, ``log_assistant``,
  ``log_tool_use``, ``log_tool_result``, ``log_thinking``, ``finish``.
  One implicit ``Run`` lives on the module; great for scripts.

* **Class-based**: ``Run`` for explicit lifetimes, multiple concurrent runs,
  or library code where module globals are a footgun.

Both speak the same wire protocol (``POST /v1/traces`` and ``POST /v1/spans``)
and accept the same ``api_key`` / ``api_base`` / ``machine_id`` config.

Example::

    import claude_monitor as cm

    cm.init(project="my-bot", session_id="run-001")
    cm.log_user("hello")
    cm.log_assistant("hi")
    cm.log_tool_use("Read", {"file_path": "x.py"})
    cm.log_tool_result("file contents")
    cm.finish(outcome="good", metadata={"k": "v"})
"""

from .client import (
    ApiError,
    ClaudeMonitorError,
    Run,
    Span,
)
from .training import TrainingRun
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

__version__ = "0.1.0"
