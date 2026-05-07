"""Module-level shim that holds an implicit ``Run`` for the wandb-style API.

Library code should prefer ``claude_monitor.Run`` directly — globals don't
play well with concurrent runs or with libraries that mustn't reach into
their host process state.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from .client import Run, Span, ClaudeMonitorError

_current: Optional[Run] = None


def init(**kwargs: Any) -> Run:
    """Create the implicit ``Run`` used by the module-level ``log_*`` helpers.

    Re-calling ``init`` finishes any prior run (with ``outcome=neutral``) and
    starts a new one — same shape as ``wandb.init``.
    """
    global _current
    if _current is not None:
        try:
            _current.finish()
        except Exception:  # noqa: BLE001
            pass
    _current = Run(**kwargs)
    return _current


def current() -> Run:
    if _current is None:
        raise ClaudeMonitorError("no active run — call claude_monitor.init() first")
    return _current


def finish(**kwargs: Any) -> None:
    global _current
    if _current is None:
        return
    _current.finish(**kwargs)
    _current = None


def log(**kwargs: Any) -> Span:
    return current().log(**kwargs)


def log_user(text: str, **kw: Any) -> Span:
    return current().log_user(text, **kw)


def log_assistant(text: str, **kw: Any) -> Span:
    return current().log_assistant(text, **kw)


def log_thinking(text: str, **kw: Any) -> Span:
    return current().log_thinking(text, **kw)


def log_tool_use(
    tool: str,
    input: Optional[Mapping[str, Any]] = None,
    **kw: Any,
) -> Span:
    return current().log_tool_use(tool, input, **kw)


def log_tool_result(
    text: str,
    *,
    tool: str = "tool_result",
    parent_span_id: Optional[str] = None,
    **kw: Any,
) -> Span:
    return current().log_tool_result(
        text, tool=tool, parent_span_id=parent_span_id, **kw
    )


def log_attachment(
    name: str,
    attributes: Optional[Mapping[str, Any]] = None,
    **kw: Any,
) -> Span:
    return current().log_attachment(name, attributes, **kw)
