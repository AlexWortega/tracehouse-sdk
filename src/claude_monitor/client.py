"""HTTP client + Run / Span types.

Stdlib-only — relies on ``urllib.request`` so the SDK has zero install deps.
``Transport`` is injectable so tests can substitute an in-memory mock.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import platform
import socket
import ssl
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Optional

# Library code follows the stdlib pattern: a single module-level logger per
# file, namespaced under ``claude_monitor.<module>``. Users wire it up via
# ``logging.basicConfig(level=logging.DEBUG)`` or ``logging.getLogger(
# "claude_monitor").setLevel(logging.INFO)``. The SDK never calls
# ``basicConfig`` itself — that's the application's job.
_log = logging.getLogger(__name__)


SPAN_KINDS = (
    "user_msg",
    "assistant_msg",
    "tool_use",
    "tool_result",
    "thinking",
    "attachment",
)
OUTCOMES = ("good", "bad", "neutral")


class ClaudeMonitorError(Exception):
    """Base error for everything raised by this SDK."""


class ApiError(ClaudeMonitorError):
    """Non-2xx response from the claude-monitor API."""

    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _machine_id_default() -> str:
    """Stable per-machine identifier. Hashed hostname is fine for grouping."""
    return f"{platform.node() or 'unknown'}-{platform.system()}".lower()


# --------------------------------------------------------------------------- #
# Transport — pluggable so tests can capture requests without hitting network.
# --------------------------------------------------------------------------- #


@dataclass
class HttpResponse:
    status: int
    body: bytes


Transport = Callable[[str, str, Mapping[str, str], Optional[bytes]], HttpResponse]


def _urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: Optional[bytes],
) -> HttpResponse:
    req = urllib.request.Request(url=url, data=body, headers=dict(headers), method=method)
    ctx = ssl.create_default_context() if url.startswith("https://") else None
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            return HttpResponse(status=resp.status, body=resp.read())
    except urllib.error.HTTPError as e:  # 4xx / 5xx
        return HttpResponse(status=e.code, body=e.read() if e.fp else b"")
    except (urllib.error.URLError, socket.timeout) as e:
        raise ClaudeMonitorError(f"network error: {e}") from e


def _resolve_api_base(api_base: Optional[str]) -> str:
    return (
        api_base
        or os.environ.get("CLAUDE_MONITOR_API_BASE")
        or "https://api-production-a0da.up.railway.app"
    ).rstrip("/")


def _validate_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.environ.get("CLAUDE_MONITOR_API_KEY")
    if not key:
        raise ClaudeMonitorError(
            "api_key is required (or set CLAUDE_MONITOR_API_KEY)"
        )
    if not key.startswith("ba_"):
        raise ClaudeMonitorError(
            "api_key should start with 'ba_' — did you paste a session token by mistake?"
        )
    return key


def _do_request(
    *,
    transport: Transport,
    api_base: str,
    api_key: str,
    machine_id: str,
    hostname: Optional[str],
    agent_version: Optional[str],
    method: str,
    path: str,
    body: Optional[Any],
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
        "X-Claude-Monitor-Machine-Id": machine_id,
    }
    if hostname:
        headers["X-Claude-Monitor-Hostname"] = hostname
    if agent_version:
        headers["X-Claude-Monitor-Agent-Version"] = agent_version
    payload = json.dumps(body).encode("utf-8") if body is not None else None
    _log.debug(
        "→ %s %s (body=%d bytes)", method, path, len(payload) if payload else 0
    )
    resp = transport(method, f"{api_base}{path}", headers, payload)
    if resp.status >= 400:
        text = resp.body.decode("utf-8", errors="replace")
        _log.warning(
            "← %s %s %d: %s", method, path, resp.status, text[:200]
        )
        raise ApiError(resp.status, text)
    _log.debug(
        "← %s %s %d (body=%d bytes)", method, path, resp.status, len(resp.body)
    )
    if not resp.body:
        return {}
    try:
        return json.loads(resp.body)
    except json.JSONDecodeError:
        return {"raw": resp.body.decode("utf-8", errors="replace")}


# --------------------------------------------------------------------------- #
# Span / Run.
# --------------------------------------------------------------------------- #


@dataclass
class Span:
    id: str
    session_id: str
    kind: str
    name: str
    start_at: str
    end_at: Optional[str] = None
    parent_span_id: Optional[str] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    status: Optional[str] = None

    def to_payload(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "session_id": self.session_id,
            "kind": self.kind,
            "name": self.name,
            "start_at": self.start_at,
            "attributes": self.attributes,
        }
        if self.end_at is not None:
            out["end_at"] = self.end_at
        if self.parent_span_id is not None:
            out["parent_span_id"] = self.parent_span_id
        if self.status is not None:
            out["status"] = self.status
        return out


class Run:
    """A single trace + the spans being pushed into it.

    A ``Run`` is identified by its ``session_id``. The server upserts on
    ``(user, machine, session_id)``, so re-instantiating with the same
    ``session_id`` resumes the existing trace.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        project: Optional[str] = None,
        session_id: Optional[str] = None,
        scaffold: Optional[str] = None,
        task_name: Optional[str] = None,
        model: Optional[str] = None,
        machine_id: Optional[str] = None,
        hostname: Optional[str] = None,
        agent_version: Optional[str] = None,
        transport: Optional[Transport] = None,
        auto_create: bool = True,
    ) -> None:
        self._api_key = _validate_api_key(api_key)
        self._api_base = _resolve_api_base(api_base)
        self._transport = transport or _urllib_transport

        self.session_id = session_id or f"py-{uuid.uuid4()}"
        self.project = project
        self.scaffold = scaffold or "python-sdk"
        self.task_name = task_name
        self.model = model
        self.machine_id = machine_id or _machine_id_default()
        self.hostname = hostname or platform.node() or None
        self.agent_version = agent_version

        self.trace_id: Optional[str] = None
        self._closed = False

        if auto_create:
            self._create_trace()

    # ----- HTTP ----------------------------------------------------------- #

    def _request(self, method: str, path: str, body: Optional[Any]) -> dict[str, Any]:
        return _do_request(
            transport=self._transport,
            api_base=self._api_base,
            api_key=self._api_key,
            machine_id=self.machine_id,
            hostname=self.hostname,
            agent_version=self.agent_version,
            method=method,
            path=path,
            body=body,
        )

    # ----- lifecycle ------------------------------------------------------ #

    def _create_trace(self) -> None:
        body: dict[str, Any] = {"session_id": self.session_id}
        if self.project is not None:
            body["project"] = self.project
        if self.scaffold is not None:
            body["scaffold"] = self.scaffold
        if self.task_name is not None:
            body["task_name"] = self.task_name
        if self.model is not None:
            body["model"] = self.model
        body["started_at"] = _utcnow_iso()
        resp = self._request("POST", "/v1/traces", body)
        self.trace_id = resp.get("id")
        if not self.trace_id:
            raise ClaudeMonitorError(
                f"server did not return a trace id (response: {resp!r})"
            )
        _log.info(
            "trace ready: id=%s session=%s%s",
            self.trace_id,
            self.session_id,
            " (resumed)" if resp.get("created") is False else "",
        )

    def finish(
        self,
        *,
        outcome: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        task_name: Optional[str] = None,
        model: Optional[str] = None,
        scaffold: Optional[str] = None,
    ) -> None:
        if self._closed:
            return
        if outcome is not None and outcome not in OUTCOMES:
            raise ClaudeMonitorError(
                f"outcome must be one of {OUTCOMES}, got {outcome!r}"
            )
        if not self.trace_id:
            raise ClaudeMonitorError("run was not created — no trace_id")
        patch: dict[str, Any] = {}
        if outcome is not None:
            patch["outcome"] = outcome
        if metadata is not None:
            patch["metadata"] = dict(metadata)
        if task_name is not None:
            patch["task_name"] = task_name
        if model is not None:
            patch["model"] = model
        if scaffold is not None:
            patch["scaffold"] = scaffold
        if patch:
            self._request("PATCH", f"/v1/traces/{self.trace_id}", patch)
        self._closed = True
        _log.info(
            "trace finished: id=%s%s",
            self.trace_id,
            f" outcome={outcome}" if outcome else "",
        )

    # ----- span helpers --------------------------------------------------- #

    def log(
        self,
        *,
        kind: str,
        name: str,
        text: Optional[str] = None,
        attributes: Optional[Mapping[str, Any]] = None,
        parent_span_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Span:
        if kind not in SPAN_KINDS:
            raise ClaudeMonitorError(
                f"kind must be one of {SPAN_KINDS}, got {kind!r}"
            )
        attrs: dict[str, Any] = dict(attributes or {})
        if text is not None and "text" not in attrs and "result_text" not in attrs:
            attrs["text"] = text
        span = Span(
            id=str(uuid.uuid4()),
            session_id=self.session_id,
            kind=kind,
            name=name,
            start_at=_utcnow_iso(),
            end_at=_utcnow_iso(),
            parent_span_id=parent_span_id,
            attributes=attrs,
            status=status,
        )
        self.push_spans([span])
        return span

    def log_user(self, text: str, **kw: Any) -> Span:
        return self.log(kind="user_msg", name="user message", text=text, **kw)

    def log_assistant(self, text: str, **kw: Any) -> Span:
        return self.log(kind="assistant_msg", name="assistant message", text=text, **kw)

    def log_thinking(self, text: str, **kw: Any) -> Span:
        return self.log(kind="thinking", name="thinking", text=text, **kw)

    def log_tool_use(
        self,
        tool: str,
        input: Optional[Mapping[str, Any]] = None,
        **kw: Any,
    ) -> Span:
        return self.log(
            kind="tool_use",
            name=tool,
            attributes={"tool_input": dict(input or {})},
            **kw,
        )

    def log_tool_result(
        self,
        text: str,
        *,
        tool: str = "tool_result",
        parent_span_id: Optional[str] = None,
        **kw: Any,
    ) -> Span:
        return self.log(
            kind="tool_result",
            name=tool,
            attributes={"result_text": text},
            parent_span_id=parent_span_id,
            **kw,
        )

    def log_attachment(
        self,
        name: str,
        attributes: Optional[Mapping[str, Any]] = None,
        **kw: Any,
    ) -> Span:
        return self.log(
            kind="attachment",
            name=name,
            attributes=dict(attributes or {}),
            **kw,
        )

    def push_spans(self, spans: Iterable[Span]) -> None:
        items = [s.to_payload() for s in spans]
        if not items:
            return
        _log.debug("pushing %d span(s) to /v1/spans", len(items))
        self._request(
            "POST",
            "/v1/spans",
            {"traces": [], "spans": items},
        )

    # ----- context manager ------------------------------------------------ #

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc is not None and self.trace_id:
            try:
                self.finish(
                    outcome="bad",
                    metadata={"error": f"{exc_type.__name__}: {exc}"},
                )
            except Exception:  # noqa: BLE001 — don't mask the original
                pass
        else:
            try:
                self.finish()
            except Exception:  # noqa: BLE001
                pass
