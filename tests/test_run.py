"""Test the SDK against an in-memory transport — no network required."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import pytest

import tracehouse as cm
from tracehouse.client import (
    ApiError,
    HttpResponse,
    Run,
    Span,
)


@dataclass
class FakeTransport:
    """Captures every (method, url, headers, body) and returns canned responses."""

    calls: List[dict] = field(default_factory=list)
    responses: List[HttpResponse] = field(default_factory=list)

    def push(self, status: int, body: Any) -> None:
        payload = json.dumps(body).encode() if not isinstance(body, bytes) else body
        self.responses.append(HttpResponse(status=status, body=payload))

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: Optional[bytes],
    ) -> HttpResponse:
        decoded = json.loads(body) if body else None
        self.calls.append(
            {"method": method, "url": url, "headers": dict(headers), "body": decoded}
        )
        if not self.responses:
            return HttpResponse(status=200, body=b"{}")
        return self.responses.pop(0)


@pytest.fixture
def transport() -> FakeTransport:
    t = FakeTransport()
    # Default: every request gets 200 with a sensible body.
    t.push(200, {"id": str(uuid.uuid4()), "session_id": "sess", "created": True})
    return t


def test_run_init_creates_trace_and_sends_auth_header(transport: FakeTransport):
    run = Run(api_key="ba_test", session_id="sess", project="p", transport=transport)
    assert run.trace_id is not None

    assert transport.calls[0]["method"] == "POST"
    assert transport.calls[0]["url"].endswith("/v1/traces")
    assert transport.calls[0]["headers"]["Authorization"] == "Bearer ba_test"
    body = transport.calls[0]["body"]
    assert body["session_id"] == "sess"
    assert body["project"] == "p"
    assert body["scaffold"] == "python-sdk"


def test_log_user_emits_span_with_text_attribute(transport: FakeTransport):
    transport.push(200, {})  # for the spans push
    run = Run(api_key="ba_test", session_id="sess", transport=transport)
    span = run.log_user("hello")
    assert span.kind == "user_msg"
    last = transport.calls[-1]
    assert last["url"].endswith("/v1/spans")
    payload = last["body"]
    assert payload["spans"][0]["kind"] == "user_msg"
    assert payload["spans"][0]["attributes"]["text"] == "hello"


def test_log_tool_use_and_result_chain(transport: FakeTransport):
    transport.push(200, {})  # tool_use
    transport.push(200, {})  # tool_result
    run = Run(api_key="ba_test", session_id="sess", transport=transport)
    use = run.log_tool_use("Read", {"file_path": "/etc/hosts"})
    result = run.log_tool_result("file contents", parent_span_id=use.id)

    assert use.attributes == {"tool_input": {"file_path": "/etc/hosts"}}
    assert result.parent_span_id == use.id
    # result_text lives under attributes per the wire format.
    payload = transport.calls[-1]["body"]
    assert payload["spans"][0]["attributes"]["result_text"] == "file contents"


def test_finish_patches_trace_with_outcome(transport: FakeTransport):
    transport.push(200, {"ok": True})  # for the patch
    run = Run(api_key="ba_test", session_id="sess", transport=transport)
    run.finish(outcome="good", metadata={"k": "v"})
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["url"].endswith(f"/v1/traces/{run.trace_id}")
    assert last["body"] == {"outcome": "good", "metadata": {"k": "v"}}


def test_finish_validates_outcome(transport: FakeTransport):
    run = Run(api_key="ba_test", session_id="sess", transport=transport)
    with pytest.raises(cm.ClaudeMonitorError):
        run.finish(outcome="excellent")  # type: ignore[arg-type]


def test_run_requires_api_key_when_anon_disabled(monkeypatch):
    monkeypatch.delenv("TRACEHOUSE_API_KEY", raising=False)
    monkeypatch.setenv("TRACEHOUSE_ANON", "0")
    with pytest.raises(cm.ClaudeMonitorError):
        Run(session_id="x")


def test_run_rejects_non_ba_key():
    with pytest.raises(cm.ClaudeMonitorError):
        Run(api_key="not_a_ba_key", session_id="x")


def test_run_anonymous_bootstraps_session_and_warns(monkeypatch, capsys):
    monkeypatch.delenv("TRACEHOUSE_API_KEY", raising=False)
    monkeypatch.setenv("TRACEHOUSE_WEB_URL", "https://web.example")
    import tracehouse.client as client

    client._anon_banner_shown = False  # reset one-time banner guard

    t = FakeTransport()
    # 1) anon session mint, 2) trace create
    t.push(200, {
        "user_id": "sentinel-1",
        "token": "anon_abc",
        "read_token": "anon_read_abc",
        "claim_token": "claim_xyz",
        "web_url": "https://web.example",
    })
    t.push(200, {"id": "trace-1", "session_id": "x", "created": True})

    run = Run(session_id="x", transport=t)

    # First call mints the anon session, with no real api key.
    assert t.calls[0]["url"].endswith("/v1/anon/session")
    # Subsequent calls carry the ingest bearer (NOT the read token).
    assert run.trace_id == "trace-1"
    assert t.calls[1]["headers"]["Authorization"] == "Bearer anon_abc"

    err = capsys.readouterr().err
    assert "YOU ARE NOT LOGGED IN" in err
    # Share link uses the read-only token, never the ingest bearer.
    assert "https://web.example/t/trace-1?t=anon_read_abc" in err
    assert "?t=anon_abc" not in err
    assert "https://web.example/claim?token=claim_xyz" in err


def test_api_error_on_non_2xx(transport: FakeTransport):
    transport.responses.clear()
    transport.push(401, {"error": "unauthorized"})
    with pytest.raises(ApiError) as excinfo:
        Run(api_key="ba_test", session_id="x", transport=transport)
    assert excinfo.value.status == 401


def test_kind_validation(transport: FakeTransport):
    run = Run(api_key="ba_test", session_id="sess", transport=transport)
    with pytest.raises(cm.ClaudeMonitorError):
        run.log(kind="not_a_kind", name="x")


def test_module_init_and_finish(transport: FakeTransport):
    transport.push(200, {})  # log_user spans push
    transport.push(200, {"ok": True})  # finish patch
    cm.init(api_key="ba_test", session_id="mod", transport=transport)
    cm.log_user("hi")
    cm.finish(outcome="neutral")
    # current() should now raise.
    with pytest.raises(cm.ClaudeMonitorError):
        cm.current()


def test_context_manager_marks_bad_on_exception(transport: FakeTransport):
    transport.push(200, {})  # span on exception path may not be sent — keep buffer
    transport.push(200, {"ok": True})  # finish patch
    with pytest.raises(RuntimeError):
        with Run(api_key="ba_test", session_id="ctx", transport=transport):
            raise RuntimeError("boom")
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["body"]["outcome"] == "bad"
    assert "boom" in last["body"]["metadata"]["error"]


def test_machine_id_header_present(transport: FakeTransport):
    Run(
        api_key="ba_test",
        session_id="mh",
        machine_id="machine-xyz",
        hostname="host-xyz",
        transport=transport,
    )
    h = transport.calls[0]["headers"]
    assert h["X-Claude-Monitor-Machine-Id"] == "machine-xyz"
    assert h["X-Claude-Monitor-Hostname"] == "host-xyz"


def test_span_dataclass_payload_skips_none_fields():
    s = Span(
        id="00000000-0000-0000-0000-000000000000",
        session_id="x",
        kind="user_msg",
        name="n",
        start_at="2026-05-08T00:00:00Z",
    )
    payload = s.to_payload()
    assert "end_at" not in payload
    assert "parent_span_id" not in payload
    assert "status" not in payload
