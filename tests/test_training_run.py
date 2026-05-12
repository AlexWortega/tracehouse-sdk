"""TrainingRun tests against an in-memory transport."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import pytest

import claude_monitor as cm
from claude_monitor.client import ApiError, HttpResponse
from claude_monitor.training import TrainingRun


@dataclass
class FakeTransport:
    calls: List[dict] = field(default_factory=list)
    responses: List[HttpResponse] = field(default_factory=list)

    def push(self, status: int, body: Any) -> None:
        if isinstance(body, bytes):
            payload = body
        elif isinstance(body, str):
            payload = body.encode("utf-8")
        else:
            payload = json.dumps(body).encode()
        self.responses.append(HttpResponse(status=status, body=payload))

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: Optional[bytes],
    ) -> HttpResponse:
        try:
            decoded = json.loads(body) if body else None
        except json.JSONDecodeError:
            decoded = body
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
    t.push(200, {"id": str(uuid.uuid4()), "name": "demo", "created": True})
    return t


def test_init_run_posts_run_with_config_and_auth(transport: FakeTransport):
    run = TrainingRun(
        api_key="ba_test",
        name="demo",
        project="proj",
        config={"lr": 1e-4, "batch": 32},
        transport=transport,
    )
    assert run.run_id is not None

    first = transport.calls[0]
    assert first["method"] == "POST"
    assert first["url"].endswith("/v1/runs")
    assert first["headers"]["Authorization"] == "Bearer ba_test"
    assert first["body"]["name"] == "demo"
    assert first["body"]["project"] == "proj"
    assert first["body"]["config"] == {"lr": 1e-4, "batch": 32}
    assert "client_run_id" in first["body"]


def test_log_scalar_posts_metrics(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    run.log({"train/loss": 0.4}, step=10)

    last = transport.calls[-1]
    assert last["method"] == "POST"
    assert last["url"].endswith(f"/v1/runs/{run.run_id}/metrics")
    pts = last["body"]["points"]
    assert len(pts) == 1
    assert pts[0]["key"] == "train/loss"
    assert pts[0]["step"] == 10
    assert pts[0]["value"] == 0.4


def test_log_array_value_passes_through(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    run.log({"grad/norm_hist": [0.1, 0.2, 0.3]}, step=1)

    pts = transport.calls[-1]["body"]["points"]
    assert pts[0]["value"] == [0.1, 0.2, 0.3]


def test_log_auto_step(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    run.log({"a": 1.0})
    run.log({"a": 2.0})
    steps = [c["body"]["points"][0]["step"] for c in transport.calls[1:]]
    assert steps == [0, 1]


def test_log_buffered_until_threshold(transport: FakeTransport):
    run = TrainingRun(
        api_key="ba_test", name="demo", transport=transport, flush_threshold=3
    )
    n_before = len(transport.calls)
    run.log({"a": 1.0}, step=1, commit=False)
    run.log({"a": 2.0}, step=2, commit=False)
    assert len(transport.calls) == n_before  # nothing posted yet
    run.log({"a": 3.0}, step=3, commit=False)  # 3rd hits threshold, auto-flush
    assert len(transport.calls) == n_before + 1
    pts = transport.calls[-1]["body"]["points"]
    assert [p["step"] for p in pts] == [1, 2, 3]


def test_link_dataset_patches_run(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    run.link_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["url"].endswith(f"/v1/runs/{run.run_id}")
    assert last["body"] == {
        "hf_dataset": "HuggingFaceH4/ultrachat_200k",
        "hf_dataset_split": "train_sft",
    }


def test_link_model_patches_run(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    run.link_model("alex/my-finetune", revision="abc123")
    last = transport.calls[-1]
    assert last["body"] == {"hf_model": "alex/my-finetune", "hf_model_revision": "abc123"}


def test_add_artifact_posts(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    run.add_artifact("hparams", data={"warmup_ratio": 0.03})
    last = transport.calls[-1]
    assert last["url"].endswith(f"/v1/runs/{run.run_id}/artifacts")
    assert last["body"] == {
        "name": "hparams",
        "kind": "json",
        "data": {"warmup_ratio": 0.03},
    }


def test_finish_flushes_then_patches(transport: FakeTransport):
    run = TrainingRun(
        api_key="ba_test", name="demo", transport=transport, flush_threshold=100
    )
    run.log({"a": 1.0}, step=1, commit=False)  # buffered
    n_calls_before = len(transport.calls)
    run.finish(status="finished")
    # Expect: one metrics POST (flush), then one PATCH (finish).
    assert len(transport.calls) == n_calls_before + 2
    assert transport.calls[-2]["url"].endswith("/metrics")
    assert transport.calls[-1]["method"] == "PATCH"
    assert transport.calls[-1]["body"]["status"] == "finished"


def test_context_manager_crashes_on_exception(transport: FakeTransport):
    with pytest.raises(RuntimeError):
        with TrainingRun(api_key="ba_test", name="demo", transport=transport) as r:
            raise RuntimeError("boom")
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["body"]["status"] == "crashed"


def test_current_run_raises_without_init(monkeypatch):
    # Sanity: the module-level shim refuses without init_run.
    import claude_monitor._global as g

    monkeypatch.setattr(g, "_current_run", None)
    with pytest.raises(cm.ClaudeMonitorError):
        cm.current_run()


def test_module_level_init_run(transport: FakeTransport, monkeypatch):
    import claude_monitor._global as g

    monkeypatch.setattr(g, "_current_run", None)
    r = cm.init_run(api_key="ba_test", name="x", transport=transport)
    assert cm.current_run() is r
    cm.run_log({"loss": 0.1}, step=0)
    last = transport.calls[-1]
    assert last["url"].endswith(f"/v1/runs/{r.run_id}/metrics")


def test_model_card_returns_markdown(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    transport.push(200, "# fake markdown body")
    md = run.model_card()
    assert "fake markdown" in md
    assert transport.calls[-1]["url"].endswith("/model_card.md")


def test_invalid_status_rejected(transport: FakeTransport):
    run = TrainingRun(api_key="ba_test", name="demo", transport=transport)
    with pytest.raises(cm.ClaudeMonitorError):
        run.finish(status="not-a-status")


def test_api_key_must_start_with_ba(transport: FakeTransport):
    with pytest.raises(cm.ClaudeMonitorError):
        TrainingRun(api_key="not_valid", name="demo", transport=transport)
