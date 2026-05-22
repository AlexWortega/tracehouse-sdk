"""wandb-compat tests for claude_monitor.wandb against an in-memory transport."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

import pytest

from claude_monitor import wandb
from claude_monitor.client import HttpResponse


@dataclass
class FakeTransport:
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
def transport(monkeypatch) -> FakeTransport:
    t = FakeTransport()
    t.push(200, {"id": str(uuid.uuid4()), "name": "demo", "created": True})

    # trackio.init() builds a TrainingRun internally. Patch the default
    # transport so we don't hit network, and disable env/system probes that
    # would otherwise spawn a background thread + post an artifact.
    from claude_monitor import client as cm_client
    from claude_monitor import training as cm_training

    monkeypatch.setattr(cm_client, "_urllib_transport", t)

    orig_init = cm_training.TrainingRun.__init__

    def patched_init(self, **kw):
        kw.setdefault("transport", t)
        kw.setdefault("capture_env", False)
        kw.setdefault("system_metrics", False)
        orig_init(self, **kw)

    monkeypatch.setattr(cm_training.TrainingRun, "__init__", patched_init)
    monkeypatch.setattr(wandb, "_current", None)
    return t


def test_init_returns_wandb_shaped_run(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    run = wandb.init(project="demo", name="exp", config={"lr": 1e-4})

    assert isinstance(run, wandb.Run)
    assert run.name == "exp"
    assert run.project == "demo"
    assert run.id  # uuid from FakeTransport response
    assert wandb.run is run

    first = transport.calls[0]
    assert first["url"].endswith("/v1/runs")
    assert first["body"]["name"] == "exp"
    assert first["body"]["config"] == {"lr": 1e-4}


def test_log_dict_scalars(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x")
    wandb.log({"train/loss": 0.4, "eval/acc": 0.91}, step=10)

    last = transport.calls[-1]
    assert last["url"].endswith("/metrics")
    keys = {p["key"] for p in last["body"]["points"]}
    assert keys == {"train/loss", "eval/acc"}
    assert all(p["step"] == 10 for p in last["body"]["points"])


def test_log_histogram_is_unwrapped(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x")
    h = wandb.Histogram([0.1, 0.2, 0.4, 0.7, 0.9], num_bins=4)
    wandb.log({"grad/norm": h})

    last = transport.calls[-1]
    val = last["body"]["points"][0]["value"]
    assert isinstance(val, dict)
    assert val["_type"] == "histogram"
    assert "counts" in val and "bins" in val


def test_config_update_patches_run(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x")

    n_before = len(transport.calls)
    wandb.config.update({"warmup_ratio": 0.03, "weight_decay": 0.0})
    assert len(transport.calls) == n_before + 1
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["body"]["config_patch"] == {
        "warmup_ratio": 0.03,
        "weight_decay": 0.0,
    }


def test_config_attribute_set(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x", config={"lr": 1e-4})

    wandb.config.batch = 32
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["body"]["config_patch"] == {"batch": 32}
    assert wandb.config.lr == 1e-4
    assert wandb.config.batch == 32


def test_summary_write_through_logs_metric(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x")

    wandb.summary["best_loss"] = 0.05
    last = transport.calls[-1]
    assert last["url"].endswith("/metrics")
    assert last["body"]["points"][0]["key"] == "best_loss"
    assert last["body"]["points"][0]["value"] == 0.05


def test_finish_marks_finished(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x")
    wandb.finish()
    last = transport.calls[-1]
    assert last["method"] == "PATCH"
    assert last["body"]["status"] == "finished"
    assert wandb.run is None


def test_finish_with_exit_code_marks_failed(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    wandb.init(project="demo", name="x")
    wandb.finish(exit_code=1)
    last = transport.calls[-1]
    assert last["body"]["status"] == "failed"


def test_disabled_mode_is_noop(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    n_before = len(transport.calls)
    r = wandb.init(project="demo", mode="disabled")
    wandb.log({"loss": 0.5}, step=1)
    wandb.summary["x"] = 1
    wandb.finish()
    # No HTTP traffic, no exceptions.
    assert len(transport.calls) == n_before
    assert r.id == ""


def test_unknown_init_kwargs_dont_crash(transport: FakeTransport, monkeypatch):
    monkeypatch.setenv("CLAUDE_MONITOR_API_KEY", "ba_test")
    run = wandb.init(
        project="demo",
        name="x",
        entity="my-team",
        tags=["a", "b"],
        notes="hello",
        job_type="train",
        group="sweep-1",
        save_code=True,
        anonymous="never",
        resume="allow",
    )
    assert isinstance(run, wandb.Run)
    body = transport.calls[0]["body"]
    cfg = body["config"]
    assert cfg["wandb_entity"] == "my-team"
    assert cfg["wandb_tags"] == ["a", "b"]
    assert cfg["wandb_notes"] == "hello"
    assert cfg["wandb_job_type"] == "train"
    assert cfg["wandb_group"] == "sweep-1"


def test_save_watch_login_are_noops():
    # No active run required.
    wandb.save("model.pt")
    wandb.watch(object())
    assert wandb.login() is True
    wandb.alert(title="hi", text="bye")
