"""wandb-style training runs.

Separate from ``Run`` (which is chat-trace shaped) because the data model
and lifecycle are different: a training run has a config blob, a
time-series metric stream, links to HF dataset/model, and lives for hours
or days. Conflating it with the chat-trace ``Run`` would force every
``log()`` call to multiplex two unrelated wire formats.

Example::

    import claude_monitor as cm
    run = cm.init_run(project="demo", name="qwen-sft",
                      config={"lr": 1e-4, "batch": 32})
    for step in range(1000):
        run.log({"train/loss": loss}, step=step)
    run.link_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    run.link_model("alex/my-finetune")
    run.finish(status="finished")
    print(run.push_model_card()["commit_url"])
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import uuid
from typing import Any, Iterable, Mapping, Optional

from .client import (
    ClaudeMonitorError,
    Transport,
    _do_request,
    _machine_id_default,
    _resolve_api_base,
    _urllib_transport,
    _utcnow_iso,
    _validate_api_key,
)

ALLOWED_STATUSES = ("running", "finished", "failed", "crashed", "killed")
ALLOWED_ARTIFACT_KINDS = ("json", "text", "params")


class TrainingRun:
    """A single training run: config + time-series of metrics + linked refs.

    Resumable via ``client_run_id`` — re-instantiating with the same id from
    the same machine returns the existing run.
    """

    def __init__(
        self,
        *,
        project: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Mapping[str, Any]] = None,
        client_run_id: Optional[str] = None,
        hf_dataset: Optional[str] = None,
        hf_dataset_split: Optional[str] = None,
        hf_model: Optional[str] = None,
        hf_model_revision: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        machine_id: Optional[str] = None,
        hostname: Optional[str] = None,
        agent_version: Optional[str] = None,
        transport: Optional[Transport] = None,
        auto_create: bool = True,
        flush_threshold: int = 500,
    ) -> None:
        self._api_key = _validate_api_key(api_key)
        self._api_base = _resolve_api_base(api_base)
        self._transport = transport or _urllib_transport
        self.machine_id = machine_id or _machine_id_default()
        self.hostname = hostname or platform.node() or None
        self.agent_version = agent_version

        self.project = project
        self.name = name or f"run-{uuid.uuid4().hex[:8]}"
        self.client_run_id = client_run_id or f"py-{uuid.uuid4()}"
        self.config: dict[str, Any] = dict(config or {})

        self.hf_dataset = hf_dataset
        self.hf_dataset_split = hf_dataset_split
        self.hf_model = hf_model
        self.hf_model_revision = hf_model_revision

        self.run_id: Optional[str] = None
        self._closed = False
        self._auto_step = 0
        self._buf: list[dict[str, Any]] = []
        self.flush_threshold = max(1, flush_threshold)

        if auto_create:
            self._create_run()

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

    def _create_run(self) -> None:
        body: dict[str, Any] = {
            "name": self.name,
            "client_run_id": self.client_run_id,
            "started_at": _utcnow_iso(),
        }
        if self.project is not None:
            body["project"] = self.project
        if self.config:
            body["config"] = self.config
        if self.hf_dataset is not None:
            body["hf_dataset"] = self.hf_dataset
        if self.hf_dataset_split is not None:
            body["hf_dataset_split"] = self.hf_dataset_split
        if self.hf_model is not None:
            body["hf_model"] = self.hf_model
        if self.hf_model_revision is not None:
            body["hf_model_revision"] = self.hf_model_revision
        resp = self._request("POST", "/v1/runs", body)
        self.run_id = resp.get("id")
        if not self.run_id:
            raise ClaudeMonitorError(
                f"server did not return a run id (response: {resp!r})"
            )

    # ----- metric ingest -------------------------------------------------- #

    def log(
        self,
        values: Mapping[str, Any],
        *,
        step: Optional[int] = None,
        wall_time: Optional[str] = None,
        commit: bool = True,
    ) -> None:
        """Record one (or more) metric points at ``step``.

        ``values`` is a ``{key: value}`` dict. ``value`` may be:
          * an ``int``/``float`` — stored in the fast scalar column;
          * a ``list`` / ``dict`` — stored as JSON (histograms, distributions);
          * a ``bool`` — coerced to 0/1.

        ``step`` auto-increments when omitted. Pass ``commit=False`` to buffer
        many ``log()`` calls into a single POST (auto-flushed at
        ``flush_threshold``).
        """
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        if not values:
            return
        s = step if step is not None else self._auto_step
        if step is None:
            self._auto_step += 1
        else:
            self._auto_step = max(self._auto_step, step + 1)
        wt = wall_time or _utcnow_iso()
        for key, value in values.items():
            self._buf.append(
                {"key": key, "step": int(s), "value": value, "wall_time": wt}
            )
        if commit or len(self._buf) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        if not self.run_id or not self._buf:
            return
        points = self._buf
        self._buf = []
        self._request(
            "POST",
            f"/v1/runs/{self.run_id}/metrics",
            {"points": points},
        )

    # ----- linking + artifacts ------------------------------------------- #

    def link_dataset(self, hf_slug: str, *, split: Optional[str] = None) -> None:
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        self.hf_dataset = hf_slug
        if split is not None:
            self.hf_dataset_split = split
        patch: dict[str, Any] = {"hf_dataset": hf_slug}
        if split is not None:
            patch["hf_dataset_split"] = split
        self._request("PATCH", f"/v1/runs/{self.run_id}", patch)

    def link_model(self, hf_repo: str, *, revision: Optional[str] = None) -> None:
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        self.hf_model = hf_repo
        if revision is not None:
            self.hf_model_revision = revision
        patch: dict[str, Any] = {"hf_model": hf_repo}
        if revision is not None:
            patch["hf_model_revision"] = revision
        self._request("PATCH", f"/v1/runs/{self.run_id}", patch)

    def add_artifact(
        self,
        name: str,
        *,
        data: Any,
        kind: str = "json",
    ) -> None:
        if kind not in ALLOWED_ARTIFACT_KINDS:
            raise ClaudeMonitorError(
                f"kind must be one of {ALLOWED_ARTIFACT_KINDS}, got {kind!r}"
            )
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        self._request(
            "POST",
            f"/v1/runs/{self.run_id}/artifacts",
            {"name": name, "kind": kind, "data": data},
        )

    # ----- model card ----------------------------------------------------- #

    def model_card(self) -> str:
        """Fetch the auto-generated model card markdown."""
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        path = f"/v1/runs/{self.run_id}/model_card.md"
        # Reuse the JSON request machinery — the markdown body comes back
        # as `{"raw": "..."}` from ``_do_request`` since it isn't JSON.
        resp = self._request("GET", path, None)
        if "raw" in resp:
            return resp["raw"]
        # Server shouldn't return JSON here, but be defensive.
        import json as _json
        return _json.dumps(resp)

    def push_model_card(
        self,
        *,
        commit_message: Optional[str] = None,
    ) -> dict[str, Any]:
        """Push the auto-generated README.md to the linked HF model repo.

        Requires the user to have saved a write-scope ``hf_token`` in their
        claude-monitor profile, and ``link_model`` to have been called.
        Returns ``{commit_url, commit_oid, repo}``.
        """
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        body: dict[str, Any] = {}
        if commit_message is not None:
            body["commit_message"] = commit_message
        return self._request("POST", f"/v1/runs/{self.run_id}/push_model_card", body)

    # ----- finish + context manager -------------------------------------- #

    def finish(self, *, status: str = "finished") -> None:
        if self._closed:
            return
        if status not in ALLOWED_STATUSES:
            raise ClaudeMonitorError(
                f"status must be one of {ALLOWED_STATUSES}, got {status!r}"
            )
        if not self.run_id:
            raise ClaudeMonitorError("run is not active")
        try:
            self.flush()
        finally:
            self._request(
                "PATCH",
                f"/v1/runs/{self.run_id}",
                {"status": status, "ended_at": _utcnow_iso()},
            )
            self._closed = True

    def __enter__(self) -> "TrainingRun":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._closed:
            return
        try:
            if exc is not None:
                self.finish(status="crashed")
            else:
                self.finish()
        except Exception:  # noqa: BLE001 — don't mask the original
            pass
