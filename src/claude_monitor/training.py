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
import logging
import os
import platform
import uuid
from typing import Any, Iterable, Mapping, Optional

_log = logging.getLogger(__name__)

from .client import (
    AuthContext,
    ClaudeMonitorError,
    Run,
    Transport,
    _do_request,
    _machine_id_default,
    _resolve_api_base,
    _resolve_auth,
    _resolve_web_url,
    _urllib_transport,
    _utcnow_iso,
    _warn_anonymous,
)
from .system import SystemMonitor, capture_environment

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
        capture_env: bool = True,
        system_metrics: bool = True,
        system_metrics_interval: float = 15.0,
    ) -> None:
        self._api_base = _resolve_api_base(api_base)
        self._transport = transport or _urllib_transport
        auth = _resolve_auth(
            api_key, self._api_base, self._transport, allow_cache=transport is None
        )
        self._api_key = auth.token
        self._is_anonymous = auth.is_anonymous
        self._web_url = _resolve_web_url(None, auth.web_url)
        self._claim_token = auth.claim_token
        self._read_token = auth.read_token
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
        self._capture_env = capture_env
        self._system_metrics_enabled = system_metrics
        self._system_metrics_interval = system_metrics_interval
        self._monitor: Optional[SystemMonitor] = None

        if auto_create:
            self._create_run()
            if self._is_anonymous and self.run_id:
                _warn_anonymous(
                    f"{self._web_url}/r/{self.run_id}?t={self._read_token}",
                    f"{self._web_url}/claim?token={self._claim_token}",
                )
            if self._capture_env:
                self._attach_environment()
            if self._system_metrics_enabled:
                self._start_system_monitor()

    def _attach_environment(self) -> None:
        try:
            env = capture_environment()
            self._request(
                "POST",
                f"/v1/runs/{self.run_id}/artifacts",
                {"name": "environment", "kind": "json", "data": env},
            )
            n_gpus = len(env.get("gpus") or [])
            _log.info(
                "environment captured: python=%s os=%s gpus=%d",
                env.get("python", {}).get("version"),
                env.get("os", {}).get("system"),
                n_gpus,
            )
        except Exception as e:  # noqa: BLE001
            _log.warning("failed to capture environment: %s", e)

    def _start_system_monitor(self) -> None:
        # The monitor calls back into self.log() with system/* keys. We use
        # commit=True so each sample lands immediately — system metrics are
        # low-volume (one sample / 15s) so the per-sample HTTP cost is fine.
        def _emit(values):  # type: ignore[no-untyped-def]
            try:
                # No step provided → use the SDK's auto-incrementing counter.
                # System samples and user log() calls share the step space,
                # which is intentional — they're plotted on the same x-axis.
                self.log(values)
            except Exception as e:  # noqa: BLE001
                _log.debug("system log() failed: %s", e)

        self._monitor = SystemMonitor(
            _emit, interval=self._system_metrics_interval
        )
        self._monitor.start()

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
        _log.info(
            "run ready: id=%s name=%s%s%s",
            self.run_id,
            self.name,
            f" project={self.project}" if self.project else "",
            "" if resp.get("created", True) else " (resumed)",
        )

    # ----- rollouts ------------------------------------------------------- #

    def rollout(
        self,
        *,
        step: Optional[int] = None,
        name: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Open a chat trace tied to this run as one rollout (RL pattern).

        Returns a :class:`Run` already tagged with this run's id and the given
        step, so it shows up under the run's *Rollouts* tab. Use it as a
        context manager::

            for step in range(n):
                with run.rollout(step=step) as t:
                    t.log_user(state)
                    t.log_assistant(action)
                    t.log_tool_result(f"reward={reward}")
                run.log({"reward": reward}, step=step)

        Auth/transport are inherited from the run, so an anonymous run produces
        anonymous rollouts under the same identity (one claim link covers both).
        """
        if not self.run_id:
            raise ClaudeMonitorError("run was not created — no run_id")
        if step is None:
            step = self._auto_step
        return Run(
            api_base=self._api_base,
            transport=self._transport,
            _auth=AuthContext(
                token=self._api_key,
                is_anonymous=self._is_anonymous,
                web_url=self._web_url,
                claim_token=self._claim_token,
                read_token=self._read_token,
            ),
            project=self.project,
            session_id=session_id or f"{self.name}-step{step}",
            task_name=name or f"rollout step {step}",
            machine_id=self.machine_id,
            hostname=self.hostname,
            agent_version=self.agent_version,
            run_id=self.run_id,
            rollout_step=step,
            **kwargs,
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
        resp = self._request(
            "POST",
            f"/v1/runs/{self.run_id}/metrics",
            {"points": points},
        )
        ingested = resp.get("ingested", len(points))
        dropped = resp.get("dropped", 0)
        if dropped:
            _log.warning(
                "dropped %d metric point(s) (NaN/Inf or empty key) — ingested=%d",
                dropped,
                ingested,
            )
        else:
            _log.debug(
                "flushed %d metric point(s) (ingested=%d)", len(points), ingested
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
        _log.info(
            "linked dataset: %s%s", hf_slug, f" (split={split})" if split else ""
        )

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
        _log.info(
            "linked model: %s%s", hf_repo, f"@{revision}" if revision else ""
        )

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
        _log.info("artifact stored: name=%s kind=%s", name, kind)

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
        resp = self._request("POST", f"/v1/runs/{self.run_id}/push_model_card", body)
        commit_url = resp.get("commit_url")
        _log.info(
            "model card pushed: repo=%s commit=%s",
            resp.get("repo"),
            commit_url or "(no commit_url returned)",
        )
        return resp

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
        if self._monitor is not None:
            self._monitor.stop()
            self._monitor = None
        try:
            self.flush()
        finally:
            self._request(
                "PATCH",
                f"/v1/runs/{self.run_id}",
                {"status": status, "ended_at": _utcnow_iso()},
            )
            self._closed = True
            _log.info("run finished: id=%s status=%s", self.run_id, status)

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
