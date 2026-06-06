# tracehouse (Python)

Push traces and spans to [tracehouse](https://github.com/AlexWortega/claude-monitor)
from any Python script — wandb-style, zero install dependencies.

```bash
pip install tracehouse
```

## Quickstart

```python
import tracehouse as cm

cm.init(
    api_key="ba_…",          # or set TRACEHOUSE_API_KEY
    project="my-bot",
    session_id="run-001",    # idempotent: same id resumes the same trace
    task_name="demo task",
    model="claude-opus-4-7",
)

cm.log_user("hello")
cm.log_assistant("hi there")
cm.log_tool_use("Read", {"file_path": "x.py"})
cm.log_tool_result("file contents")

cm.finish(outcome="good", metadata={"k": "v"})
```

## Class-based / `with`

```python
import tracehouse as cm

with cm.Run(project="my-bot", session_id="run-002") as run:
    run.log_user("how do I install jq?")
    run.log_assistant("brew install jq")
    # implicit run.finish() on exit; on exception → outcome="bad" + error metadata
```

## API

* `cm.init(**kwargs) -> Run` — create the module-level run (wandb style).
* `cm.Run(**kwargs)` — explicit run; identical kwargs.
* `cm.log_user(text)`, `cm.log_assistant(text)`, `cm.log_thinking(text)`,
  `cm.log_tool_use(tool, input)`, `cm.log_tool_result(text, parent_span_id=…)`,
  `cm.log_attachment(name, attributes)` — convenience helpers.
* `cm.log(kind=…, name=…, text=…, attributes=…, parent_span_id=…)` — generic.
* `cm.finish(outcome="good"|"bad"|"neutral", metadata={…}, task_name=…, model=…)`.

### Configuration

| Argument         | Env var                       | Default |
|------------------|-------------------------------|---------|
| `api_key`        | `TRACEHOUSE_API_KEY`      | required |
| `api_base`       | `TRACEHOUSE_API_BASE`     | hosted Railway URL |
| `session_id`     | —                             | random `py-<uuid>` |
| `project`        | —                             | `None` |
| `scaffold`       | —                             | `"python-sdk"` |
| `machine_id`     | —                             | derived from hostname |

### Span kinds

`user_msg | assistant_msg | tool_use | tool_result | thinking | attachment`

Common attributes the UI surfaces directly: `text` (string), `result_text`
(string), `tool_input` (object). Anything else lands in the raw JSON view.

## RL: runs + rollout traces

Log a training run's **metrics** and its per-step **rollout conversations** together.
`run.rollout(step=…)` opens a chat trace already linked to the run, so every rollout
shows up under the run's **Rollouts** tab (step → trace).

```python
import tracehouse as cm

run = cm.init_run(project="rl", name="ppo-v1", config={"lr": 1e-5})

for step in range(1000):
    # One chat trace per rollout, tied to this run + step.
    with run.rollout(step=step) as t:
        t.log_user(state)
        t.log_assistant(action)
        t.log_tool_result(f"reward={reward}")
    run.log({"reward": reward, "kl": kl}, step=step)   # metrics on the run

run.finish()
```

`rollout()` returns a normal `Run` (any `log_*` helper works) and inherits the run's
auth — so an anonymous run produces anonymous rollouts under the **same** identity, and
a single claim link covers the run and all its traces. `step` defaults to the run's
auto-incrementing counter; pass `name=` / `session_id=` to override the trace labels.
