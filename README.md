# claude-monitor (Python)

Push traces and spans to [claude-monitor](https://github.com/AlexWortega/claude-monitor)
from any Python script — wandb-style, zero install dependencies.

```bash
pip install claude-monitor
```

## Quickstart

```python
import claude_monitor as cm

cm.init(
    api_key="ba_…",          # or set CLAUDE_MONITOR_API_KEY
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
import claude_monitor as cm

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
| `api_key`        | `CLAUDE_MONITOR_API_KEY`      | required |
| `api_base`       | `CLAUDE_MONITOR_API_BASE`     | hosted Railway URL |
| `session_id`     | —                             | random `py-<uuid>` |
| `project`        | —                             | `None` |
| `scaffold`       | —                             | `"python-sdk"` |
| `machine_id`     | —                             | derived from hostname |

### Span kinds

`user_msg | assistant_msg | tool_use | tool_result | thinking | attachment`

Common attributes the UI surfaces directly: `text` (string), `result_text`
(string), `tool_input` (object). Anything else lands in the raw JSON view.
