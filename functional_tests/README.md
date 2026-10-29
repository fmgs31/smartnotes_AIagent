# Functional (behavioural) tests

Real LLM planner + fake in-memory Drive (no real MCP calls, no disk writes).
They check the agent's *behaviour* (which tools it calls, what ends up in the
docs), not unit logic. Not collected by pytest (`pytest.ini` limits to `tests/`).

All commands run from the repo root, in the project env (`conda activate llm`).
`OPENAI_API_KEY` must be set (or present in the repo-root `.env`).

## Files

| File | What it does |
|---|---|
| `run.py` | Main entry point. Runs all tests (or one) for a single model. |
| `run_tests_concurrent.py` | Runs the suite for many models × repeats **in-process**, up to `MAX_CONCURRENT` at a time. One log per run in `functional_tests_logs/<OUTPUT_LOG_FOLDER>/`; console shows only start/finish lines and a summary. Preferred for comparisons. |
| `run_tests_batch.py` | Same idea, but **sequential**, one `run.py` subprocess per (model, repeat). Slower; useful if in-process runs misbehave. |
| `functional_tests.py` | Test cases (`ALL_TESTS`), `TestAgent`, `make_planner`. |
| `fake_drive.py` | In-memory Drive stub; records every tool call (`get_calls()`). |
| `drive/` | Seed content loaded into the fake Drive. |
| `test_config.py` | Shared static config and helpers (user guidelines / trigger mappings fed to the agent). |
| `tracking.py` | Optional MLflow logging used by all three runners. |

## Running

```
python -m functional_tests.run --list                     # list tests
python -m functional_tests.run                            # all tests, default model
python -m functional_tests.run --model gpt-5.4-nano_CB9   # specific model key
python -m functional_tests.run --test append_terapia -v   # one test, with agent/planner logs
```

Model keys come from `app/prompts/model_prompt_map.json` (key → full model name + prompt file).

For multi-model comparisons edit the config block at the top of
`run_tests_concurrent.py` (or `run_tests_batch.py`):

```python
OUTPUT_LOG_FOLDER = 'v9b'                 # log folder + MLflow `batch` tag
RUNS_PER_MODEL = {'gpt-5.4-nano_OR9': 3, ...}
MAX_CONCURRENT = 2                        # concurrent runner only
USE_MLFLOW = True
MLFLOW_EXPERIMENT = 'planner-behavioural' # MLflow experiment the runs go to
```

then:

```
python -m functional_tests.run_tests_concurrent
python -m functional_tests.run_tests_batch
```

## Tracking results in MLflow

### Setup

Nothing to do. When tracking is on, `tracking.py` defaults
`MLFLOW_TRACKING_URI` to `http://127.0.0.1:5000` and, if no server answers
there, starts one in the background and leaves it running:

```
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Its output goes to `mlflow_server.log` (repo root). To stop it (the server and
its uvicorn child), in PowerShell:

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe' AND CommandLine LIKE '%mlflow server%'" | ForEach-Object { taskkill /PID $_.ProcessId /T /F }
```

A non-local `MLFLOW_TRACKING_URI` is used as-is, never auto-started.

UI: <http://127.0.0.1:5000>, experiment `planner-behavioural` by default
(`MLFLOW_EXPERIMENT` in the runners, `--mlflow-experiment` for `run.py`).
A server is used rather than the SQLite file directly: the concurrent
runner has parallel writers.

The concurrent/batch runners have `USE_MLFLOW = True` by default; set it to
`False` to skip tracking. Run them with the env activated (`conda activate llm`),
not via `conda run`: `conda run` waits for the background server and never returns.

`run.py` only logs when asked:

```
python -m functional_tests.run --model gpt-5.4-nano --mlflow --mlflow-batch-name v10 --repeat 1
python -m functional_tests.run --test append_terapia --mlflow --mlflow-batch-name debug --trace
```

### What is stored

One MLflow run = one model × one repeat.

- **Tags**: `batch` (the `--mlflow-batch-name` / `OUTPUT_LOG_FOLDER` label), `model_key`, `full_name`, `prompt_file`, `prompt_sha`,
  `git_commit`, `git_dirty`, `repeat`, and `failed_tests` (comma-separated
  names of failed/errored tests; filter with `tags.failed_tests LIKE '%name%'`).
  `prompt_sha` is a hash of the prompt *text*, so a prompt edited in place
  counts as a new version even if the file name stays the same.
- **Params**: `model_key`, `full_name`, `prompt_file`, `prompt_sha`,
  `max_plan_steps`, `n_tests` — usable in Compare / parallel-coordinates views.
- **Description**: markdown summary (pass count, time, tool calls, warnings)
  plus a table of failed tests with the first line of each error.
- **Metrics**: `pass_rate`, `passed`, `failed`, `errored`, `total_time_s`,
  `total_tool_calls`, `avg_tool_calls`, `total_steps`, `avg_steps`,
  `total_llm_calls`, `avg_llm_calls`, `warnings` (loop-retry warnings;
  concurrent runner only), and per test `test.<name>` (1 = pass, 0 = fail/error),
  `time.<name>`, `steps.<name>` (planner steps incl. the done step),
  `llm_calls.<name>` (planner + summarizer calls) and `tool_calls.<name>`
  (Drive calls, incl. refiner follow-ups).
- **Artifacts**: `prompt.txt` (and `summarizer_prompt.txt` if any),
  `failures/<test>.txt` tracebacks, `results.json` table (one row per test:
  status, elapsed, steps, LLM calls, error, tool calls), and the run's `.log` file
  (concurrent runner only).
- **Traces** (`--trace` in `run.py`, `USE_TRACE` in the batch and concurrent
  runners): one trace per test with its OpenAI calls nested (messages, tokens,
  latency) and a `passed` feedback, so tests show as rows in the run's
  evaluation panel — open the exact planner output behind a failure. Traces
  are linked by explicit run id, so concurrent runs don't mix.

### Comparing

- In the UI, filter `tags.batch = 'v10'` and group by `model_key` or
  `prompt_sha` to compare mean `pass_rate`.
- Results vary run to run (LLM nondeterminism): compare averages over
  **at least 3 repeats**, never single runs.
- Compare runs on `test.*` metrics to see which tests flip between models/prompts.
- From Python:

  ```python
  import mlflow
  exp = mlflow.get_experiment_by_name("planner-behavioural")
  runs = mlflow.search_runs([exp.experiment_id], filter_string="tags.batch = 'v10'")
  runs.groupby("tags.model_key")["metrics.pass_rate"].agg(["mean", "std", "count"])

  # All per-test rows across runs
  mlflow.MlflowClient().load_table(exp.experiment_id, "results.json", extra_columns=["run_id"])
  ```

### Notes

- `mlflow.db`, `mlartifacts/`, `mlruns/` are git-ignored.
- MLflow prints emoji "View run" lines when a run ends; `tracking.py` makes
  stdout replace unencodable characters so cp1252 consoles don't crash.
- Installing `mlflow` in the `llm` env pulled protobuf 7, which breaks
  `google-cloud-firestore`; it was pinned back to `protobuf==5.29.3`.
  `databricks-sdk` (an mlflow dependency) complains about that version; this
  only matters if you use Databricks.
