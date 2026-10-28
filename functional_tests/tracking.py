"""
Optional MLflow tracking for functional test runs.

One MLflow run = one model × one repeat. Uses MlflowClient with explicit run
ids (not the fluent active-run API) so concurrent asyncio runs in the same
thread don't write into each other's run.

Tracking server is taken from MLFLOW_TRACKING_URI, e.g.:
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
    set MLFLOW_TRACKING_URI=http://localhost:5000
"""
import hashlib
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "planner-behavioural"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


class Tracker:
    """No-op unless enabled; every method is safe to call either way."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.run_id: str | None = None
        self._rows: list[dict] = []
        self._t0 = 0.0
        if enabled:
            # MLflow prints emoji "View run" lines on set_terminated; don't crash on cp1252 consoles.
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(errors="replace")
            from mlflow import MlflowClient
            self._client = MlflowClient()

    def start(self, model_key: str, resolved, mlflow_batch_name: str, repeat: int) -> None:
        if not self.enabled:
            return
        exp = self._client.get_experiment_by_name(EXPERIMENT)
        exp_id = exp.experiment_id if exp else self._client.create_experiment(EXPERIMENT)
        tags = {
            # Tag key stays "batch" so filters on runs logged before the rename keep working.
            "batch": mlflow_batch_name,
            "model_key": model_key,
            "full_name": resolved.full_name,
            "prompt_file": resolved.prompt_file,
            "prompt_sha": hashlib.sha256(resolved.prompt.encode("utf-8")).hexdigest()[:12],
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
            "repeat": str(repeat),
        }
        run = self._client.create_run(exp_id, run_name=f"{mlflow_batch_name}/{model_key}/r{repeat}", tags=tags)
        self.run_id = run.info.run_id
        self._client.log_text(self.run_id, resolved.prompt, "prompt.txt")
        if resolved.summarizer_prompt:
            self._client.log_text(self.run_id, resolved.summarizer_prompt, "summarizer_prompt.txt")
        self._t0 = time.perf_counter()

    def record(self, name: str, status: str, elapsed: float, calls: list[dict],
               error: str = "", tb: str = "") -> None:
        if not self.run_id:
            return
        self._client.log_metric(self.run_id, f"test.{name}", 1.0 if status == "passed" else 0.0)
        self._client.log_metric(self.run_id, f"time.{name}", elapsed)
        self._rows.append({
            "test": name,
            "status": status,
            "elapsed_s": round(elapsed, 2),
            "n_tool_calls": len(calls),
            "tool_calls": "\n".join(f"[{c['step']}] {c['tool']} {c['args']}" for c in calls),
            "error": error,
        })
        if tb:
            self._client.log_text(self.run_id, tb, f"failures/{name}.txt")

    def finish(self, passed: int, failed: int, errored: int, log_file: Path | None = None,
               crashed: bool = False) -> None:
        if not self.run_id:
            return
        total = passed + failed + errored
        for key, value in {
            "passed": passed, "failed": failed, "errored": errored,
            "pass_rate": passed / total if total else 0.0,
            "total_time_s": time.perf_counter() - self._t0,
        }.items():
            self._client.log_metric(self.run_id, key, value)
        if self._rows:
            import pandas as pd
            self._client.log_table(self.run_id, data=pd.DataFrame(self._rows), artifact_file="results.json")
        if log_file is not None and log_file.exists():
            self._client.log_artifact(self.run_id, str(log_file))
        self._client.set_terminated(self.run_id, "FAILED" if crashed else "FINISHED")
