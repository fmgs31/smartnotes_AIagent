"""
Optional MLflow tracking for functional test runs.

One MLflow run = one model × one repeat. Uses MlflowClient with explicit run
ids (not the fluent active-run API) so concurrent asyncio runs in the same
thread don't write into each other's run.

Tracking server is taken from MLFLOW_TRACKING_URI (default
http://127.0.0.1:5000). For a local URI with nothing listening, the server is
started automatically and left running so the UI stays available:
    mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000
"""
import hashlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT = "planner-behavioural"
DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"  # not localhost: IPv6-first lookup adds ~2s per connect on Windows
SERVER_LOG = ROOT / "mlflow_server.log"
_SERVER_START_TIMEOUT_S = 90  # first start runs DB migrations


def _server_up(uri: str) -> bool:
    try:
        with urllib.request.urlopen(f"{uri.rstrip('/')}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def ensure_mlflow_server() -> str:
    """
    Make sure MLFLOW_TRACKING_URI points at a live server; return the URI.
    Defaults the URI to 127.0.0.1:5000 and, for a local URI, starts
    `mlflow server` detached (output in mlflow_server.log) if it isn't up.
    A remote URI is used as-is.
    """
    uri = os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    if _server_up(uri):
        return uri
    parsed = urlparse(uri)
    if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1"):
        return uri

    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--host", parsed.hostname, "--port", str(parsed.port or 5000),
    ]
    if os.name == "nt":
        detach = {"creationflags": subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        detach = {"start_new_session": True}
    print(f"MLflow server not running at {uri}; starting it (log: {SERVER_LOG.name})", flush=True)
    log = open(SERVER_LOG, "a", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, **detach)
    log.close()  # the child keeps its own handle

    deadline = time.monotonic() + _SERVER_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _server_up(uri):
            return uri
        if proc.poll() is not None:
            raise RuntimeError(f"mlflow server exited with code {proc.returncode}; see {SERVER_LOG}")
        time.sleep(1)
    raise RuntimeError(f"mlflow server not healthy after {_SERVER_START_TIMEOUT_S}s; see {SERVER_LOG}")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        return ""


class Tracker:
    """No-op unless enabled; every method is safe to call either way."""

    def __init__(self, enabled: bool, experiment: str = EXPERIMENT):
        self.enabled = enabled
        self.experiment = experiment
        self.run_id: str | None = None
        self._rows: list[dict] = []
        self._test_metrics: list[tuple[str, float]] = []
        self._t0 = 0.0
        if enabled:
            # MLflow prints emoji "View run" lines on set_terminated; don't crash on cp1252 consoles.
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(errors="replace")
            ensure_mlflow_server()  # before MlflowClient(), which reads MLFLOW_TRACKING_URI
            from mlflow import MlflowClient
            self._client = MlflowClient()

    def start(self, model_key: str, resolved, mlflow_batch_name: str, repeat: int,
              max_plan_steps: int, n_tests: int) -> None:
        if not self.enabled:
            return
        from mlflow.entities import Param
        exp = self._client.get_experiment_by_name(self.experiment)
        exp_id = exp.experiment_id if exp else self._client.create_experiment(self.experiment)
        config = {
            "model_key": model_key,
            "full_name": resolved.full_name,
            "prompt_file": resolved.prompt_file,
            "prompt_sha": hashlib.sha256(resolved.prompt.encode("utf-8")).hexdigest()[:12],
        }
        tags = {
            # Tag key stays "batch" so filters on runs logged before the rename keep working.
            "batch": mlflow_batch_name,
            **config,  # also kept as tags so existing tag filters keep working
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "git_dirty": str(bool(_git("status", "--porcelain", "--untracked-files=no"))),
            "repeat": str(repeat),
        }
        run = self._client.create_run(exp_id, run_name=f"{mlflow_batch_name}/{model_key}/r{repeat}", tags=tags)
        self.run_id = run.info.run_id
        params = {**config, "max_plan_steps": max_plan_steps, "n_tests": n_tests}
        self._client.log_batch(self.run_id, params=[Param(k, str(v)) for k, v in params.items()])
        self._client.log_text(self.run_id, resolved.prompt, "prompt.txt")
        if resolved.summarizer_prompt:
            self._client.log_text(self.run_id, resolved.summarizer_prompt, "summarizer_prompt.txt")
        self._t0 = time.perf_counter()

    def record(self, name: str, status: str, elapsed: float, calls: list[dict],
               error: str = "", tb: str = "") -> None:
        if not self.run_id:
            return
        # Logged in finish(), after the summary metrics: the UI lists metrics in logging order.
        self._test_metrics += [(f"test.{name}", 1.0 if status == "passed" else 0.0),
                               (f"time.{name}", elapsed)]
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

    def _description(self, passed: int, total: int, total_time: float, tool_calls: int,
                     warnings: int | None, crashed: bool) -> str:
        """Markdown shown as the run's Description in the MLflow UI."""
        parts = [f"**{passed}/{total} passed**", f"{total_time:.0f}s", f"{tool_calls} tool calls"]
        if warnings is not None:
            parts.append(f"{warnings} loop-retry warnings")
        lines = [" · ".join(parts)]
        if crashed:
            lines.append("\n**Run crashed** — see the log artifact.")
        bad = [r for r in self._rows if r["status"] != "passed"]
        if bad:
            lines += ["", "| Test | Status | Error |", "|---|---|---|"]
            for r in bad:
                err = (r["error"].strip().splitlines() or [""])[0][:200].replace("|", "\\|")
                lines.append(f"| {r['test']} | {r['status']} | {err} |")
        return "\n".join(lines)

    def finish(self, passed: int, failed: int, errored: int, log_file: Path | None = None,
               crashed: bool = False, warnings: int | None = None) -> None:
        if not self.run_id:
            return
        total = passed + failed + errored
        total_time = time.perf_counter() - self._t0
        tool_calls = sum(r["n_tool_calls"] for r in self._rows)
        metrics = {
            "passed": passed, "failed": failed, "errored": errored,
            "pass_rate": passed / total if total else 0.0,
            "total_time_s": total_time,
            "total_tool_calls": tool_calls,
            "avg_tool_calls": tool_calls / len(self._rows) if self._rows else 0.0,
        }
        if warnings is not None:
            metrics["warnings"] = warnings
        for key, value in [*metrics.items(), *self._test_metrics]:
            self._client.log_metric(self.run_id, key, value)
        failed_tests = [r["test"] for r in self._rows if r["status"] != "passed"]
        self._client.set_tag(self.run_id, "failed_tests", ",".join(failed_tests))
        self._client.set_tag(self.run_id, "mlflow.note.content",
                             self._description(passed, total, total_time, tool_calls, warnings, crashed))
        if self._rows:
            import pandas as pd
            self._client.log_table(self.run_id, data=pd.DataFrame(self._rows), artifact_file="results.json")
        if log_file is not None and log_file.exists():
            self._client.log_artifact(self.run_id, str(log_file))
        self._client.set_terminated(self.run_id, "FAILED" if crashed else "FINISHED")
