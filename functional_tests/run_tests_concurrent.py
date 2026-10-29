"""
Concurrent in-process functional-test runner.

Usage: python -m functional_tests.run_tests_concurrent

Runs the same tests as `python -m functional_tests.run --model <M>` but:
  * in-process (no subprocess / no Unicode piping issues on Windows)
  * many (model, run) pairs run concurrently with a configurable cap
  * each run's full output goes to its own log file
  * the real console shows only start/finish lines + a final summary

Configure MODELS / OUTPUT_LOG_FOLDER / RUNS_PER_MODEL / MAX_CONCURRENT below.
"""
import asyncio
import contextvars
import logging
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
os.environ.setdefault("APP_API_KEY", "functional-test-key")
os.environ.setdefault("ALLOWED_EMAILS", "test@example.com")

from functional_tests.fake_drive import FakeDrive  # noqa: E402
from functional_tests.functional_tests import ALL_TESTS, TestAgent, make_planner  # noqa: E402
from functional_tests.tracking import Tracker  # noqa: E402

# ── Configuration ────────────────────────────────────────────────────────────
OUTPUT_LOG_FOLDER = 'v9b'
RUNS_PER_MODEL = {
    'gpt-5.4-nano_OR9': 3,
    'gpt-5.4-nano_CB9': 3,
    'gpt-5-nano_CB9':   2,
    'gpt-5-nano_OR9':   2,
}
MODELS = list(RUNS_PER_MODEL.keys())
MAX_CONCURRENT = 2
USE_MLFLOW = True  # needs MLFLOW_TRACKING_URI pointing at a running `mlflow server`

WARNING_MARKER = "WARNING loop retries:"

# ── Per-task stdout/stderr dispatch ──────────────────────────────────────────
# Every run writes to its own log file. Tests use print() which goes to
# sys.stdout. We replace sys.stdout / sys.stderr with a dispatcher that looks
# up a contextvars-bound log file, so concurrent asyncio tasks don't interleave.
_current_log: contextvars.ContextVar = contextvars.ContextVar('current_log', default=None)
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr


class _Dispatch:
    def __init__(self, fallback):
        self._fallback = fallback

    def write(self, s):
        target = _current_log.get()
        if target is not None:
            target.write(s)
            return len(s)
        try:
            return self._fallback.write(s)
        except UnicodeEncodeError:
            enc = getattr(self._fallback, 'encoding', None) or 'ascii'
            return self._fallback.write(s.encode(enc, errors='replace').decode(enc, errors='replace'))

    def flush(self):
        target = _current_log.get()
        if target is not None:
            target.flush()
        else:
            self._fallback.flush()

    def __getattr__(self, name):
        return getattr(self._fallback, name)


sys.stdout = _Dispatch(_orig_stdout)
sys.stderr = _Dispatch(_orig_stderr)

# Route agent/planner logs through the dispatcher too.
logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
for _name in ("agent", "planner"):
    _log = logging.getLogger(_name)
    _log.propagate = False
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        _h = logging.StreamHandler()  # uses sys.stderr (dispatcher) at emit time
        _h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        _log.addHandler(_h)


def _console(msg: str) -> None:
    """Write to the real console, bypassing the dispatcher. ASCII only."""
    _orig_stdout.write(msg + '\n')
    _orig_stdout.flush()


# ── Per-run execution ────────────────────────────────────────────────────────

def _print_call_log(drive: FakeDrive) -> None:
    calls = drive.get_calls()
    if not calls:
        print("     Tool calls: (none)")
        return
    print(f"     Tool calls ({len(calls)}):")
    for c in calls:
        print(f"       [{c['step']}] {c['tool']}  {c['args']}")


async def _execute_tests(model: str, tracker: Tracker, run_num: int) -> None:
    """Runs all tests for one model. All print() goes to the current log file."""
    planner = make_planner()
    resolved = planner._model_registry.resolve(model)
    prompt_first_line = resolved.prompt.splitlines()[0]
    print(f"model: {model}  →  {resolved.full_name}")
    print(f"prompt: {prompt_first_line}")
    tracker.start(model, resolved, OUTPUT_LOG_FOLDER, run_num)

    passed = failed = errored = 0
    for tc in ALL_TESTS:
        print(f"\n{'─' * 64}")
        print(f"TEST  {tc.name}")
        print(f"      {tc.description}")

        drive = FakeDrive()
        agent = TestAgent(fake_drive=drive, planner=planner, max_plan_steps=10, planner_model=model)

        t0 = time.perf_counter()
        try:
            await tc.fn(agent, drive)
            elapsed = time.perf_counter() - t0
            print(f"  ✓  PASSED  ({elapsed:.1f}s)")
            passed += 1
            tracker.record(tc.name, "passed", elapsed, drive.get_calls())
        except AssertionError as exc:
            elapsed = time.perf_counter() - t0
            print(f"  ✗  FAILED  ({elapsed:.1f}s)")
            traceback.print_exc()
            _print_call_log(drive)
            failed += 1
            tracker.record(tc.name, "failed", elapsed, drive.get_calls(), str(exc), traceback.format_exc())
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  !  ERROR   ({elapsed:.1f}s)  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            _print_call_log(drive)
            errored += 1
            tracker.record(tc.name, "error", elapsed, drive.get_calls(),
                           f"{type(exc).__name__}: {exc}", traceback.format_exc())

    total = passed + failed + errored
    print(f"\n{'═' * 64}")
    print(f"Results: {passed} passed  |  {failed} failed  |  {errored} errors  |  {total} total")


def _count_markers(log_path: Path) -> tuple[int, int, int, int]:
    """Return (passed, failed, errored, warnings) by scanning the log file."""
    passed = failed = errored = warnings = 0
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            if "✓  PASSED" in line:
                passed += 1
            elif "✗  FAILED" in line:
                failed += 1
            elif "!  ERROR" in line:
                errored += 1
            if WARNING_MARKER in line:
                warnings += 1
    return passed, failed, errored, warnings


async def _run_one(
    model: str,
    run_num: int,
    total_runs: int,
    log_dir: Path,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        log_file = log_dir / f'{model}_run{run_num}.log'
        rel = log_file.relative_to(ROOT)
        _console(f'>>> START  {model} run {run_num}/{total_runs}  ->  {rel}')
        t0 = time.perf_counter()
        crashed = False
        tracker = Tracker(USE_MLFLOW)
        with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
            f.write(f'=== {model} run {run_num}/{total_runs} ===\n\n')
            f.flush()
            token = _current_log.set(f)
            try:
                await _execute_tests(model, tracker, run_num)
            except Exception:
                traceback.print_exc()
                crashed = True
            finally:
                _current_log.reset(token)
        elapsed = time.perf_counter() - t0
        passed, failed, errored, warnings = _count_markers(log_file)
        try:
            tracker.finish(passed, failed, errored, log_file=log_file, crashed=crashed)
        except Exception as exc:
            _console(f'!!! MLflow logging failed for {model} run {run_num}: {type(exc).__name__}: {exc}')
        status = 'CRASH' if crashed else ('OK' if (failed + errored) == 0 else 'FAIL')
        _console(
            f'<<< DONE   {model} run {run_num}/{total_runs}  '
            f'({elapsed:.1f}s)  {status}  '
            f'[p={passed} f={failed} e={errored} w={warnings}]'
        )
        return {
            'model': model, 'run': run_num, 'status': status,
            'passed': passed, 'failed': failed, 'errored': errored,
            'warnings': warnings, 'elapsed': elapsed,
        }


async def main() -> int:
    if not os.environ.get("OPENAI_API_KEY"):
        _console("ERROR: OPENAI_API_KEY is not set.")
        return 1

    log_dir = ROOT / 'functional_tests_logs' / OUTPUT_LOG_FOLDER
    log_dir.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = []
    for model in MODELS:
        n_runs = RUNS_PER_MODEL.get(model, 1)
        for run_num in range(1, n_runs + 1):
            tasks.append(_run_one(model, run_num, n_runs, log_dir, sem))

    _console(
        f'Scheduling {len(tasks)} run(s) across {len(MODELS)} model(s); '
        f'up to {MAX_CONCURRENT} concurrent. Logs -> {log_dir.relative_to(ROOT)}'
    )
    t0 = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_elapsed = time.perf_counter() - t0

    _console('')
    _console('=' * 72)
    _console('SUMMARY')
    _console('=' * 72)
    total_failed = 0
    total_warned = 0
    for r in results:
        tests_with_warn = r['warnings']
        failures = r['failed'] + r['errored']
        total_failed += failures
        total_warned += tests_with_warn
        _console(
            f"  {r['model']:30s} run{r['run']}  {r['status']:5s}  "
            f"passed={r['passed']:2d}  failed={failures:2d}  warnings={tests_with_warn:2d}  "
            f"({r['elapsed']:.1f}s)"
        )
    _console('-' * 72)
    _console(
        f"Totals: {total_failed} failed test(s), {total_warned} test(s) with warnings, "
        f"across {len(results)} run(s) in {total_elapsed:.1f}s"
    )

    any_bad = any(r['status'] != 'OK' for r in results)
    return 1 if any_bad else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
