#!/usr/bin/env python3
"""
smartnotes functional tests — real LLM, fake MCP stub.

Usage:
    python -m functional_tests.run
    python -m functional_tests.run --model gpt-4o-mini
    python -m functional_tests.run --test append_terapia
    python -m functional_tests.run --list

Requires:
    OPENAI_API_KEY environment variable (or .env file in repo root).

Tests never run via pytest because pytest.ini limits testpaths to 'tests/'.
"""
import argparse
import asyncio
import logging
import os
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

# ── Make the repo root importable when invoked as a script ───────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── Load .env before importing app modules (pydantic-settings reads at import) ─
from dotenv import load_dotenv  # noqa: E402 — must come after sys.path tweak
load_dotenv(ROOT / ".env")

# ── Env vars required by pydantic-settings / app.config ──────────────────────
os.environ.setdefault("OPENAI_API_KEY",       os.environ.get("OPENAI_API_KEY", ""))
os.environ.setdefault("APP_API_KEY",          "functional-test-key")
os.environ.setdefault("ALLOWED_EMAILS",       "test@example.com")

from functional_tests.fake_drive import FakeDrive                   # noqa: E402
from functional_tests.functional_tests import ALL_TESTS, TestAgent, make_planner  # noqa: E402
from functional_tests.tracking import EXPERIMENT, Tracker           # noqa: E402

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
# DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"
DEFAULT_MODEL = "gpt-5.4-nano"


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(levelname)s [%(name)s] %(message)s",
        level=logging.WARNING,
    )

    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))

    for name in ("agent", "planner"):
        log = logging.getLogger(name)
        log.propagate = False
        log.setLevel(level)
        if not log.handlers:
            log.addHandler(handler)


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="smartnotes functional tests (real LLM, fake MCP)"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--test", default=None,
        metavar="NAME",
        help="Run only the named test (see --list for names)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available test names and exit",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show agent/planner log output",
    )
    parser.add_argument(
        "--mlflow", action="store_true",
        help="Log results to MLflow (server from MLFLOW_TRACKING_URI)",
    )
    parser.add_argument(
        "--mlflow-batch-name", default="adhoc",
        help="MLflow 'batch' tag grouping runs to compare, e.g. v9b (only used with --mlflow)",
    )
    parser.add_argument(
        "--mlflow-experiment", default=EXPERIMENT,
        help=f"MLflow experiment name (default: {EXPERIMENT})",
    )
    parser.add_argument("--repeat", type=int, default=1, help="MLflow repeat number")
    parser.add_argument(
        "--trace", action="store_true",
        help="With --mlflow: record one MLflow trace per test (OpenAI calls nested, "
             "pass/fail as feedback) so tests show in the run's evaluation panel",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if args.list:
        print("Available tests:")
        for tc in ALL_TESTS:
            print(f"  {tc.name:30s}  {tc.description}")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)

    tests = ALL_TESTS
    if args.test:
        tests = [t for t in ALL_TESTS if t.name == args.test]
        if not tests:
            names = [t.name for t in ALL_TESTS]
            print(f"ERROR: test '{args.test}' not found. Available: {names}", file=sys.stderr)
            sys.exit(1)

    def _print_call_log(drive: FakeDrive) -> None:
        calls = drive.get_calls()
        if not calls:
            print("     Tool calls: (none)")
            return
        print(f"     Tool calls ({len(calls)}):")
        for c in calls:
            print(f"       [{c['step']}] {c['tool']}  {c['args']}")

    planner = make_planner()
    resolved = planner._model_registry.resolve(args.model)
    prompt_first_line = resolved.prompt.splitlines()[0]
    print(f"model: {args.model}  →  {resolved.full_name}")
    print(f"prompt: {prompt_first_line}")

    tracker = Tracker(args.mlflow, args.mlflow_experiment)
    tracker.start(args.model, resolved, args.mlflow_batch_name, args.repeat,
                  max_plan_steps=10, n_tests=len(tests))
    trace = args.mlflow and args.trace
    if trace:
        # Traces link to the active run, so open the tracker's run in the fluent API.
        import mlflow
        from mlflow.entities import AssessmentSource
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.openai.autolog()
        mlflow.start_run(run_id=tracker.run_id)

    passed = failed = errored = 0

    for tc in tests:
        print(f"\n{'─' * 64}")
        print(f"TEST  {tc.name}")
        print(f"      {tc.description}")

        # Fresh drive + agent per test — isolated in-memory state, no disk writes
        drive = FakeDrive()
        agent = TestAgent(fake_drive=drive, planner=planner, max_plan_steps=10, planner_model=args.model)

        status, error, tb = "passed", "", ""
        t0 = time.perf_counter()
        # One root span per test; autologged OpenAI calls nest under it.
        with (mlflow.start_span(name=tc.name, span_type="AGENT") if trace else nullcontext()) as span:
            if span:
                span.set_inputs({"test": tc.name, "description": tc.description})
            try:
                await tc.fn(agent, drive)
            except AssertionError as exc:
                status, error, tb = "failed", str(exc), traceback.format_exc()
            except Exception as exc:
                status, error, tb = "error", f"{type(exc).__name__}: {exc}", traceback.format_exc()
            if span:
                span.set_outputs({"status": status, "error": error,
                                  "tool_calls": [f"[{c['step']}] {c['tool']} {c['args']}" for c in drive.get_calls()]})
        elapsed = time.perf_counter() - t0

        if status == "passed":
            print(f"  ✓  PASSED  ({elapsed:.1f}s)")
            passed += 1
        else:
            if status == "failed":
                print(f"  ✗  FAILED  ({elapsed:.1f}s)")
                failed += 1
            else:
                print(f"  !  ERROR   ({elapsed:.1f}s)  {error}")
                errored += 1
            print(tb, end="", file=sys.stderr)
            _print_call_log(drive)
        tracker.record(tc.name, status, elapsed, drive.get_calls(), error, tb)

        if span:
            mlflow.flush_trace_async_logging()  # log_feedback needs the trace already stored
            mlflow.log_feedback(trace_id=span.trace_id, name="passed", value=status == "passed",
                                rationale=error or None,
                                source=AssessmentSource("CODE", "functional_tests"))

    total = passed + failed + errored
    print(f"\n{'═' * 64}")
    print(f"Results: {passed} passed  |  {failed} failed  |  {errored} errors  |  {total} total")

    if trace:
        mlflow.end_run()
    tracker.finish(passed, failed, errored)

    sys.exit(0 if (failed + errored) == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
