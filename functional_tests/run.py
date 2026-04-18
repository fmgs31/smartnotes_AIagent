#!/usr/bin/env python3
"""
SmarNotes functional tests — real LLM, fake MCP stub.

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

from functional_tests.fake_drive import FakeDrive, DRIVE_ROOT       # noqa: E402
from functional_tests.functional_tests import ALL_TESTS, TestAgent, make_planner  # noqa: E402

DEFAULT_MODEL = "gpt-5-nano-2025-08-07"
# DEFAULT_MODEL = "gpt-5.4-nano-2026-03-17"


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
        description="SmarNotes functional tests (real LLM, fake MCP)"
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
    drive   = FakeDrive(DRIVE_ROOT)

    passed = failed = errored = 0

    for tc in tests:
        print(f"\n{'─' * 64}")
        print(f"TEST  {tc.name}")
        print(f"      {tc.description}")

        # Fresh agent per test (clears tool-discovery cache)
        agent = TestAgent(fake_drive=drive, planner=planner, max_plan_steps=10, planner_model=args.model)
        drive.clear_calls()

        t0 = time.perf_counter()
        try:
            await tc.fn(agent, drive)
            elapsed = time.perf_counter() - t0
            print(f"  ✓  PASSED  ({elapsed:.1f}s)")
            passed += 1
        except AssertionError as exc:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"  ✗  FAILED  ({elapsed:.1f}s)")
            traceback.print_exc()
            _print_call_log(drive)
            failed += 1
        except Exception as exc:
            import traceback
            elapsed = time.perf_counter() - t0
            print(f"  !  ERROR   ({elapsed:.1f}s)  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            _print_call_log(drive)
            errored += 1

    total = passed + failed + errored
    print(f"\n{'═' * 64}")
    print(f"Results: {passed} passed  |  {failed} failed  |  {errored} errors  |  {total} total")

    sys.exit(0 if (failed + errored) == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
