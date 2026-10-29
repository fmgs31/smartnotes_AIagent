import os
import subprocess
import sys
from pathlib import Path

OUTPUT_LOG_FOLDER = 'v7'
USE_MLFLOW = True  # needs MLFLOW_TRACKING_URI pointing at a running `mlflow server`
USE_TRACE = True  # with USE_MLFLOW: one trace per test, shown in the run's evaluation panel
MLFLOW_EXPERIMENT = 'planner-behavioural'
RUNS_PER_MODEL = {
    'gpt-5.4-nano_OR7':3,
    'gpt-5.4-nano_CB7':3,
    'gpt-5-nano_CB7':2,
    'gpt-5-nano_OR7':2
}
MODELS = list(RUNS_PER_MODEL.keys())


def run_one(model: str, run_num: int, total_runs: int, log_dir: Path, base_dir: Path) -> int:
    log_file = log_dir / f'{model}_run{run_num}.log'
    header = f'=== {model} run {run_num}/{total_runs} | {log_file.relative_to(base_dir)} ==='
    print(f'\n{header}', flush=True)

    cmd = [sys.executable, '-u', '-m', 'functional_tests.run', '--model', model]
    if USE_MLFLOW:
        cmd += ['--mlflow', '--mlflow-experiment', MLFLOW_EXPERIMENT,
                '--mlflow-batch-name', OUTPUT_LOG_FOLDER, '--repeat', str(run_num)]
        if USE_TRACE:
            cmd.append('--trace')

    # Force the child's stdout/stderr to UTF-8 so its print() of e.g. '\u2192' doesn't crash
    # under Windows' default cp1252 console encoding when piped.
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'

    with open(log_file, 'w', encoding='utf-8', errors='replace') as f:
        f.write(header + '\n')
        f.write('cmd: ' + ' '.join(cmd) + '\n\n')
        f.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            cwd=str(base_dir),
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            f.write(line)
            f.flush()
            try:
                sys.stdout.write(line)
            except UnicodeEncodeError:
                enc = sys.stdout.encoding or 'ascii'
                sys.stdout.write(line.encode(enc, errors='replace').decode(enc, errors='replace'))
            sys.stdout.flush()
        proc.wait()

        footer = f'=== exit code: {proc.returncode} ==='
        f.write('\n' + footer + '\n')

    print(footer, flush=True)
    return proc.returncode


def main() -> int:
    base_dir = Path(__file__).resolve().parent.parent
    log_dir = base_dir / 'functional_tests_logs' / OUTPUT_LOG_FOLDER
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, int, int]] = []
    for model in MODELS:
        n_runs = RUNS_PER_MODEL.get(model, 1)
        for run_num in range(1, n_runs + 1):
            rc = run_one(model, run_num, n_runs, log_dir, base_dir)
            results.append((model, run_num, rc))

    print('\n=== summary ===', flush=True)
    for model, run_num, rc in results:
        status = 'OK' if rc == 0 else f'FAIL({rc})'
        print(f'  {model} run{run_num}: {status}', flush=True)

    return 0 if all(rc == 0 for _, _, rc in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
