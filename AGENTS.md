# Project Environment and Mandatory Preflight

> **MANDATORY:** Every AI agent and developer must read this file before
> inspecting code, running tests, launching experiments, or changing the
> environment. Do not spend time searching for another Python installation.
> The canonical environment and commands are locked below.

Last audited: 2026-08-01 (Asia/Shanghai)

## 1. Canonical Paths

| Purpose | Windows path | Ubuntu WSL path |
|---|---|---|
| Git root | `D:\project\DNN_Agent` | `/mnt/d/project/DNN_Agent` |
| Working/code root | `D:\project\DNN_Agent\sa_hmarl` | `/mnt/d/project/DNN_Agent/sa_hmarl` |
| Python package | `D:\project\DNN_Agent\sa_hmarl\sa_hmarl` | `/mnt/d/project/DNN_Agent/sa_hmarl/sa_hmarl` |
| Only project virtual environment | `D:\project\DNN_Agent\.venv` | `/mnt/d/project/DNN_Agent/.venv` |
| Only project Python | Linux executable; do not run from Windows | `/mnt/d/project/DNN_Agent/.venv/bin/python` |
| Pure RMSA main package | `...\sa_hmarl\sa_hmarl\pure_rmsa_v13` | `.../sa_hmarl/sa_hmarl/pure_rmsa_v13` |
| PDS-RMSA main package | `...\sa_hmarl\sa_hmarl\pds_rmsa` | `.../sa_hmarl/sa_hmarl/pds_rmsa` |

The Git repository root and Python working root are different. The canonical
working directory for Python commands is:

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl
```

From that directory, use `PYTHONPATH=.`. Some legacy documents run from the
Git root and use `PYTHONPATH=sa_hmarl`; do not mix the two conventions.

## 2. Mandatory Execution Rules

1. Run project Python only inside the `Ubuntu` WSL2 distribution.
2. Use `/mnt/d/project/DNN_Agent/.venv/bin/python` explicitly. Activation is
   optional and must not be used as evidence that the right Python was found.
3. Never use Windows Python, `/usr/bin/python3`, plain `python` before
   activation, Conda, or a newly created virtual environment.
4. Do not install, upgrade, or remove packages unless the user explicitly
   approves an environment change.
5. This is a `uv` environment and has no `pip` module. Do not run
   `python -m pip`. Use the existing `/home/lds/.local/bin/uv` only after
   approval.
6. Do not set `SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL=1` unless the protocol
   explicitly calls for the optional native-kernel appendix. Main fair
   Direct-Sketch experiments are Python/NumPy and require this variable unset.
7. Before launching a long task, check for a duplicate process and inspect the
   intended output directory. Never restart a healthy run.
8. Formal latency comparisons must use one worker, the same Python runtime,
   the same route-cache warmup, and implementation-matched backends.
9. GPU training normally uses one worker/seed at a time unless a locked
   protocol explicitly says otherwise. Avoid GPU contention.
10. Preserve existing dirty worktree changes. Do not clean, reset, or revert
    files that are unrelated to the current task.

## 3. Mandatory Preflight

Run this at the beginning of every project task:

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl

PY=/mnt/d/project/DNN_Agent/.venv/bin/python
test -x "$PY"

echo "cwd=$(pwd)"
echo "python=$($PY -c 'import sys; print(sys.executable)')"
echo "version=$($PY -c 'import platform; print(platform.python_version())')"

PYTHONPATH=. "$PY" -c \
  'import sa_hmarl, torch; print(sa_hmarl.__file__); print("cuda", torch.cuda.is_available())'

git -C /mnt/d/project/DNN_Agent status --short
pgrep -af 'python.*sa_hmarl' || true
echo "native_direct=${SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL:-unset}"
```

Expected critical values:

```text
python=/mnt/d/project/DNN_Agent/.venv/bin/python
version=3.14.4
sa_hmarl=/mnt/d/project/DNN_Agent/sa_hmarl/sa_hmarl/__init__.py
cuda=True
native_direct=unset
```

If any critical value differs, stop and diagnose. Do not create another
environment as a workaround.

## 4. Canonical Command Templates

### Enter WSL from Windows

```powershell
wsl -d Ubuntu
```

The warning about a localhost proxy not being mirrored into WSL NAT may appear
on startup. It is informational when the Linux command still exits with code
0; judge success by the command exit code and actual output.

### Optional activation

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl
source /mnt/d/project/DNN_Agent/.venv/bin/activate
export PYTHONPATH=.
```

Absolute interpreter paths remain preferred for automation:

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl
PYTHONPATH=. /mnt/d/project/DNN_Agent/.venv/bin/python -m <module> <args>
```

### Tests

```bash
cd /mnt/d/project/DNN_Agent/sa_hmarl
PYTHONPATH=. /mnt/d/project/DNN_Agent/.venv/bin/python -m pytest <test-path> -q
```

Run focused tests first. Run a full suite only when the blast radius or locked
protocol requires it; several project suites take many minutes.

### Inspect installed packages

```bash
/home/lds/.local/bin/uv pip list \
  --python /mnt/d/project/DNN_Agent/.venv/bin/python
```

### Process and GPU checks

```bash
pgrep -af 'python.*sa_hmarl'
ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd | grep '[p]ython'
nvidia-smi
```

Use the exact module name in `pgrep` before launching a formal task. For
multi-worker CPU jobs, verify one coordinator and exactly the requested number
of workers.

## 5. Audited Environment Snapshot

### Host and WSL

| Item | Audited value |
|---|---|
| WSL distribution | Ubuntu 26.04 LTS |
| WSL version | WSL2 |
| Linux kernel | `6.6.87.2-microsoft-standard-WSL2` |
| Linux user | `lds` |
| Shell | `/bin/bash` |
| CPU | AMD Ryzen 5 9600X, 6 cores / 12 threads |
| WSL-visible RAM | 15 GiB |
| WSL swap | 4 GiB |
| Project disk | `D:` mounted at `/mnt/d`, 632 GiB total at audit time |

RAM usage and free disk space are dynamic. Recheck them before clone-heavy or
multi-worker rollout experiments.

### GPU and CUDA

| Item | Audited value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 |
| VRAM | 12,227 MiB |
| Windows/WSL driver | 591.86 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| `torch.cuda.is_available()` | `True` |

### Python and core packages

| Item | Audited value |
|---|---|
| Python | CPython 3.14.4, GCC 15.2.0 |
| Environment manager | `uv 0.11.11` |
| NumPy | 2.4.4 |
| SciPy | 1.17.1 |
| NetworkX | 3.6.1 |
| pandas | 3.0.2 |
| PyYAML | 6.0.3 |
| tqdm | 4.67.3 |
| pytest | 9.0.3 |
| Matplotlib | 3.10.9 |
| psutil | 7.2.2 |
| TensorBoard | 2.20.0 |

`scikit-learn` is not installed. Do not assume a package exists merely
because a proposed experiment mentions it. Check with `uv pip list` or a
read-only import probe.

The root `requirements.txt` contains broad minimum versions only. It is not a
complete lock file and does not reproduce the audited environment exactly.

## 6. Repository Layout and Artifact Rules

The importable package is the inner `sa_hmarl/sa_hmarl` directory. Current
work should generally target modules below it:

```text
/mnt/d/project/DNN_Agent/sa_hmarl/
  sa_hmarl/
    pure_rmsa_v13/
      rollout_lab/
      direct_sketch_residual_rl/
      artifacts/
    pds_rmsa/
      training/
      tests/
      artifacts/
```

There are legacy or parallel directories elsewhere in the repository. Do not
select one by name alone. Confirm the import path and the protocol document
used by the requested experiment.

Artifact rules:

- Use a new protocol-specific output directory for a new experiment.
- Never overwrite old formal artifacts or checkpoints.
- Never load an old checkpoint unless the locked protocol explicitly permits
  it.
- Smoke outputs must never be reported as formal results.
- Record seeds, warmup, evaluation requests, workers, device, protocol ID,
  backend, and command in the output manifest/report.
- For paired comparisons, use the same traces and report per-seed differences
  and confidence intervals.

## 7. Long-Run Safety Checklist

Before launch:

1. Read the experiment's protocol lock and existing output directory.
2. Run targeted tests and a small smoke test.
3. Confirm the interpreter and `PYTHONPATH` with the mandatory preflight.
4. Confirm no duplicate process exists.
5. Confirm CUDA/backend/worker settings match the protocol.
6. Confirm formal seeds do not overlap training, calibration, diagnosis, or
   smoke seeds.
7. Prefer foreground, tool-managed execution. If a persistent terminal is
   required, record PID, command, log path, and output path immediately.

During a run:

- Do not edit code used by the active process.
- Do not restart while the process is healthy.
- Check process uniqueness, worker count, finite losses, device audit, log
  progress, memory, and disk space.
- Detached WSL jobs may disappear when their parent terminal/tool session is
  cleaned up; do not assume a missing detached job completed successfully.

After completion:

- Verify expected artifacts and row/seed counts before interpreting metrics.
- Verify protocol ID and backend fields from artifacts, not from memory.
- Preserve failed and negative results; do not silently replace them.

## 8. Fair Latency Measurement Rules

The project contains old and optimized latency artifacts. Never mix them.
The primary fair reference is the pure-Python bit-parallel early-exit KSP-FF
backend. It stops at the first feasible path and does not materialize the full
candidate list. The audited 10-seed selector latencies are:

| Topology | KSP-FF K=5 | KSP-FF K=50 | Direct Sketch | Direct/K=5 | Direct/K=50 |
|---|---:|---:|---:|---:|---:|
| NSFNET | 0.0128 ms | 0.0197 ms | 0.4453 ms | 34.88x | 22.58x |
| USNET | 0.0150 ms | 0.0230 ms | 0.4321 ms | 28.88x | 18.77x |
| JPN48 | 0.0180 ms | 0.0279 ms | 0.5470 ms | 30.31x | 19.61x |

The earlier `1.13x-1.19x` result compared Direct Sketch against a legacy
Python KSP path that called `env.build_candidates()` and materialized all
candidates before returning the first feasible one. Its blocking decisions
were correct, but its latency was not representative of standard early-exit
KSP-FF. Do not use that ratio as the primary latency claim.

The older Direct values `0.588/0.507/0.839 ms` are pre-optimization baselines,
not current latency. Optional C-kernel results must be compared only against
an equally compiled KSP-FF implementation and must not replace the main
Python-vs-Python result. See the formal corrected report under
`direct_vs_ksp_k5_k50_python_early_exit_10seed_v1_20260801`.

## 9. Known Environment Traps

- The WSL startup proxy warning is common and does not by itself indicate a
  failed command.
- Windows `PATH` entries are visible inside WSL and include Windows Python
  installations. Never use them for this project.
- `/usr/bin/python3` currently has the same version number as the venv base,
  but it is not the project environment.
- `.venv/bin/python` is a Linux executable/symlink. PowerShell cannot execute
  it directly; invoke it through WSL.
- `python -m pip` fails because pip is intentionally absent from the uv venv.
- Do not compare multi-worker wall-clock throughput with one-worker
  per-request selector latency.
- Do not infer the active implementation from similarly named top-level
  directories. Verify `sa_hmarl.__file__` and the target module path.
