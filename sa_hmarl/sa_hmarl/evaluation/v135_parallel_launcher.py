"""Generic sharded parallel launcher for v1.35 diagnostic experiments.

Design goals
------------
- Process-level parallelism only (spawn context).
- Deterministic merge independent of worker completion order.
- Atomic shard writes + checksums + config-hash based resume.
- Progress logging every 30 s with CPU/RSS/swap/ETA.
- Retry only failed shards; never delete completed shards.
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import resource
import signal
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _set_worker_thread_limits():
    """Disable multi-threading in numerical/NN libraries inside workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _config_hash(config: Dict[str, Any]) -> str:
    """Stable hash of a JSON-serializable config dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _checksum_arrays(**arrays: np.ndarray) -> str:
    """Stable checksum over numpy arrays."""
    h = hashlib.sha256()
    for key in sorted(arrays.keys()):
        h.update(key.encode("utf-8"))
        arr = np.asarray(arrays[key])
        h.update(arr.tobytes())
        h.update(str(arr.shape).encode("utf-8"))
    return h.hexdigest()[:16]


def _atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class ShardResult:
    shard_id: str
    status: str  # "ok", "error", "skipped"
    output_path: Optional[Path] = None
    error: Optional[str] = None
    elapsed_sec: float = 0.0
    groups: int = 0
    peak_rss_mb: float = 0.0


class ShardedParallelLauncher:
    """Run a worker function over a list of shards with resume and logging."""

    def __init__(
        self,
        output_dir: Path,
        config: Dict[str, Any],
        worker_fn: Callable[[Dict[str, Any], Path], Dict[str, Any]],
        max_workers: Optional[int] = None,
        shard_subdir: str = "shards",
        log_interval_sec: float = 30.0,
        profile: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.config = config
        self.config_hash = _config_hash(config)
        self.worker_fn = worker_fn
        self.shard_dir = self.output_dir / shard_subdir
        self.log_interval_sec = log_interval_sec
        self.profile = profile
        self.max_workers = max_workers

    def _shard_path(self, shard_id: str) -> Path:
        return self.shard_dir / f"{shard_id}.npz"

    def _meta_path(self, shard_id: str) -> Path:
        return self.shard_dir / f"{shard_id}.metadata.json"

    def _is_shard_done(self, shard_id: str) -> bool:
        npz = self._shard_path(shard_id)
        meta = self._meta_path(shard_id)
        if not npz.exists() or not meta.exists():
            return False
        try:
            m = json.loads(meta.read_text(encoding="utf-8"))
            return m.get("config_hash") == self.config_hash and m.get("status") == "ok"
        except Exception:
            return False

    def _profile_worker(self) -> Tuple[int, float]:
        """Run one representative shard in a child process to measure RSS.

        Returns (suggested_max_workers, peak_rss_mb).
        """
        print("[launcher] Profiling one shard to pick worker count...", flush=True)
        tmpdir = Path(tempfile.mkdtemp(prefix="v135_profile_"))
        test_shards = self.config.get("_profile_shards", None)
        if not test_shards:
            return 4, 0.0
        shard = test_shards[0]
        shard_id = shard["shard_id"]
        local_config = {**self.config, "shards": [shard]}

        def _target():
            _set_worker_thread_limits()
            out = self.worker_fn(shard, tmpdir / f"{shard_id}.npz")
            _atomic_write_json(tmpdir / f"{shard_id}.profile.json", out)

        p = mp.get_context("spawn").Process(target=_target)
        p.start()
        p.join()
        peak_rss = 0.0
        try:
            usage = resource.getrusage(resource.RUSAGE_CHILDREN)
            peak_rss = usage.ru_maxrss / 1024.0  # KB -> MB on Linux
        except Exception:
            pass
        # Also try to read from /proc if available
        try:
            for f in tmpdir.glob("*.profile.json"):
                prof = json.loads(f.read_text())
                peak_rss = max(peak_rss, prof.get("peak_rss_mb", 0.0))
        except Exception:
            pass
        # Clean up
        try:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

        mem_total_mb = self._total_memory_mb()
        if peak_rss > 0 and peak_rss < 2000 and mem_total_mb > 6000:
            suggested = 5
        elif peak_rss > 0 and peak_rss < 1500 and mem_total_mb > 8000:
            suggested = 6
        else:
            suggested = 4
        if mem_total_mb - peak_rss * suggested < 3000:
            suggested = max(1, int((mem_total_mb - 3000) / max(peak_rss, 500)))
        print(f"[launcher] Profile peak RSS ~{peak_rss:.0f} MB, total RAM {mem_total_mb:.0f} MB -> suggested workers={suggested}", flush=True)
        return suggested, peak_rss

    @staticmethod
    def _total_memory_mb() -> float:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb / 1024.0
        except Exception:
            pass
        return 16000.0

    @staticmethod
    def _memory_info_mb() -> Dict[str, float]:
        """Return memory info dict with available, used, swap, total."""
        info: Dict[str, float] = {"available": 0.0, "used": 0.0, "swap": 0.0, "total": 0.0}
        try:
            import psutil
            vm = psutil.virtual_memory()
            info["total"] = vm.total / 1024.0 / 1024.0
            info["available"] = vm.available / 1024.0 / 1024.0
            info["used"] = vm.used / 1024.0 / 1024.0
            info["swap"] = psutil.swap_memory().used / 1024.0 / 1024.0
            return info
        except Exception:
            pass
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    parts = line.split()
                    key = parts[0].rstrip(":")
                    if key in ("MemTotal", "MemAvailable", "MemFree", "Buffers", "Cached", "SwapFree"):
                        info[key] = int(parts[1]) / 1024.0
            total = info.get("MemTotal", 0.0)
            avail = info.get("MemAvailable", 0.0)
            if avail == 0.0:
                avail = (
                    info.get("MemFree", 0.0)
                    + info.get("Buffers", 0.0)
                    + info.get("Cached", 0.0)
                )
            used = total - avail if total else 0.0
            info["total"] = total
            info["available"] = avail
            info["used"] = used
            info["swap"] = info.get("SwapFree", 0.0)
            return info
        except Exception:
            return info

    @staticmethod
    def _available_memory_mb() -> float:
        return ShardedParallelLauncher._memory_info_mb()["available"]

    @staticmethod
    def _cpu_percent_per_core() -> float:
        try:
            with open("/proc/stat") as f:
                line = f.readline()
                fields = list(map(int, line.split()[1:]))
                idle = fields[3]
                total = sum(fields)
                return 100.0 * (1.0 - idle / max(total, 1))
        except Exception:
            return 0.0

    @staticmethod
    def _read_heartbeat(hb_path: Path) -> Optional[Dict[str, Any]]:
        try:
            if not hb_path.exists():
                return None
            return json.loads(hb_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def _worker_proc_info(pid: int) -> Dict[str, Any]:
        info = {"cpu_percent": 0.0, "rss_mb": 0.0, "status": "?"}
        try:
            with open(f"/proc/{pid}/stat") as f:
                parts = f.read().split()
                utime = int(parts[13])
                stime = int(parts[14])
                info["status"] = parts[2]
        except Exception:
            return info
        try:
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        info["rss_mb"] = int(line.split()[1]) / 1024.0
                        break
        except Exception:
            pass
        # We do not have a previous sample to compute CPU%, so report 0.0 here.
        # The heartbeat itself carries elapsed_seconds and seconds_per_group.
        return info

    def _active_worker_summary(self, todo_shards: List[Dict[str, Any]], completed: Dict[str, bool]) -> Tuple[int, List[str]]:
        """Return (active_count, summary_lines) from heartbeat files."""
        hb_dirs = [self.output_dir / "heartbeats", self.output_dir / "shards" / "heartbeats"]
        lines: List[str] = []
        active = 0
        for shard in todo_shards:
            sid = shard["shard_id"]
            if completed.get(sid):
                continue
            hb = None
            for hb_dir in hb_dirs:
                hb = self._read_heartbeat(hb_dir / f"{sid}.heartbeat.json")
                if hb is not None:
                    break
            if hb is None:
                continue
            active += 1
            pid = hb.get("pid", 0)
            proc = self._worker_proc_info(pid)
            lines.append(
                f"  worker {pid} shard={sid} {hb.get('split','')} "
                f"req={hb.get('processed_requests',0)}/{self.config.get('requests_per_episode','?')} "
                f"groups={hb.get('collected_groups',0)}/{hb.get('target_groups',0)} "
                f"cands={hb.get('current_group_candidates',0)} "
                f"s/g={hb.get('seconds_per_group',0):.3f} "
                f"rss={proc['rss_mb']:.0f}MB status={proc['status']}"
            )
        return active, lines

    def _run_worker(self, shard: Dict[str, Any]) -> ShardResult:
        """Wrapper executed inside each child process."""
        _set_worker_thread_limits()
        shard_id = shard["shard_id"]
        out_path = self._shard_path(shard_id)
        meta_path = self._meta_path(shard_id)
        t0 = time.perf_counter()
        try:
            result = self.worker_fn(shard, out_path)
            arrays = result.get("arrays", {})
            metadata = {
                "shard_id": shard_id,
                "config_hash": self.config_hash,
                "status": "ok",
                "groups": result.get("groups", 0),
                "elapsed_sec": time.perf_counter() - t0,
                "checksum": _checksum_arrays(**arrays) if arrays else "",
                "version": self.config.get("version", "1.0"),
                "schema": self.config.get("schema", ""),
            }
            if arrays:
                _atomic_save_npz(out_path, **arrays)
            _atomic_write_json(meta_path, metadata)
            return ShardResult(
                shard_id=shard_id,
                status="ok",
                output_path=out_path,
                elapsed_sec=metadata["elapsed_sec"],
                groups=metadata["groups"],
                peak_rss_mb=result.get("peak_rss_mb", 0.0),
            )
        except Exception as exc:
            err = traceback.format_exc()
            metadata = {
                "shard_id": shard_id,
                "config_hash": self.config_hash,
                "status": "error",
                "error": err,
                "elapsed_sec": time.perf_counter() - t0,
            }
            try:
                _atomic_write_json(meta_path, metadata)
            except Exception:
                pass
            return ShardResult(shard_id=shard_id, status="error", error=err)

    def run(self, shards: Sequence[Dict[str, Any]]) -> List[ShardResult]:
        """Run all shards, skipping completed ones, and return results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Write global config for provenance.
        _atomic_write_json(self.output_dir / "launcher_config.json", {
            "config": self.config,
            "config_hash": self.config_hash,
            "max_workers": self.max_workers,
        })

        todo = [s for s in shards if not self._is_shard_done(s["shard_id"])]
        skipped = [s for s in shards if self._is_shard_done(s["shard_id"])]
        print(f"[launcher] total={len(shards)} done={len(skipped)} todo={len(todo)}", flush=True)

        if not todo:
            return [ShardResult(s["shard_id"], "skipped") for s in skipped]

        # Profiling
        if self.profile and self.max_workers is None:
            suggested, _ = self._profile_worker()
            self.max_workers = suggested
        elif self.max_workers is None:
            self.max_workers = 4
        self.max_workers = max(1, min(self.max_workers, len(todo)))
        print(f"[launcher] Using max_workers={self.max_workers}", flush=True)

        ctx = mp.get_context("spawn")
        results: List[ShardResult] = []
        completed = {s["shard_id"]: False for s in todo}
        start_time = time.perf_counter()
        last_log = start_time
        lock = threading.Lock()

        def _progress_logger():
            nonlocal last_log
            while True:
                time.sleep(self.log_interval_sec)
                with lock:
                    n_done = sum(completed.values())
                    if n_done >= len(todo):
                        break
                    now = time.perf_counter()
                    elapsed = now - start_time
                    cpu = self._cpu_percent_per_core()
                    mem = self._memory_info_mb()
                    active, worker_lines = self._active_worker_summary(todo, completed)
                    if n_done > 0 and elapsed > 5.0:
                        groups_per_sec = n_done / elapsed
                        remaining = len(todo) - n_done
                        eta_str = f"{(remaining / max(groups_per_sec, 1e-6)) / 60:.1f}min"
                    else:
                        groups_per_sec = 0.0
                        eta_str = "warming_up"
                    print(
                        f"[launcher] completed={n_done}/{len(todo)} "
                        f"active_workers={active}/{self.max_workers} cpu~{cpu:.1f}% "
                        f"avail={mem['available']:.0f}MB used={mem['used']:.0f}MB swap={mem['swap']:.0f}MB "
                        f"shards/sec={groups_per_sec:.3f} ETA={eta_str}",
                        flush=True,
                    )
                    for wline in worker_lines:
                        print(wline, flush=True)
                    last_log = now

        logger_thread = threading.Thread(target=_progress_logger, daemon=True)
        logger_thread.start()

        with ctx.Pool(processes=self.max_workers) as pool:
            def _callback(res: ShardResult):
                with lock:
                    completed[res.shard_id] = True
                results.append(res)
                if res.status == "error":
                    print(f"[launcher] ERROR shard {res.shard_id}: {res.error[:200]}", flush=True)

            def _error_callback(err):
                print(f"[launcher] pool error: {err}", flush=True)

            for shard in todo:
                pool.apply_async(
                    self._run_worker,
                    args=(shard,),
                    callback=_callback,
                    error_callback=_error_callback,
                )
            pool.close()
            pool.join()

        # Retry failed shards once
        failed_results = [r for r in results if r.status == "error"]
        if failed_results:
            print(f"[launcher] Retrying {len(failed_results)} failed shards...", flush=True)
            retry_shards = [s for s in todo if any(r.shard_id == s["shard_id"] and r.status == "error" for r in results)]
            results = [r for r in results if r.status != "error"]
            with ctx.Pool(processes=min(self.max_workers, len(retry_shards))) as pool:
                retry_res = pool.map(self._run_worker, retry_shards)
            results.extend(retry_res)

        # Add skipped results
        results.extend([ShardResult(s["shard_id"], "skipped") for s in skipped])

        total_elapsed = time.perf_counter() - start_time
        n_ok = sum(1 for r in results if r.status == "ok")
        n_err = sum(1 for r in results if r.status == "error")
        print(f"[launcher] Finished: ok={n_ok} skipped={len(skipped)} error={n_err} elapsed={total_elapsed/60:.2f}min", flush=True)
        return results

