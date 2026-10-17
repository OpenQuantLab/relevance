"""
Background backtest jobs.

Backtests run in a worker thread inside the web process so the UI can show a
live progress bar, which is why the dashboard must be served with a single
gunicorn worker: the job registry below is per-process state.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid

from relevance.backtest import lab


class Job:
    def __init__(self, specs: list[lab.SignalSpec], n_perm: int):
        self.id = uuid.uuid4().hex[:12]
        self.specs = specs
        self.n_perm = n_perm
        self.status = "running"          # running | done | error | cancelled
        self.stage = "Loading prices…"
        self.step = 0
        self.total = 0
        self.spec_index = 0
        self.run_ids: list[str] = []
        self.error: str | None = None
        self.cancelled = False

    @property
    def fraction(self) -> float:
        per_spec = 1 / len(self.specs)
        inner = (self.step / self.total) if self.total else 0.0
        return min(1.0, self.spec_index * per_spec + inner * per_spec)


JOBS: dict[str, Job] = {}
_JOB_LOCK = threading.Lock()


def _run_job(job: Job) -> None:
    try:
        for i, spec in enumerate(job.specs):
            if job.cancelled:
                break
            job.spec_index = i
            job.step, job.total = 0, 0
            job.stage = f"[{i + 1}/{len(job.specs)}] loading prices…"
            prices = lab.load_prices(spec)

            job.stage = f"[{i + 1}/{len(job.specs)}] building panel…"
            panel = lab.build_panel(prices, spec)

            def progress(step, total, when, _i=i):
                job.step, job.total = step, total
                job.stage = (f"[{_i + 1}/{len(job.specs)}] "
                             f"backtesting {when} ({step}/{total})")

            t0 = time.time()
            results = lab.run_backtest(
                panel, spec, progress=progress,
                should_cancel=lambda: job.cancelled,
            )
            if job.cancelled:
                break

            job.stage = f"[{i + 1}/{len(job.specs)}] evaluating…"
            report = lab.evaluate(results, spec, n_perm=job.n_perm)
            summary = dict(report["summary"])
            if report["perm"]:
                summary["p_ic"] = report["perm"]["p_ic"]
                summary["p_sharpe"] = report["perm"]["p_sharpe"]
            job.run_ids.append(
                lab.save_run(spec, results, summary,
                             elapsed=time.time() - t0, perm=report["perm"])
            )
        job.status = "cancelled" if job.cancelled else "done"
        job.stage = ("Cancelled." if job.cancelled
                     else f"Finished {len(job.run_ids)} run(s).")
    except Exception as exc:                      # surfaced in the UI
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.stage = "Failed."
        traceback.print_exc()


def start_job(specs: list[lab.SignalSpec], n_perm: int) -> Job:
    job = Job(specs, n_perm)
    with _JOB_LOCK:
        JOBS[job.id] = job
        for old in [k for k, v in JOBS.items()
                    if v.status != "running" and k != job.id][:-10]:
            JOBS.pop(old, None)
    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return job


def job_in_flight() -> bool:
    with _JOB_LOCK:
        return any(j.status == "running" for j in JOBS.values())
