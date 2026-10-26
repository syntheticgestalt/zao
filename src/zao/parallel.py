"""How much of the machine to use, decided in one place."""

import os

import joblib


def cpu_budget() -> int:
    """CPUs this process may use.

    ``joblib.cpu_count`` reads the affinity mask, a container's cgroup quota and
    ``LOKY_MAX_CPU_COUNT``. Inside a joblib worker the share it was given,
    ``OMP_NUM_THREADS``, bounds it further.
    """
    budget = joblib.cpu_count()
    if "LOKY_MAX_CPU_COUNT" not in os.environ and _inside_a_joblib_worker():
        share = os.environ.get("OMP_NUM_THREADS", "")
        # joblib writes a plain integer; OpenMP's list syntax ("4,2") is someone else's.
        if share.isdigit():
            budget = min(budget, int(share))
    return max(1, budget)


def _inside_a_joblib_worker() -> bool:
    """Whether this process is a worker that joblib started and gave a share to."""
    try:
        from joblib.externals.loky import process_executor

        return process_executor._CURRENT_DEPTH > 0
    except Exception:  # pragma: no cover - depends on joblib's internals
        return False


def process_workers(n_tasks: int, requested: int = -1) -> int:
    """Worker processes for per-molecule work.

    Parameters
    ----------
    n_tasks : int
        How many molecules there are. Workers beyond that have nothing to do.
    requested : int, default=-1
        -1 means the whole budget, 1 means stay in this process, and a positive
        number is an upper bound.
    """
    budget = cpu_budget() if requested < 1 else min(requested, cpu_budget())
    return min(budget, max(1, n_tasks))


def native_threads(cap: int | None = None) -> int:
    """Threads for a library that runs its own pool, such as LightGBM or RDKit.

    Real cores, never every logical CPU, and never more than ``cap``.
    """
    budget = cpu_budget()
    physical = joblib.cpu_count(only_physical_cores=True) or budget
    threads = min(physical, budget)
    if cap is not None:
        threads = min(threads, cap)
    return max(1, threads)
