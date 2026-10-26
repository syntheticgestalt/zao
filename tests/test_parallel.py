"""Where the CPUs go. The numbers differ by machine; the rules hold on all of them."""

import joblib
import pytest

import zao.parallel as p


def test_a_narrowed_process_narrows_everything(monkeypatch):
    """A container, a taskset or LOKY_MAX_CPU_COUNT limits the budget, and every answer follows."""
    monkeypatch.setattr(p.joblib, "cpu_count", lambda only_physical_cores=False: 4)
    assert p.cpu_budget() == 4
    assert p.process_workers(10**6) == 4
    assert p.process_workers(3) == 3
    assert p.native_threads() <= 4


def test_a_persons_own_omp_setting_is_not_read_as_a_cpu_limit(monkeypatch):
    """OMP_NUM_THREADS=1 means one OpenMP thread per process, not one CPU for the program."""
    if joblib.cpu_count() < 2:
        pytest.skip("a one-CPU machine has nothing to narrow")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    assert p.cpu_budget() > 1


def test_a_share_handed_down_by_joblib_is_read():
    """Inside somebody else's Parallel, this process has a share of the machine, not all of it."""
    from joblib import Parallel, delayed

    outer = p.cpu_budget()
    if outer < 4:
        pytest.skip("too few CPUs to be given a share of")
    inside = Parallel(n_jobs=4)(delayed(p.cpu_budget)() for _ in range(1))[0]
    assert inside < outer
