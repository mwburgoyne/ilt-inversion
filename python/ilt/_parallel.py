"""
Parallel evaluation of inverse Laplace transforms across time points.

Uses ProcessPoolExecutor for multi-core parallelism.  Each worker
re-initialises its own mpmath/gmpy2 precision context to avoid
shared-state issues across processes.
"""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Callable, List


def _gwr_worker(
    time: float,
    fn: Callable[..., Any],
    M: int,
    prec: int,
    backend: str,
) -> Any:
    """
    Worker function for parallel GWR evaluation.

    Must be a module-level function (picklable).  Each worker
    re-creates its own ops and coefficients to avoid cross-process
    sharing of mpmath state.
    """
    from mpmath import mp
    from ._backends import get_ops, precision_context
    from ._gwr import gwr_single, precompute_coefficients

    mp.dps = prec
    ops, _ = get_ops(backend)
    with precision_context(prec):
        coeffs = precompute_coefficients(M, prec, ops)
        return gwr_single(fn, time, M, prec, ops, coeffs)


def parallel_gwr(
    fn: Callable[..., Any],
    times: List[float],
    M: int,
    prec: int,
    backend: str,
    workers: int,
) -> List[Any]:
    """
    Evaluate GWR across time points in parallel.

    Note: fn must be picklable (module-level function, not a lambda or closure).
    """
    max_workers = min(workers, len(times), os.cpu_count() or 1)
    worker = partial(_gwr_worker, fn=fn, M=M, prec=prec, backend=backend)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(worker, times))
