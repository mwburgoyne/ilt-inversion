"""
ilt - Numerical Inverse Laplace Transforms
===========================================

Fast, accurate inverse Laplace transforms using the Gaver-Wynn-Rho (GWR)
algorithm with arbitrary-precision arithmetic, plus a Fixed Talbot fallback
for well-behaved transforms.

Quick start::

    from ilt import gwr
    from mpmath import mp

    # Define Laplace transform: L{e^(-t)} = 1/(s+1)
    def F(s):
        return 1 / (s + 1)

    # Single time point
    result = gwr(F, 1.0)           # -> 0.367879...

    # Array of time points
    results = gwr(F, [0.1, 1.0, 10.0])

    # With parallel evaluation
    results = gwr(F, times, workers=4)

Performance backends (auto-detected, fastest first):
    - Rust/MPFR:    ~70x faster (pip install with Rust toolchain)
    - gmpy2:        ~10x faster GWR internal arithmetic
    - python-flint: ~15x faster Bessel functions (via ARB)

References:
    Valko & Abate (2004), Computers and Mathematics with Application 48(3).
"""

from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
from mpmath import mp

from ._backends import get_ops, precision_context, HAS_GMPY2
from ._bessel import besselk, besseli, HAS_FLINT
from ._gwr import gwr_single, precompute_coefficients
from ._talbot import talbot_single

# Try importing Rust acceleration
try:
    from ._native import gwr_rust as _gwr_rust, besseli_rust, besselk_rust
    HAS_RUST = True
except ImportError:
    _gwr_rust = None
    besseli_rust = None
    besselk_rust = None
    HAS_RUST = False

__all__ = [
    "gwr", "talbot",
    "besselk", "besseli",
    "HAS_GMPY2", "HAS_FLINT", "HAS_RUST",
]


def gwr(
    fn: Union[Callable[[float], Any], Callable[[float, int], Any]],
    time: Union[float, Iterable[float], np.ndarray],
    M: int = 32,
    precin: Optional[int] = None,
    backend: str = "auto",
    workers: int = 1,
    as_float: bool = True,
) -> Any:
    """
    Inverse Laplace transform via the Gaver-Wynn-Rho algorithm.

    Parameters
    ----------
    fn : callable
        Laplace-domain function.  Signature: fn(s) or fn(s, prec).
    time : float, list, or ndarray
        Time value(s) at which to evaluate the inverse transform.
    M : int
        Number of Gaver functional terms (default 32).  Higher = more accurate
        but slower.  Typical range: 6-12 for smooth transforms, 32+ for general,
        768+ for hard oscillatory transforms.
    precin : int or None
        Decimal digits of precision.  Default: round(2.1 * M).
    backend : str
        'auto' (Rust if available, else gmpy2 if available, else mpmath),
        'rust', 'gmpy2', or 'mpmath'.
    workers : int
        Parallel workers for array inputs.  >1 requires fn to be picklable
        (module-level function, not a lambda).  Default 1 (sequential).
    as_float : bool
        If True (default), convert results to Python float.
        If False, return mpmath.mpf values with full precision.

    Returns
    -------
    float or list[float] or np.ndarray
        Inverse Laplace transform evaluated at the given time(s).
        Type matches the input: scalar -> scalar, list -> list, ndarray -> ndarray.
    """
    prec = round(21 * M / 10) if precin is None else precin

    # Rust fast path: fn(s) -> float, all evaluated at f64.
    # Only valid when M is small enough that f64 (~16 digits) suffices.
    # GWR needs ~2.1*M decimal digits; f64 provides ~15.9.
    use_rust = (
        HAS_RUST
        and backend in ("auto", "rust")
        and workers <= 1
        and prec <= 15
    )

    if use_rust:
        times_list = _to_times_list(time)
        if times_list is not None:
            results = _gwr_rust(fn, times_list, M, prec)
            return _format_output(results, time, as_float)

    # Python path
    saved_dps = mp.dps
    mp.dps = prec

    ops, backend_name = get_ops(backend if backend != "rust" else "auto")
    convert = (lambda v: float(ops["to_float"](v))) if as_float else (lambda v: _to_mpf(v, ops, prec))

    try:
        with precision_context(prec):
            coeffs = precompute_coefficients(M, prec, ops)
            fn_arity = _detect_arity(fn)

            if not isinstance(time, Iterable):
                val = gwr_single(fn, time, M, prec, ops, coeffs, fn_arity)
                return convert(val)

            if isinstance(time, np.ndarray):
                if time.ndim >= 2:
                    time = np.squeeze(time)
                if time.ndim < 1:
                    val = gwr_single(fn, time.item(), M, prec, ops, coeffs, fn_arity)
                    return np.array([convert(val)])

                times = list(time)
                if workers > 1:
                    from ._parallel import parallel_gwr
                    results = parallel_gwr(fn, times, M, prec, backend, workers)
                else:
                    results = [gwr_single(fn, t, M, prec, ops, coeffs, fn_arity) for t in times]
                return np.array([convert(r) for r in results],
                                dtype=float if as_float else object)

            times = list(time)
            if workers > 1:
                from ._parallel import parallel_gwr
                results = parallel_gwr(fn, times, M, prec, backend, workers)
            else:
                results = [gwr_single(fn, t, M, prec, ops, coeffs, fn_arity) for t in times]
            return [convert(r) for r in results]

    finally:
        mp.dps = saved_dps


def _to_times_list(time):
    """Convert time input to a flat list of floats, or None if not possible."""
    if isinstance(time, np.ndarray):
        return time.ravel().tolist()
    if isinstance(time, Iterable):
        return [float(t) for t in time]
    return [float(time)]


def _format_output(results, time, as_float):
    """Format Rust results to match input type."""
    if isinstance(time, np.ndarray):
        return np.array(results, dtype=float if as_float else object)
    if isinstance(time, Iterable):
        return results if as_float else results
    return results[0]


def _detect_arity(fn: Callable) -> int:
    """Detect whether fn takes 1 or 2 positional args (without calling it)."""
    from inspect import signature, Parameter
    try:
        sig = signature(fn)
        positional = sum(
            1 for p in sig.parameters.values()
            if p.default is Parameter.empty
            and p.kind in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
        )
        return min(positional, 2)
    except (ValueError, TypeError):
        return 0


def _to_mpf(val: Any, ops, prec: int) -> Any:
    """Convert backend value to mpmath mpf."""
    return ops["to_mpf"](val, prec)


def talbot(
    fn: Callable[[float], Any],
    time: Union[float, Iterable[float], np.ndarray],
    degree: int = 32,
    as_float: bool = True,
) -> Any:
    """
    Inverse Laplace transform via the Fixed Talbot method.

    A good choice for well-behaved, non-oscillatory transforms where
    moderate precision suffices.  Faster than GWR for simple functions.

    Parameters
    ----------
    fn : callable
        Laplace-domain function using mpmath.
    time : float, list, or ndarray
        Time value(s).
    degree : int
        Number of terms (default 32).
    as_float : bool
        If True (default), convert results to Python float.
    """
    convert = float if as_float else (lambda x: x)

    if not isinstance(time, Iterable):
        return convert(talbot_single(fn, time, degree))

    if isinstance(time, np.ndarray):
        return np.array(
            [convert(talbot_single(fn, float(t), degree)) for t in time],
            dtype=float if as_float else object,
        )

    return [convert(talbot_single(fn, t, degree)) for t in time]
