"""
Backend abstraction: mpmath (always available), gmpy2 (optional ~10x speedup).

All arithmetic during the GWR algorithm runs through the ops dict returned by
get_ops().  When gmpy2 is available, factorials, binomials, and log(2) use
GMP/MPFR instead of mpmath, which is dramatically faster for the integer
combinatorics that dominate the Gaver functional computation.
"""

from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Tuple

from mpmath import mp

# ---------------------------------------------------------------------------
# gmpy2 detection
# ---------------------------------------------------------------------------
try:
    import gmpy2
    from gmpy2 import mpfr
    HAS_GMPY2 = True
except ImportError:
    gmpy2 = None  # type: ignore[assignment]
    mpfr = None   # type: ignore[assignment,misc]
    HAS_GMPY2 = False


def _dec_to_bits(dec_digits: int) -> int:
    """Convert decimal digits of precision to binary bits (with guard bits)."""
    return int(dec_digits * 3.3219280948873626) + 10


@contextmanager
def precision_context(precin: int) -> Generator[None, None, None]:
    """Set gmpy2 context precision (no-op when gmpy2 is absent)."""
    if not HAS_GMPY2:
        yield
        return
    with gmpy2.context(gmpy2.get_context(), precision=_dec_to_bits(precin)):
        yield


def get_backend(name: str = "auto") -> str:
    if name in ("auto", "rust"):
        return "gmpy2" if HAS_GMPY2 else "mpmath"
    if name == "gmpy2" and not HAS_GMPY2:
        raise ImportError("gmpy2 is not installed.  pip install gmpy2")
    return name


# ── mpmath ops ─────────────────────────────────────────────────────────────

def _mp_log2(_p: int) -> Any:
    return mp.log(2)

def _mp_mpf(val: Any, _p: int) -> Any:
    return mp.mpf(val)

def _mp_fac(n: int, _p: int) -> Any:
    return mp.fac(n)

def _mp_binomial(n: int, i: int, _p: int) -> Any:
    return mp.binomial(n, i)

def _mp_to_mpf(val: Any, _p: int) -> Any:
    return val

def _mp_from_mpf(val: Any, _p: int) -> Any:
    return val

def _mp_to_float(val: Any) -> float:
    return float(val)


# ── gmpy2 ops ──────────────────────────────────────────────────────────────

def _gm_log2(_p: int) -> Any:
    return gmpy2.log(mpfr(2))

def _gm_mpf(val: Any, _p: int) -> Any:
    return mpfr(str(val))

def _gm_fac(n: int, _p: int) -> Any:
    return mpfr(gmpy2.fac(n))

def _gm_binomial(n: int, i: int, _p: int) -> Any:
    return mpfr(gmpy2.comb(n, i))

def _gm_to_mpf(val: Any, precin: int) -> Any:
    """gmpy2 mpfr -> mpmath mpf (for passing to user's Laplace-domain fn)."""
    saved = mp.dps
    mp.dps = precin
    result = mp.mpf(str(val))
    mp.dps = saved
    return result

def _gm_from_mpf(val: Any, _p: int) -> Any:
    """mpmath mpf -> gmpy2 mpfr."""
    return mpfr(str(val))

def _gm_to_float(val: Any) -> float:
    return float(val)


# ── dispatch ───────────────────────────────────────────────────────────────

OpsDict = Dict[str, Callable[..., Any]]

_MPMATH_OPS: OpsDict = {
    "log2":     _mp_log2,
    "mpf":      _mp_mpf,
    "fac":      _mp_fac,
    "binomial": _mp_binomial,
    "to_mpf":   _mp_to_mpf,
    "from_mpf": _mp_from_mpf,
    "to_float": _mp_to_float,
}

_GMPY2_OPS: OpsDict = {
    "log2":     _gm_log2,
    "mpf":      _gm_mpf,
    "fac":      _gm_fac,
    "binomial": _gm_binomial,
    "to_mpf":   _gm_to_mpf,
    "from_mpf": _gm_from_mpf,
    "to_float": _gm_to_float,
}

_BACKENDS: Dict[str, OpsDict] = {
    "mpmath": _MPMATH_OPS,
    "gmpy2":  _GMPY2_OPS,
}


def get_ops(backend_name: str) -> Tuple[OpsDict, str]:
    name = get_backend(backend_name)
    return _BACKENDS[name], name
