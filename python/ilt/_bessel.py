"""
Accelerated Bessel functions using python-flint (ARB) when available.

ARB provides ~15x speedup over mpmath for modified Bessel functions K_n and I_n.
Falls back transparently to mpmath when python-flint is not installed.

When the Rust native extension is available, besseli_rust / besselk_rust
provide even faster f64 evaluation at arbitrary internal precision.
"""

from mpmath import mp

try:
    from flint import arb, ctx as _flint_ctx
    HAS_FLINT = True
except ImportError:
    arb = None         # type: ignore[assignment,misc]
    _flint_ctx = None  # type: ignore[assignment]
    HAS_FLINT = False


def _sync_flint_prec() -> None:
    """Match flint context precision to current mp.dps."""
    _flint_ctx.prec = int(mp.dps * 3.3219280948873626) + 10


def _arb_to_mpf(x):
    """Convert arb -> mpmath mpf via internal _mpf_ tuple (zero-copy)."""
    result = mp.mpf.__new__(mp.mpf)
    result._mpf_ = x._mpf_
    return result


def _mpf_to_arb(x):
    """Convert mpmath mpf -> arb via string representation."""
    return arb(mp.nstr(x, int(mp.dps * 1.05) + 5, strip_zeros=False))


def besselk(n: int, x):
    """Modified Bessel function K_n(x).  Uses ARB if available."""
    if not HAS_FLINT:
        return mp.besselk(n, x)
    _sync_flint_prec()
    return _arb_to_mpf(_mpf_to_arb(x).bessel_k(n))


def besseli(n: int, x):
    """Modified Bessel function I_n(x).  Uses ARB if available."""
    if not HAS_FLINT:
        return mp.besseli(n, x)
    _sync_flint_prec()
    return _arb_to_mpf(_mpf_to_arb(x).bessel_i(n))
