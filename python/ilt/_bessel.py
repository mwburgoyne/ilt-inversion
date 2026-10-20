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


# ARB returns a ball, and at a fixed working precision the ball can hold no
# correct digits (K_0 near x = 23.5 at 60 bits is [+/- 6.4e-7] around a true
# 1.5e-11). The midpoint must not be taken blindly: each call raises the
# precision until the ball is accurate to the mpmath working precision, and
# falls back to mpmath if it never is. A short ball is retried at its measured
# shortfall plus a margin. No guard is added up front: ARB chooses its algorithm
# by precision, and at large x extra bits move K_n off its fast, accurate
# asymptotic expansion onto a slower series that is less accurate (x = 100:
# 113 good bits at 118, -5 at 236).
_FLINT_GUARD_BITS = 16
_FLINT_MAX_ATTEMPTS = 6


def _target_bits() -> int:
    return int(mp.dps * 3.3219280948873626) + 10


def _arb_to_mpf(x):
    """Convert arb -> mpmath mpf via internal _mpf_ tuple (zero-copy)."""
    result = mp.mpf.__new__(mp.mpf)
    result._mpf_ = x._mpf_
    return result


def _mpf_to_arb(x):
    """Convert mpmath mpf -> arb via string representation."""
    return arb(mp.nstr(x, int(mp.dps * 1.05) + 5, strip_zeros=False))


def _flint_bessel(kind: str, n: int, x):
    """Evaluate an ARB Bessel function to the mpmath working precision, or None."""
    target = _target_bits()
    prec = target + _FLINT_GUARD_BITS
    saved = _flint_ctx.prec
    try:
        for _ in range(_FLINT_MAX_ATTEMPTS):
            _flint_ctx.prec = prec
            ball = getattr(_mpf_to_arb(x), kind)(n)
            accuracy = ball.rel_accuracy_bits()
            if accuracy >= target:
                return _arb_to_mpf(ball.mid())
            prec += max(target - accuracy, 0) + 2 * _FLINT_GUARD_BITS
    finally:
        _flint_ctx.prec = saved
    return None


def besselk(n: int, x):
    """Modified Bessel function K_n(x).  Uses ARB if available."""
    if HAS_FLINT:
        result = _flint_bessel("bessel_k", n, x)
        if result is not None:
            return result
    return mp.besselk(n, x)


def besseli(n: int, x):
    """Modified Bessel function I_n(x).  Uses ARB if available."""
    if HAS_FLINT:
        result = _flint_bessel("bessel_i", n, x)
        if result is not None:
            return result
    return mp.besseli(n, x)
