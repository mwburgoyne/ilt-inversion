"""Bessel functions and GWR on a Bessel transform, checked against mpmath.

Regression cover for the defects fixed in 0.1.6: ARB balls taken at their
midpoint with no correct digits (K_0 near x = 23.5 at 15 digits), and the
Rust asymptotic I expansion missing its (-1)^k, with f64 pi, a 50-digit
Euler gamma and a fixed x > 25 switch capping precision.
"""
import numpy as np
import pytest
from mpmath import mp

import ilt
from ilt import besseli, besselk

XS = [0.01, 0.5, 1.0, 5.0, 10.0, 16.65, 23.5482, 25.5, 28.84, 31.15, 40.0, 60.0, 100.0, 300.0]


def _rel(a, b):
    return abs(a / b - 1)


@pytest.mark.parametrize("dps", [15, 30, 70])
@pytest.mark.parametrize("fn, ref", [(besseli, mp.besseli), (besselk, mp.besselk)])
@pytest.mark.parametrize("n", [0, 1])
def test_python_bessel_matches_mpmath(dps, fn, ref, n):
    saved = mp.dps
    mp.dps = dps
    try:
        tol = mp.mpf(10) ** (-(dps - 2))
        for x in XS:
            X = mp.mpf(x)
            assert _rel(fn(n, X), ref(n, X)) < tol, f"n={n} x={x} dps={dps}"
    finally:
        mp.dps = saved


@pytest.mark.skipif(not ilt.HAS_RUST, reason="Rust extension not built")
@pytest.mark.parametrize("prec", [16, 34, 70])
@pytest.mark.parametrize("n", [0, 1])
def test_rust_bessel_matches_mpmath(prec, n):
    from ilt import besseli_rust, besselk_rust
    mp.dps = 40
    for x in XS:
        if x > 700:
            continue
        assert _rel(besseli_rust(n, x, prec), float(mp.besseli(n, x))) < 4e-16, f"I{n}({x})"
        assert _rel(besselk_rust(n, x, prec), float(mp.besselk(n, x))) < 4e-16, f"K{n}({x})"
    mp.dps = 15


def _veh(I, K, red):
    """Van Everdingen-Hurst constant-terminal-rate, bounded no-flow transform."""
    def f(s):
        x = mp.sqrt(s)
        i1, k1 = I(1, x * red), K(1, x * red)
        return (k1 * I(0, x) + i1 * K(0, x)) / (s ** 1.5 * (i1 * K(1, x) - k1 * I(1, x)))
    return f


# At M=7 (15 digits) GWR amplifies last-bit differences to ~1e-9: two mpmath
# runs on different arithmetic backends differ by 3e-9. Before 0.1.6 this was 2e4.
@pytest.mark.parametrize("M, tol", [(7, 1e-8), (12, 1e-9)])
def test_gwr_with_ilt_bessel_matches_mpmath_bessel(M, tol):
    """At ReD 1.5, tD 0.01 the GWR abscissae put x where ARB at 15 digits lost K_0."""
    t = [0.01, 0.03, 0.1, 1.0, 10.0]
    a = np.array(ilt.gwr(_veh(besseli, besselk, 1.5), t, M))
    b = np.array(ilt.gwr(_veh(mp.besseli, mp.besselk, 1.5), t, M))
    assert np.max(np.abs(a / b - 1)) < tol
