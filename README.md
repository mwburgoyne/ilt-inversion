# ilt-inversion

Fast, accurate numerical inverse Laplace transforms for Python.

## Why this library?

### The problem with standard tools

Numerical inverse Laplace transforms (ILTs) are deceptively hard. The
transform involves evaluating `f(t) = L^{-1}{F(s)}` from samples of `F(s)`
in the complex plane — a problem that is inherently ill-conditioned.

**NumPy/SciPy have no ILT.** There is no `numpy.inverse_laplace` or
`scipy.special.ilt`. The standard scientific Python stack simply does not
provide this capability, leaving users to implement their own or find
third-party packages.

**SymPy's `inverse_laplace_transform`** is *symbolic*, not numerical. It
attempts to find a closed-form expression for `f(t)` using pattern matching,
table lookups, and the Bromwich integral. This works for textbook transforms
(rational functions, simple exponentials) but fails — often silently returning
an unevaluated integral — for the kinds of transforms that arise in practice:
Bessel-function ratios, branch cuts, transforms defined by numerical
subroutines rather than symbolic expressions. You cannot pass SymPy a Python
function `F(s)` that calls a numerical solver and expect it to invert it.
SymPy also carries significant import overhead (~1-2 seconds) and its CAS
engine is not designed for high-throughput numerical work.

**mpmath's `invertlaplace`** is numerical and provides three methods: the
de Hoog, Fixed Talbot, and Stehfest algorithms. It is a solid general-purpose
tool and this library wraps its Talbot method as a convenience. However,
mpmath does not implement GWR — the most accurate method for difficult
transforms. Its Stehfest implementation shares the fundamental precision
ceiling of all fixed-order Stehfest codes. The de Hoog and Talbot methods
evaluate `F(s)` at *complex* values of `s`, which is problematic when `F(s)`
is only defined (or only easily computed) on the real axis — a common
situation in reservoir engineering where `F(s)` involves real-valued Bessel
function ratios or comes from a numerical ODE solver. GWR evaluates `F(s)`
only at real, positive points, making it compatible with any callable that
accepts a real number.

**The Stehfest algorithm** (the most commonly implemented ILT in engineering
codes) uses fixed 64-bit floating point. It works for smooth, monotonic
transforms but fails badly on oscillatory or steep-gradient functions. The
fixed-precision coefficients suffer from catastrophic cancellation at higher
orders, capping accuracy at ~6-8 significant figures regardless of how many
terms you use.

**The [`gwr_inversion`](https://github.com/petbox-dev/gwr) package** by David
S. Fulford provides an excellent pure-Python implementation of the
Gaver-Wynn-Rho algorithm with arbitrary precision via mpmath and optional
gmpy2 acceleration. It is the basis for the GWR algorithm used in this
library, and we gratefully acknowledge Fulford's work in making a clean,
correct GWR implementation available to the Python community. However,
`gwr_inversion` is pure Python throughout — for applications that invert
thousands of transforms (e.g., pressure transient analysis, aquifer
modelling), the per-call overhead of Python arbitrary-precision arithmetic
becomes the bottleneck.

### What this library does differently

`ilt-inversion` builds on the same GWR foundation as `gwr_inversion`, adding:

- **Rust/MPFR acceleration** (~15-70x speedup) for the GWR coefficient
  computation and, where the Laplace-domain function is implemented in Rust,
  the entire inversion pipeline
- **MPFR-precision Bessel functions** (I_0, I_1, K_0, K_1) with dynamic guard
  bits and exponential scaling — critical for pressure transient and radial
  diffusion problems where Bessel-function cancellation silently destroys
  precision
- **Automatic backend selection** across three performance tiers (Rust/MPFR,
  gmpy2, mpmath) with correct precision routing so the user doesn't need to
  know which tier is active
- **Wynn-rho breakdown detection** that gracefully handles convergence
  singularities instead of propagating NaN/Inf

## The GWR algorithm

GWR (Valko & Abate, 2004) is a three-stage process:

1. **Gaver functionals**: Evaluate `F(s)` at `2M` points along the real axis
   and combine with pre-computed factorial/binomial coefficients
2. **Wynn-rho acceleration**: Apply sequence acceleration to the Gaver
   functionals, dramatically improving convergence
3. **Best estimate extraction**: Select the most converged value from odd
   levels of the acceleration tableau

The catch: the factorial coefficients grow as `(2M)!` and the alternating
sums in the Gaver functionals produce catastrophic cancellation. With `M=32`,
the coefficients reach ~10^67 while the result is O(1) — meaning you need
**at least 67 decimal digits** of precision throughout the calculation, or the
answer is pure noise.

This is why f64 (15.9 digits) and even quad precision (33 digits) are
completely inadequate for high-M GWR. The library automatically computes
`ceil(2.1 * M)` decimal digits of working precision to guarantee enough
significant figures survive the cancellation.

## Cancellation traps and numerical safeguards

### 1. GWR coefficient cancellation

The Gaver functional for order `n` computes:

```
G_n = tau * c_n * sum_{i=0}^{n} (-1)^i * C(n,i) * F(s_{n+i})
```

where `c_n = (2n)! / (n * ((n-1)!)^2)`. For `n=32`, `c_n ~ 10^66`. The
alternating sum then cancels most of these digits. **If your arithmetic
doesn't carry enough precision, the result is dominated by rounding error.**

This library pre-computes all coefficients at the required MPFR precision
(GMP/MPFR via the `rug` crate in Rust, or mpmath/gmpy2 in Python) and
performs the full GWR algorithm in that extended precision.

### 2. Wynn-rho sequence acceleration breakdown

The Wynn-rho step computes successive differences `G_0[n+1] - G_0[n]`. When
two adjacent Gaver functionals are nearly equal (convergence), the difference
approaches zero, and dividing by it amplifies noise. The library detects
zero-differences and breaks out of the acceleration loop early, returning
the best estimate obtained before breakdown.

### 3. K_0 Bessel series catastrophic cancellation

For Laplace-domain functions involving modified Bessel functions (common in
radial diffusion, heat conduction, and pressure transient analysis), the
power series for `K_0(x)` is:

```
K_0(x) = -(ln(x/2) + gamma) * I_0(x) + sum_{k=1}^inf H_k * (x^2/4)^k / (k!)^2
```

For large `x`, the first term `-(ln(x/2) + gamma) * I_0(x)` and the series
sum are both enormous (growing as `e^x`) but nearly cancel. At `x = 20`, this
means ~18 decimal digits of cancellation — the two terms agree to 18 digits
and the answer lives in the remaining digits.

**Most implementations ignore this.** They compute `K_0` at working precision
and get garbage for moderate-to-large arguments, or they switch to asymptotic
expansions too early (which have their own accuracy limits).

This library adds **dynamic guard bits** proportional to the argument:

```
guard_bits = ceil(x * log2(e)) + 20
```

This exactly compensates for the `e^x` cancellation factor. The series is
evaluated at `prec + guard_bits` precision, then rounded back to the
requested precision after the cancellation has occurred — preserving all
significant digits.

### 4. Bessel function overflow for large arguments

`I_0(x)` grows as `e^x / sqrt(2*pi*x)` and `K_0(x)` decays as
`e^{-x} * sqrt(pi/(2x))`. For `x > 700`, these overflow/underflow f64.
Even with arbitrary precision, computing them directly and then combining
creates unnecessary dynamic range.

The library uses **exponentially-scaled Bessel functions** throughout:

```
I_ne(x) = I_n(x) * exp(-x)    # O(1) for all x
K_ne(x) = K_n(x) * exp(x)     # O(1) for all x
```

The exponential factors cancel algebraically in the ratios that appear in
physical Laplace-domain solutions (e.g., the Van Everdingen-Hurst radial flow
formula), so the scaled forms avoid overflow entirely.

### 5. Asymptotic / series transition

For large arguments (`x > 25`), the power series converges too slowly and
suffers the cancellation described above. The library switches to asymptotic
expansions with **optimal truncation** — summing terms until they start
growing (the point of minimum error in a divergent asymptotic series), rather
than using a fixed number of terms.

## Performance tiers

The library auto-detects available backends and uses the fastest appropriate
one:

| Tier | Backend | Speedup | When used |
|------|---------|---------|-----------|
| 1 | **Rust/MPFR** | ~15-70x | M <= 7 (f64 callable sufficient) or internal Bessel evaluation |
| 2 | **gmpy2** | ~10x | M > 7; GMP-backed factorial/binomial arithmetic |
| 3 | **mpmath** | baseline | Always available; pure Python arbitrary precision |

**Why the Rust tier has an M limit:** The Rust GWR engine runs the full
algorithm in MPFR arbitrary precision. However, it calls your Python function
`F(s)` across the Rust-Python boundary, receiving the result as a 64-bit
float (~15.9 digits). For `M > 7`, GWR needs more than 15 digits in the
`F(s)` values, so the f64 bottleneck at the boundary would corrupt the
result. The library automatically detects this and routes high-M calls
through the Python path where mpmath carries full precision end-to-end.

For domain-specific applications where the Laplace-domain function is
implemented *entirely in Rust* (e.g., Bessel-based radial flow solutions),
the full 70x speedup applies at any M.

## Installation

```bash
pip install ilt-inversion
```

### Optional accelerators

```bash
pip install gmpy2        # ~10x faster GWR for high-M
pip install python-flint  # ~15x faster Bessel functions via ARB
```

Binary wheels include the Rust/MPFR extension for Linux, macOS, and Windows.
If no binary wheel is available for your platform, a pure-Python fallback
installs automatically (Rust acceleration will simply be absent).

## Quick start

```python
from ilt import gwr, talbot

# L{e^(-t)} = 1/(s+1)
def F(s):
    return 1 / (s + 1)

# Single time point
result = gwr(F, 1.0)           # 0.36787944... = e^(-1)

# Multiple time points
results = gwr(F, [0.1, 1.0, 10.0])

# Higher accuracy (more terms, more precision)
results = gwr(F, [1.0], M=64)

# Fixed Talbot (faster for well-behaved transforms)
result = talbot(F, 1.0)

# Parallel evaluation
results = gwr(F, times, workers=4)  # fn must be picklable
```

## API

### `gwr(fn, time, M=32, precin=None, backend='auto', workers=1, as_float=True)`

Inverse Laplace transform via Gaver-Wynn-Rho.

- **fn**: Laplace-domain function `F(s)` or `F(s, prec)`
- **time**: scalar, list, or numpy array of time values (> 0)
- **M**: number of Gaver functional terms (default 32). Typical: 6-12 for
  smooth transforms, 32 for general use, 768+ for hard oscillatory cases
- **precin**: decimal digits of precision (default: `round(2.1 * M)`)
- **backend**: `'auto'`, `'rust'`, `'gmpy2'`, or `'mpmath'`
- **workers**: parallel workers for array inputs (> 1 needs picklable fn)
- **as_float**: if True, return Python floats; if False, return mpmath.mpf

### `talbot(fn, time, degree=32, as_float=True)`

Inverse Laplace transform via Fixed Talbot method. Good for well-behaved,
non-oscillatory transforms where moderate precision suffices.

### `besselk(n, x)` / `besseli(n, x)`

Modified Bessel functions K_n(x) and I_n(x) with automatic acceleration
via python-flint/ARB when available. Useful for constructing Laplace-domain
functions in pressure transient analysis and heat conduction problems.

## References

- Valko, P.P. & Abate, J. (2004), "Comparison of Sequence Accelerators for
  the Gaver Method of Numerical Laplace Transform Inversion", *Computers and
  Mathematics with Applications* 48(3): 629-636.
- Abramowitz, M. & Stegun, I.A. (1964), *Handbook of Mathematical Functions*,
  National Bureau of Standards. (Bessel function series and asymptotics)
- Van Everdingen, A.F. & Hurst, W. (1949), "The Application of the Laplace
  Transformation to Flow Problems in Reservoirs", *SPE-949305-G*.

## License

GPL-3.0-or-later
