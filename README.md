# ilt-inversion

Numerical inverse Laplace transforms for Python, done properly.

## Why?

If you've ever needed to numerically invert a Laplace transform in Python, you've probably discovered that the standard tools either don't exist or fall over quietly when things get difficult.

**NumPy and SciPy don't have one.** There's no `numpy.inverse_laplace` or `scipy.special.ilt`. The standard scientific Python stack just doesn't cover this, which is a bit surprising given how often Laplace-domain solutions appear in engineering and physics.

**SymPy tries, but it's symbolic.** `inverse_laplace_transform` attempts to find a closed-form `f(t)` via pattern matching and table lookups. That works for textbook problems - rational functions, simple exponentials - but hand it a Bessel-function ratio or anything defined by a numerical subroutine and it silently returns an unevaluated integral. You also can't pass it a Python function that calls a numerical solver. It's a computer algebra system, not a numerical tool, and it carries ~1-2 seconds of import overhead besides.

**mpmath has `invertlaplace`**, and it's actually decent. Three methods: de Hoog, Fixed Talbot, and Stehfest. This library wraps the Talbot method as a convenience. But mpmath doesn't implement GWR (the most accurate method for difficult transforms), and its de Hoog and Talbot methods evaluate `F(s)` at *complex* values of `s`. That's a problem when your `F(s)` only works on the real axis - which is common in reservoir engineering where you're dealing with real-valued Bessel function ratios or feeding in results from a numerical ODE solver. GWR only needs `F(s)` at real, positive points.

**Stehfest** is the one everyone implements first, because it's simple and works at f64 precision. It handles smooth, monotonic transforms fine, but give it anything oscillatory or steep and it falls apart. The fixed-precision coefficients hit catastrophic cancellation at higher orders, capping you at ~6-8 significant figures no matter how many terms you throw at it.

**David Fulford's [`gwr_inversion`](https://github.com/petbox-dev/gwr)** is the package that got this right in Python. It implements the Gaver-Wynn-Rho algorithm with arbitrary precision via mpmath and optional gmpy2 acceleration - a clean, correct implementation that made GWR accessible to the Python community. This library owes a debt to that work. The limitation is that `gwr_inversion` is pure Python throughout, and when you're inverting thousands of transforms (pressure transient analysis, aquifer modelling), the per-call overhead of Python arbitrary-precision arithmetic becomes the bottleneck.

## What this library does

`ilt-inversion` builds on the same GWR foundation, adding Rust/MPFR acceleration (~15-70x speedup), MPFR-precision Bessel functions with numerical safeguards that most implementations skip, and automatic backend selection so you don't need to think about which precision tier is active.

## How GWR works

GWR (Valko & Abate, 2004) is a three-stage process. Evaluate `F(s)` at `2M` points along the real axis and combine with pre-computed factorial/binomial coefficients (the Gaver functionals). Apply Wynn-rho sequence acceleration to improve convergence. Extract the best estimate from odd levels of the acceleration tableau.

The catch is that the factorial coefficients grow as `(2M)!` and the alternating sums produce catastrophic cancellation. With `M=32`, the coefficients reach ~10^67 while the result is O(1) - so you need at least 67 decimal digits of working precision, or the answer is pure noise. Standard f64 gives you 15.9 digits. Even quad precision only gets you 33. Neither is close to enough.

The library automatically computes `ceil(2.1 * M)` decimal digits of working precision to guarantee enough significant figures survive.

## Where other implementations lose precision

There are five places this can go wrong, and most implementations only handle the first one (if that).

**GWR coefficient cancellation.** The Gaver functional for order `n` computes `G_n = tau * c_n * sum (-1)^i * C(n,i) * F(s_{n+i})`, where `c_n = (2n)! / (n * ((n-1)!)^2)`. For `n=32`, `c_n` is ~10^66. The alternating sum cancels most of those digits. If your arithmetic doesn't carry enough precision, the result is rounding error. This library pre-computes all coefficients at the required MPFR precision (GMP/MPFR via the `rug` crate in Rust, or mpmath/gmpy2 in Python) and runs the full algorithm in that extended precision.

**Wynn-rho breakdown.** The acceleration step computes successive differences `G_0[n+1] - G_0[n]`. As adjacent functionals converge, the difference approaches zero and dividing by it amplifies noise. The library detects zero-differences and breaks out early, returning the best estimate obtained before breakdown rather than propagating NaN or Inf.

**K_0 Bessel series cancellation.** For Laplace-domain functions involving modified Bessel functions (radial diffusion, heat conduction, pressure transient analysis), the power series for `K_0(x)` is `-(ln(x/2) + gamma) * I_0(x) + sum H_k * (x^2/4)^k / (k!)^2`. For large `x`, both the first term and the series sum grow as `e^x` but nearly cancel. At `x = 20`, that's ~18 digits of cancellation - the two terms agree to 18 digits and the answer lives in whatever is left.

Most implementations just compute `K_0` at working precision and get garbage for moderate-to-large arguments. This library adds dynamic guard bits proportional to the argument: `guard_bits = ceil(x * log2(e)) + 20`. That exactly compensates for the `e^x` cancellation factor. The series runs at `prec + guard_bits`, then rounds back to the requested precision after the cancellation has happened.

**Bessel overflow.** `I_0(x)` grows as `e^x / sqrt(2*pi*x)` and `K_0(x)` decays as `e^{-x} * sqrt(pi/(2x))`. Past `x > 700` these overflow or underflow f64. The library uses exponentially-scaled Bessel functions throughout - `I_ne(x) = I_n(x) * exp(-x)` and `K_ne(x) = K_n(x) * exp(x)` - which stay O(1) for all `x`. The exponential factors cancel algebraically in the ratios that appear in physical Laplace-domain solutions (Van Everdingen-Hurst, for instance), so the scaled forms avoid overflow entirely.

**Series/asymptotic transition.** For large arguments (`x > 25`), the power series converges too slowly and suffers the cancellation described above. The library switches to asymptotic expansions with optimal truncation - summing terms until they start growing (the point of minimum error in a divergent asymptotic series) rather than using a fixed number of terms.

## Performance tiers

The library picks the fastest backend that will give correct results:

| Tier | Backend | Speedup | When used |
|------|---------|---------|-----------|
| 1 | Rust/MPFR | ~15-70x | M <= 7 (f64 callable sufficient), or internal Bessel evaluation |
| 2 | gmpy2 | ~10x | M > 7, GMP-backed factorial/binomial arithmetic |
| 3 | mpmath | baseline | Always available, pure Python arbitrary precision |

The gmpy2 tier is the same acceleration that Fulford's `gwr_inversion` already provides - we inherited that design. What's new is the Rust/MPFR tier on top.

The Rust tier has an M limit because of a practical constraint. The Rust GWR engine runs in full MPFR precision internally, but it calls your Python function `F(s)` across the Rust-Python boundary and gets back a 64-bit float (~15.9 digits). For `M > 7`, GWR needs more than 15 digits in the `F(s)` values, so the f64 bottleneck at the boundary corrupts the result. The library detects this and routes high-M calls through the Python path where mpmath carries full precision end-to-end.

For domain-specific applications where the Laplace-domain function is implemented entirely in Rust (Bessel-based radial flow solutions, for example), the full 70x speedup applies at any M.

## Installation

```bash
pip install ilt-inversion
```

Binary wheels include the Rust/MPFR extension for Linux, macOS, and Windows. If no binary wheel is available for your platform, the pure-Python fallback installs automatically and the Rust acceleration is simply absent.

Optional accelerators for the Python path:

```bash
pip install gmpy2         # ~10x faster GWR for high-M
pip install python-flint  # ~15x faster Bessel functions via ARB
```

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

# Higher accuracy
results = gwr(F, [1.0], M=64)

# Fixed Talbot - faster for well-behaved transforms
result = talbot(F, 1.0)

# Parallel evaluation (fn must be picklable - module-level, not a lambda)
results = gwr(F, times, workers=4)
```

## API

**`gwr(fn, time, M=32, precin=None, backend='auto', workers=1, as_float=True)`**

Inverse Laplace transform via Gaver-Wynn-Rho. `fn` is your Laplace-domain function `F(s)` or `F(s, prec)`. `time` accepts a scalar, list, or numpy array (must be > 0). `M` controls accuracy - 6-12 for smooth transforms, 32 for general use, 768+ for hard oscillatory cases. `precin` overrides the automatic precision (`round(2.1 * M)` decimal digits). `backend` can be `'auto'`, `'rust'`, `'gmpy2'`, or `'mpmath'`. Set `as_float=False` to get mpmath.mpf values at full precision.

**`talbot(fn, time, degree=32, as_float=True)`**

Inverse Laplace transform via Fixed Talbot. Good for well-behaved, non-oscillatory transforms where moderate precision suffices.

**`besselk(n, x)` / `besseli(n, x)`**

Modified Bessel functions K_n(x) and I_n(x) with automatic python-flint/ARB acceleration when available. Useful for building Laplace-domain functions in pressure transient analysis and heat conduction problems.

## Acknowledgements

The GWR algorithm implementation builds on David S. Fulford's [`gwr_inversion`](https://github.com/petbox-dev/gwr) package, which made a clean and correct GWR available to the Python community.

## References

- Valko, P.P. & Abate, J. (2004), "Comparison of Sequence Accelerators for the Gaver Method of Numerical Laplace Transform Inversion", *Computers and Mathematics with Applications* 48(3): 629-636.
- Abramowitz, M. & Stegun, I.A. (1964), *Handbook of Mathematical Functions*, National Bureau of Standards.
- Van Everdingen, A.F. & Hurst, W. (1949), "The Application of the Laplace Transformation to Flow Problems in Reservoirs", SPE-949305-G.

## Licence

GPL-3.0-or-later
