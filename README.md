# ilt-inversion

Numerical inverse Laplace transforms for Python.

## Why?

If you've ever needed to numerically invert a Laplace transform in Python, you've probably discovered that the standard tools either don't exist or fall over quietly when things get difficult.

**NumPy and SciPy don't have one.** There's no `numpy.inverse_laplace` or `scipy.special.ilt`. The standard scientific Python stack just doesn't cover this.

**SymPy tries, but it's symbolic.** `inverse_laplace_transform` attempts to find a closed-form `f(t)` via pattern matching and table lookups. That works for textbook problems - rational functions, simple exponentials - but hand it a Bessel-function ratio or anything defined by a numerical subroutine and it silently returns an unevaluated integral. You can't pass it a Python function that calls a numerical solver.

**mpmath has `invertlaplace`**, and it's actually decent. Three methods: de Hoog, Fixed Talbot, and Stehfest. But mpmath doesn't implement GWR, and its de Hoog and Talbot methods evaluate `F(s)` at *complex* values of `s`. That's a problem when your `F(s)` only works on the real axis - common in reservoir engineering where you're dealing with real-valued Bessel function ratios or feeding in results from a numerical ODE solver. GWR only needs `F(s)` at real, positive points.

**Stehfest** is the one everyone implements first, because it's simple and works at f64 precision. Handles smooth, monotonic transforms fine, but give it anything oscillatory or steep and it falls apart. The fixed-precision coefficients hit catastrophic cancellation at higher orders, capping you at ~6-8 significant figures no matter how many terms you throw at it.

**David Fulford's [`gwr_inversion`](https://github.com/petbox-dev/gwr)** is the package that got this right in Python. It implements GWR with arbitrary precision via mpmath, with optional gmpy2 acceleration (~10x speedup) - a clean, correct implementation that made GWR accessible to the Python community, and this library owes a debt to it. Version 2.0.0 (October 2026) adds python-flint (ARB) Bessel functions, a `workers` process pool, and a library of well-test solutions in the Laplace domain: radial, linear and fractured-well geometries with dual porosity, skin and wellbore storage, returning pressure, rate, Bourdet derivative and cumulative. If you want ready-made well-test models, use `gwr_inversion`.

## Where this library adds value

For straightforward `gwr(my_function, times)` calls, this library's Python path is essentially the same algorithm as Fulford's, with the same gmpy2 acceleration option. If that's your use case, either package will serve you well.

The differences are a compiled Rust path, in two places:

**Bulk inversion with simple callables.** When `M` is small enough that f64 precision suffices (M <= 7, covering ~15 significant figures), the Rust backend bypasses Python entirely and runs the full GWR algorithm in compiled code. That's about 7x faster per call than the gmpy2 Python path for a cheap callable (less when the callable itself is expensive, since its Python cost is the same on every path), which adds up if you're inverting across large parameter sweeps or Monte Carlo runs. For higher M, the Rust path can't help with a general Python callable (the f64 boundary at the Python-Rust interface becomes the bottleneck), and the Python path with gmpy2 is the right choice - same as Fulford's package.

**Bessel functions in compiled arbitrary precision.** For Laplace-domain functions built from modified Bessel functions (pressure transient analysis, radial diffusion, heat conduction), the Rust extension includes arbitrary-precision I_0, I_1, K_0, K_1 (exposed as `besseli_rust` / `besselk_rust`, which return floats; `besseli` / `besselk` use python-flint or mpmath and return mpmath values) with numerical safeguards that matter when you're evaluating these functions across a wide range of arguments:

- Dynamic guard bits on the K_0 power series to compensate for catastrophic cancellation between the `-(ln(x/2) + gamma) * I_0(x)` term and the harmonic series. At `x = 20`, that's ~18 digits of cancellation that silently corrupts the result if you compute at working precision.
- Exponentially-scaled forms (`I_ne(x) = I_n(x) * exp(-x)`, `K_ne(x) = K_n(x) * exp(x)`) that stay O(1) for all `x`, avoiding the overflow/underflow that hits f64 past `x > 700`.
- Optimal truncation of asymptotic expansions for large arguments, switching from the power series where the asymptotic error e^(-2x) falls below the working precision (so the switch point moves out as precision rises).

These Bessel functions only pay off fully when the transform itself is also in Rust, so that Bessel evaluation, the Gaver sums and Wynn-rho acceleration all run in compiled arbitrary precision without crossing into Python. [pyResToolbox](https://github.com/mwburgoyne/pyResToolbox) does this for the Van Everdingen-Hurst bounded radial solution (`simtools.influence_tables`), spread across all CPU threads. On a 126-point table (6 ReD x 21 tD, 4-core / 8-thread machine, best of 3) it gives the same accuracy as `gwr_inversion` 2.0.0's built-in bounded radial transform at every M tested (7 to 32) and runs 2.1-2.3x faster single-threaded (0.19 s against 0.40 s at M=7) and 2.5-2.8x faster with both on all cores (0.059 s against 0.15 s with `workers=8`). That pipeline lives in pyResToolbox, not here: from this package, a transform you write in Python runs at Python speed on every path.

## How GWR works

GWR (Valko & Abate, 2004) is a three-stage process. Evaluate `F(s)` at `2M` points along the real axis and combine with pre-computed factorial/binomial coefficients (the Gaver functionals). Apply Wynn-rho sequence acceleration to improve convergence. Extract the best estimate from odd levels of the acceleration tableau.

The factorial coefficients grow as `(2M)!` and the alternating sums produce catastrophic cancellation. With `M=32`, the coefficients reach ~10^67 while the result is O(1) - so you need at least 67 decimal digits of working precision, or the answer is pure noise. Standard f64 gives you 15.9 digits. Even quad precision only gets you 33.

The library automatically computes `ceil(2.1 * M)` decimal digits of working precision to guarantee enough significant figures survive.

## Performance tiers

The library picks the fastest backend that will give correct results:

| Tier | Backend | Speedup | When used |
|------|---------|---------|-----------|
| 1 | Rust | ~7x over gmpy2 | M <= 7 with a cheap callable and `workers=1` |
| 2 | gmpy2 | ~10x | M > 7, or `workers > 1` |
| 3 | mpmath | baseline | Always available |

The gmpy2 tier is the same acceleration that Fulford's `gwr_inversion` already provides - we inherited that design. The Rust tier is what's new. Its arbitrary-precision arithmetic is the pure-Rust `astro-float` crate, so building it needs only a Rust toolchain.

## Installation

```bash
pip install ilt-inversion
```

Binary wheels include the Rust extension for Linux, macOS, and Windows. If no binary wheel is available for your platform, the pure-Python fallback installs automatically and the Rust acceleration is simply absent.

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

Inverse Laplace transform via Gaver-Wynn-Rho. `fn` is your Laplace-domain function `F(s)` or `F(s, prec)`. `time` accepts a scalar, list, or numpy array (must be > 0). `M` controls accuracy - 6-12 for smooth transforms, 32 for general use, 768+ for hard oscillatory cases. `precin` overrides the automatic precision (`round(2.1 * M)` decimal digits). `backend` can be `'auto'`, `'rust'`, `'gmpy2'`, or `'mpmath'`. `workers` sets the number of parallel processes for array inputs (default 1 = sequential); see [Parallel evaluation](#parallel-evaluation-workers). Set `as_float=False` to get mpmath.mpf values at full precision.

**`talbot(fn, time, degree=32, as_float=True)`**

Inverse Laplace transform via Fixed Talbot. Good for well-behaved, non-oscillatory transforms where moderate precision suffices.

**`besselk(n, x)` / `besseli(n, x)`**

Modified Bessel functions K_n(x) and I_n(x) with automatic python-flint/ARB acceleration when available.

## Parallel evaluation (`workers`)

`gwr(fn, times, workers=N)` inverts the time points in parallel on the Python path (gmpy2 or mpmath backend). Each time point is an independent inversion, so they are handed to a `ProcessPoolExecutor` one point per task. How it behaves:

- The pool size is `min(workers, len(times), os.cpu_count())`, so asking for more workers than points or CPUs costs nothing extra.
- A new pool starts on every call and each task rebuilds the GWR coefficients, so the overhead is a process start-up per worker plus one coefficient set per time point. It pays off when `fn` is expensive (Bessel-function transforms, high `M`) and there are many time points; for a cheap `fn` or a handful of points, `workers=1` is faster.
- `fn` must be picklable: a module-level function, or an instance of a module-level class with `__call__`. Lambdas and closures fail.
- `workers > 1` always takes the Python path. The Rust fast path (used when `backend` is `'auto'` or `'rust'` and the precision fits in f64, i.e. `M <= 7`) runs single-threaded, because it calls your Python `fn` and holds the GIL; at `M <= 7` with a cheap `fn` it is usually faster than any worker count.
- Results match the sequential Python path (`workers=1, backend='gmpy2'` or `'mpmath'`); only the scheduling changes. At `M <= 7` the default `workers=1` takes the Rust path, which works from f64 values of `fn` and can differ in the last digits.

## Acknowledgements

The GWR algorithm implementation builds on David S. Fulford's [`gwr_inversion`](https://github.com/petbox-dev/gwr) package, which made a clean and correct GWR available to the Python community.

## References

- Valko, P.P. & Abate, J. (2004), "Comparison of Sequence Accelerators for the Gaver Method of Numerical Laplace Transform Inversion", *Computers and Mathematics with Applications* 48(3): 629-636.
- Abramowitz, M. & Stegun, I.A. (1964), *Handbook of Mathematical Functions*, National Bureau of Standards.
- Van Everdingen, A.F. & Hurst, W. (1949), "The Application of the Laplace Transformation to Flow Problems in Reservoirs", SPE-949305-G.

## Licence

GPL-3.0-or-later
