"""
Gaver-Wynn-Rho (GWR) algorithm for numerical inverse Laplace transforms.

Reference:
    Valko & Abate (2004), "Comparison of Sequence Accelerators for the
    Gaver Method of Numerical Laplace Transform Inversion",
    Computers and Mathematics with Application 48(3): 629-636.

Optimizations over the reference implementation:
  - Pre-compute all factorial / binomial coefficients once per (M, prec)
    and reuse across time points.
  - Avoid inspect.signature() on every call — detect arity once.
  - In-place Wynn-rho tableau update.
"""

from typing import Any, Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Coefficient pre-computation
# ---------------------------------------------------------------------------

def precompute_coefficients(
    M: int,
    prec: int,
    ops: Dict[str, Callable[..., Any]],
) -> Tuple[List[Any], List[List[Any]]]:
    """
    Pre-compute the Gaver functional coefficients for all n = 1..M.

    Returns
    -------
    gaver_coeffs : list of length M
        gaver_coeffs[n-1] = tau_multiplier = fac(2n) / (n * fac(n-1)^2)
        (the tau factor is applied per-time-point)
    binom_table : list of lists
        binom_table[n-1][i] = C(n, i) for i = 0..n
    """
    gaver_coeffs = []
    binom_table = []
    for n in range(1, M + 1):
        n_fac = ops["fac"](n - 1, prec)
        coeff = ops["fac"](2 * n, prec) / (n * n_fac * n_fac)
        gaver_coeffs.append(coeff)

        row = [ops["binomial"](n, i, prec) for i in range(n + 1)]
        binom_table.append(row)

    return gaver_coeffs, binom_table


# ---------------------------------------------------------------------------
# Single time-point GWR
# ---------------------------------------------------------------------------

def gwr_single(
    fn: Callable[..., Any],
    time: float,
    M: int,
    prec: int,
    ops: Dict[str, Callable[..., Any]],
    coeffs: Tuple[List[Any], List[List[Any]]],
    fn_arity: int = 0,
) -> Any:
    """
    Evaluate the inverse Laplace transform at a single time point.

    Parameters
    ----------
    fn : callable
        Laplace-domain function.  Signature fn(s) or fn(s, prec).
    time : float
        Time value (must be > 0).
    M : int
        Number of Gaver functional terms.
    prec : int
        Decimal digits of precision.
    ops : dict
        Backend arithmetic operations.
    coeffs : tuple
        Pre-computed (gaver_coeffs, binom_table) from precompute_coefficients.
    fn_arity : int
        1 or 2 — number of positional params fn accepts.  0 = auto-detect.
    """
    gaver_coeffs, binom_table = coeffs

    tau = ops["log2"](prec) / ops["mpf"](time, prec)
    tau_mpf = ops["to_mpf"](tau, prec)

    # --- Evaluate Laplace transform at s = k*tau for k = 1..2M ---
    fni: List[Any] = [None] * (2 * M + 1)  # type: ignore[list-item]

    from_mpf = ops["from_mpf"]  # avoid dict lookup in inner loop

    if fn_arity == 1:
        for i in range(1, 2 * M + 1):
            fni[i] = from_mpf(fn(i * tau_mpf), prec)
    elif fn_arity == 2:
        for i in range(1, 2 * M + 1):
            fni[i] = from_mpf(fn(i * tau_mpf, prec), prec)
    else:
        # Auto-detect
        try:
            fni[1] = from_mpf(fn(tau_mpf), prec)
            fn_arity = 1
        except TypeError:
            fni[1] = from_mpf(fn(tau_mpf, prec), prec)
            fn_arity = 2
        for i in range(2, 2 * M + 1):
            if fn_arity == 1:
                fni[i] = from_mpf(fn(i * tau_mpf), prec)
            else:
                fni[i] = from_mpf(fn(i * tau_mpf, prec), prec)

    # --- Gaver functionals ---
    zero = ops["mpf"](0, prec)
    G0: List[Any] = [zero] * M
    Gp: List[Any] = [zero] * M

    M1 = M
    for n in range(1, M + 1):
        try:
            binom_row = binom_table[n - 1]
            s = zero
            for i in range(n + 1):
                term = binom_row[i] * fni[n + i]
                if i & 1:
                    s = s - term
                else:
                    s = s + term
            G0[n - 1] = tau * gaver_coeffs[n - 1] * s
        except Exception as e:
            if n == 1:
                raise
            M1 = n - 1
            break

    # --- Wynn-rho sequence acceleration ---
    best = G0[M1 - 1]
    Gm: List[Any] = [zero] * M1

    broken = False
    for k in range(M1 - 1):
        for n in range(M1 - 2 - k, -1, -1):
            try:
                expr = G0[n + 1] - G0[n]
            except Exception:
                broken = True
                break
            Gp[n] = Gm[n + 1] + (k + 1) / expr
            if k % 2 == 1 and n == M1 - 2 - k:
                best = Gp[n]

        if broken:
            break

        for n in range(M1 - k):
            Gm[n] = G0[n]
            G0[n] = Gp[n]

    return best
