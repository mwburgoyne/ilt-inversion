//! Gaver-Wynn-Rho (GWR) Inverse Laplace Transform in arbitrary precision.
//!
//! Implements the GWR algorithm with arbitrary-precision arithmetic via
//! `mpfloat::Mp` (pure-Rust `astro-float`). This is the gold-standard ILT
//! method for reservoir engineering, accurate even for oscillatory transforms.
//!
//! References:
//!   Valko & Abate (2004), Computers and Mathematics with Application 48(3).

use pyo3::prelude::*;
use crate::mpfloat::Mp;

// =========================================================================
//  Precision helpers
// =========================================================================

#[inline]
pub fn dec_to_bits(dec_digits: u32) -> u32 {
    (dec_digits as f64 * std::f64::consts::LOG2_10).ceil() as u32 + 10
}

// =========================================================================
//  GWR coefficient pre-computation
// =========================================================================

struct GwrCoefficients {
    gaver_coeffs: Vec<Mp>,
    binom_table: Vec<Vec<Mp>>,
    prec_bits: u32,
}

impl GwrCoefficients {
    fn new(m: usize, prec_dec: u32) -> Self {
        let prec = dec_to_bits(prec_dec);

        let max_k = 2 * m;
        let mut fac = Vec::with_capacity(max_k + 1);
        fac.push(Mp::from_u64(1, prec));
        for k in 1..=max_k {
            let next = &fac[k - 1] * k as u64;
            fac.push(next);
        }

        let mut gaver_coeffs = Vec::with_capacity(m);
        for n in 1..=m {
            let den = &(&fac[n - 1] * &fac[n - 1]) * n as u64;
            gaver_coeffs.push(&fac[2 * n] / &den);
        }

        let mut binom_table = Vec::with_capacity(m);
        for n in 1..=m {
            let mut row = Vec::with_capacity(n + 1);
            row.push(Mp::from_u64(1, prec));
            for i in 1..=n {
                let val = &(&row[i - 1] * (n - i + 1) as u64) / i as u64;
                row.push(val);
            }
            binom_table.push(row);
        }

        GwrCoefficients {
            gaver_coeffs,
            binom_table,
            prec_bits: prec,
        }
    }
}

// =========================================================================
//  Core GWR algorithm
// =========================================================================

fn gwr_single(
    fni: &[Mp],
    m: usize,
    tau: &Mp,
    coeffs: &GwrCoefficients,
) -> f64 {
    let prec = coeffs.prec_bits;

    let m1 = m;
    let mut g0: Vec<Mp> = Vec::with_capacity(m);

    for n in 1..=m {
        let binom_row = &coeffs.binom_table[n - 1];
        let mut s = Mp::zero(prec);
        for (i, binom) in binom_row.iter().enumerate().take(n + 1) {
            let idx = n + i - 1;
            let term = binom * &fni[idx];
            s = if i & 1 == 1 { &s - &term } else { &s + &term };
        }
        g0.push(&(tau * &coeffs.gaver_coeffs[n - 1]) * &s);
    }

    let mut best = g0[m1 - 1].clone();
    let mut gm: Vec<Mp> = vec![Mp::zero(prec); m1];
    let mut gp: Vec<Mp> = vec![Mp::zero(prec); m1];

    let mut broken = false;
    for k in 0..(m1 - 1) {
        for n in (0..=(m1 - 2 - k)).rev() {
            let diff = &g0[n + 1] - &g0[n];
            if diff.is_zero() {
                broken = true;
                break;
            }
            let ratio = &Mp::from_f64(k as f64 + 1.0, prec) / &diff;
            gp[n] = &gm[n + 1] + &ratio;
            if k % 2 == 1 && n == m1 - 2 - k {
                best = gp[n].clone();
            }
        }

        if broken {
            break;
        }

        let len = m1 - k;
        gm[..len].clone_from_slice(&g0[..len]);
        g0[..len].clone_from_slice(&gp[..len]);
    }

    best.to_f64()
}

// =========================================================================
//  PyO3 export
// =========================================================================

/// General GWR inverse Laplace transform for Python callables.
///
/// Evaluates the inverse Laplace transform of fn_obj at each time point
/// using the Gaver-Wynn-Rho algorithm with arbitrary-precision arithmetic.
///
/// Parameters:
///   fn_obj: Python callable F(s) -> float (Laplace-domain function)
///   times:  list of time values (must be > 0)
///   m:      number of Gaver functional terms (default 32)
///   prec:   decimal digits of precision (default: round(2.1 * m))
#[pyfunction]
#[pyo3(signature = (fn_obj, times, m=32, prec=None))]
pub fn gwr_rust(
    _py: Python<'_>,
    fn_obj: Bound<'_, PyAny>,
    times: Vec<f64>,
    m: usize,
    prec: Option<u32>,
) -> PyResult<Vec<f64>> {
    let prec_dec = prec.unwrap_or_else(|| (2.1 * m as f64).round() as u32);
    let prec_bits = dec_to_bits(prec_dec);
    let coeffs = GwrCoefficients::new(m, prec_dec);
    let ln2 = Mp::from_u64(2, prec_bits).ln();

    let mut results = Vec::with_capacity(times.len());
    for &t in &times {
        let tau = &ln2 / t;

        let mut fni = Vec::with_capacity(2 * m);
        for k in 1..=(2 * m) {
            let s_val = (&tau * k as u64).to_f64();
            let fs: f64 = fn_obj.call1((s_val,))?.extract()?;
            fni.push(Mp::from_f64(fs, prec_bits));
        }

        results.push(gwr_single(&fni, m, &tau, &coeffs));
    }
    Ok(results)
}
