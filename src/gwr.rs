//! Gaver-Wynn-Rho (GWR) Inverse Laplace Transform — Rust + MPFR acceleration.
//!
//! Implements the GWR algorithm with arbitrary-precision arithmetic via the
//! `rug` crate (GMP/MPFR bindings).  This is the gold-standard ILT method
//! for reservoir engineering — accurate even for oscillatory transforms.
//!
//! References:
//!   Valko & Abate (2004), Computers and Mathematics with Application 48(3).

use pyo3::prelude::*;
use rug::Float;
use rug::float::Round;
use rug::ops::AssignRound;

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
    gaver_coeffs: Vec<Float>,
    binom_table: Vec<Vec<Float>>,
    prec_bits: u32,
}

impl GwrCoefficients {
    fn new(m: usize, prec_dec: u32) -> Self {
        let prec = dec_to_bits(prec_dec);

        let max_k = 2 * m;
        let mut fac = Vec::with_capacity(max_k + 1);
        fac.push(Float::with_val(prec, 1));
        for k in 1..=max_k {
            let prev = &fac[k - 1];
            fac.push(Float::with_val(prec, prev * k as u32));
        }

        let mut gaver_coeffs = Vec::with_capacity(m);
        for n in 1..=m {
            let num = &fac[2 * n];
            let n_fac_m1 = &fac[n - 1];
            let den = Float::with_val(prec, n_fac_m1 * n_fac_m1) * n as u32;
            gaver_coeffs.push(Float::with_val(prec, num / &den));
        }

        let mut binom_table = Vec::with_capacity(m);
        for n in 1..=m {
            let mut row = Vec::with_capacity(n + 1);
            row.push(Float::with_val(prec, 1));
            for i in 1..=n {
                let prev = &row[i - 1];
                let val = Float::with_val(prec, prev * (n - i + 1) as u32) / i as u32;
                row.push(Float::with_val(prec, val));
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
    fni: &[Float],
    m: usize,
    tau: &Float,
    coeffs: &GwrCoefficients,
) -> f64 {
    let prec = coeffs.prec_bits;

    let m1 = m;
    let mut g0: Vec<Float> = Vec::with_capacity(m);

    for n in 1..=m {
        let binom_row = &coeffs.binom_table[n - 1];
        let mut s = Float::new(prec);
        for i in 0..=n {
            let idx = n + i - 1;
            let term = Float::with_val(prec, &binom_row[i] * &fni[idx]);
            if i & 1 == 1 {
                s -= term;
            } else {
                s += term;
            }
        }
        g0.push(Float::with_val(prec, tau * &coeffs.gaver_coeffs[n - 1]) * s);
    }

    let mut best = Float::with_val(prec, &g0[m1 - 1]);
    let mut gm: Vec<Float> = vec![Float::new(prec); m1];
    let mut gp: Vec<Float> = vec![Float::new(prec); m1];

    let mut broken = false;
    for k in 0..(m1 - 1) {
        for n in (0..=(m1 - 2 - k)).rev() {
            let diff = Float::with_val(prec, &g0[n + 1] - &g0[n]);
            if diff.is_zero() {
                broken = true;
                break;
            }
            let ratio = Float::with_val(prec, (k as f64 + 1.0) / &diff);
            gp[n] = Float::with_val(prec, &gm[n + 1] + &ratio);
            if k % 2 == 1 && n == m1 - 2 - k {
                best.assign_round(&gp[n], Round::Nearest);
            }
        }

        if broken {
            break;
        }

        for n in 0..(m1 - k) {
            gm[n].assign_round(&g0[n], Round::Nearest);
            g0[n].assign_round(&gp[n], Round::Nearest);
        }
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

    let mut results = Vec::with_capacity(times.len());
    for &t in &times {
        let mut tau = Float::with_val(prec_bits, 2u32).ln();
        tau /= t;

        let mut fni = Vec::with_capacity(2 * m);
        for k in 1..=(2 * m) {
            let s_val = Float::with_val(prec_bits, &tau * k as u32).to_f64();
            let fs: f64 = fn_obj.call1((s_val,))?.extract()?;
            fni.push(Float::with_val(prec_bits, fs));
        }

        results.push(gwr_single(&fni, m, &tau, &coeffs));
    }
    Ok(results)
}
