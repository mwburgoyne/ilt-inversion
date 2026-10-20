//! MPFR-precision modified Bessel functions I_n and K_n.
//!
//! Power series for small arguments, asymptotic expansion for large.
//! Guard bits compensate for catastrophic cancellation in K_n series.

use pyo3::prelude::*;
use rug::Float;
use rug::float::{Constant, Round};
use rug::ops::AssignRound;

use crate::gwr::dec_to_bits;

// The asymptotic series, optimally truncated, is good to a relative e^(-2x),
// so it is used only once that is below 2^-prec (with a margin). Below the
// switch the K_n series cancels terms of size e^x down to a result of size
// e^-x, so the series runs with 2x*log2(e) guard bits. A fixed x > 25 switch
// with x*log2(e) guard bits capped the result at about 22 digits.
const ASYMP_MARGIN_BITS: f64 = 16.0;
const SERIES_EXTRA_GUARD_BITS: u32 = 20;

/// True when the asymptotic expansion reaches `prec` bits at `x`.
fn use_asymptotic(x: &Float, prec: u32) -> bool {
    2.0 * x.to_f64() * std::f64::consts::LOG2_E > prec as f64 + ASYMP_MARGIN_BITS
}

/// True when `term` no longer changes `sum` at `prec` bits.
fn series_negligible(term: &Float, sum: &Float, prec: u32) -> bool {
    match (term.get_exp(), sum.get_exp()) {
        (Some(te), Some(se)) => te < se - prec as i32,
        _ => term.is_zero(),
    }
}

/// Guard bits for evaluating a series at `x`.
fn series_guard_bits(x: &Float) -> u32 {
    (2.0 * x.to_f64() * std::f64::consts::LOG2_E).ceil() as u32 + SERIES_EXTRA_GUARD_BITS
}

// =========================================================================
//  Asymptotic expansions (large x) — scaled forms
// =========================================================================

fn besseli_asymp(x: &Float, nu: u32, prec: u32) -> Float {
    let mu = (4 * nu * nu) as f64;
    let one_over_x = Float::with_val(prec, 1) / x;

    let mut sum = Float::with_val(prec, 1);
    let mut a_k = Float::with_val(prec, 1);
    let mut prev_term_abs = Float::with_val(prec, f64::MAX);

    for k in 1u64..200 {
        let sq = (2 * k - 1) * (2 * k - 1);
        // The I expansion carries (-1)^k (A&S 9.7.1), folded in here; K (9.7.2) does not
        let numer = sq as f64 - mu;
        a_k *= Float::with_val(prec, numer);
        a_k /= 8u32 * k as u32;
        a_k *= &one_over_x;

        let term_abs = Float::with_val(prec, a_k.clone().abs());
        if term_abs > prev_term_abs { break; }
        prev_term_abs.assign_round(&term_abs, Round::Nearest);
        sum += &a_k;
    }

    let two_pi_x = Float::with_val(prec, x * Float::with_val(prec, Constant::Pi)) * 2u32;
    Float::with_val(prec, &sum / two_pi_x.sqrt())
}

fn besselk_asymp(x: &Float, nu: u32, prec: u32) -> Float {
    let mu = (4 * nu * nu) as f64;
    let one_over_x = Float::with_val(prec, 1) / x;

    let mut sum = Float::with_val(prec, 1);
    let mut a_k = Float::with_val(prec, 1);
    let mut prev_term_abs = Float::with_val(prec, f64::MAX);

    for k in 1u64..200 {
        let sq = (2 * k - 1) * (2 * k - 1);
        let numer = mu - sq as f64;
        a_k *= Float::with_val(prec, numer);
        a_k /= 8u32 * k as u32;
        a_k *= &one_over_x;

        let term_abs = Float::with_val(prec, a_k.clone().abs());
        if term_abs > prev_term_abs { break; }
        prev_term_abs.assign_round(&term_abs, Round::Nearest);
        sum += &a_k;
    }

    let pi_over_2x = Float::with_val(prec, Float::with_val(prec, Constant::Pi) / x) / 2u32;
    Float::with_val(prec, &sum * pi_over_2x.sqrt())
}

// =========================================================================
//  Power series (small x)
// =========================================================================

fn besseli0_series(x: &Float, prec: u32) -> Float {
    let x2_over4 = Float::with_val(prec, x * x) / 4u32;
    let mut result = Float::with_val(prec, 1);
    let mut term = Float::with_val(prec, 1);
    for k in 1u64..100_000 {
        term *= &x2_over4;
        term /= k * k;
        result += &term;
        if k > 10 && series_negligible(&term, &result, prec) { break; }
    }
    result
}

fn besseli1_series(x: &Float, prec: u32) -> Float {
    let x2_over4 = Float::with_val(prec, x * x) / 4u32;
    let mut sum = Float::with_val(prec, 1);
    let mut term = Float::with_val(prec, 1);
    for k in 1u64..100_000 {
        term *= &x2_over4;
        term /= k * (k + 1);
        sum += &term;
        if k > 10 && series_negligible(&term, &sum, prec) { break; }
    }
    Float::with_val(prec, x / 2u32) * sum
}

fn besselk0_series(x: &Float, prec: u32) -> Float {
    let i0 = besseli0_series(x, prec);
    let gamma = Float::with_val(prec, Constant::Euler);
    let ln_x_over_2 = Float::with_val(prec, x / 2u32).ln();

    let mut result = Float::with_val(prec, &ln_x_over_2 + &gamma);
    result = -result * &i0;

    let x2_over4 = Float::with_val(prec, x * x) / 4u32;
    let mut term = Float::with_val(prec, &x2_over4);
    let mut h_k = Float::with_val(prec, 1);
    result += Float::with_val(prec, &h_k * &term);

    for k in 2u64..100_000 {
        term *= &x2_over4;
        term /= k * k;
        h_k += Float::with_val(prec, 1) / Float::with_val(prec, k);
        let contribution = Float::with_val(prec, &h_k * &term);
        result += &contribution;
        if k > 10 && series_negligible(&contribution, &result, prec) { break; }
    }
    result
}

fn besselk1_series(x: &Float, prec: u32) -> Float {
    let i0 = besseli0_series(x, prec);
    let i1 = besseli1_series(x, prec);
    let k0 = besselk0_series(x, prec);
    let one_over_x = Float::with_val(prec, 1) / x;
    Float::with_val(prec, &one_over_x - Float::with_val(prec, &i1 * &k0)) / &i0
}

// =========================================================================
//  Public dispatch: series (with guard bits) or asymptotic
// =========================================================================

pub fn mpfr_besseli_scaled(x: &Float, nu: u32, prec: u32) -> Float {
    if !use_asymptotic(x, prec) {
        let guard = series_guard_bits(x);
        let hp = prec + guard;
        let xhp = Float::with_val(hp, x);
        let val = match nu {
            0 => besseli0_series(&xhp, hp),
            1 => besseli1_series(&xhp, hp),
            _ => panic!("Only nu=0,1 supported"),
        };
        let result = Float::with_val(hp, &val * Float::with_val(hp, -&xhp).exp());
        Float::with_val(prec, &result)
    } else {
        besseli_asymp(x, nu, prec)
    }
}

pub fn mpfr_besselk_scaled(x: &Float, nu: u32, prec: u32) -> Float {
    if !use_asymptotic(x, prec) {
        let guard = series_guard_bits(x);
        let hp = prec + guard;
        let xhp = Float::with_val(hp, x);
        let val = match nu {
            0 => besselk0_series(&xhp, hp),
            1 => besselk1_series(&xhp, hp),
            _ => panic!("Only nu=0,1 supported"),
        };
        let result = Float::with_val(hp, &val * xhp.exp());
        Float::with_val(prec, &result)
    } else {
        besselk_asymp(x, nu, prec)
    }
}

// =========================================================================
//  PyO3 exports
// =========================================================================

/// Compute modified Bessel function I_n(x) at arbitrary precision.
/// Returns float. n must be 0 or 1.
#[pyfunction]
#[pyo3(signature = (n, x, prec=34))]
pub fn besseli_rust(n: u32, x: f64, prec: u32) -> PyResult<f64> {
    let bits = dec_to_bits(prec);
    let xf = Float::with_val(bits, x);
    let scaled = mpfr_besseli_scaled(&xf, n, bits);
    // Unscale: I_n(x) = I_ne(x) * exp(x)
    let result = Float::with_val(bits, &scaled * xf.exp());
    Ok(result.to_f64())
}

/// Compute modified Bessel function K_n(x) at arbitrary precision.
/// Returns float. n must be 0 or 1. x must be > 0.
#[pyfunction]
#[pyo3(signature = (n, x, prec=34))]
pub fn besselk_rust(n: u32, x: f64, prec: u32) -> PyResult<f64> {
    let bits = dec_to_bits(prec);
    let xf = Float::with_val(bits, x);
    let scaled = mpfr_besselk_scaled(&xf, n, bits);
    let neg_x = Float::with_val(bits, -x);
    let result = Float::with_val(bits, &scaled * neg_x.exp());
    Ok(result.to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Series and asymptotic paths agree to the working precision just past
    /// the switch point, for both kinds and orders, at 25 and 67 digits.
    #[test]
    fn series_asymptotic_overlap() {
        for prec in [dec_to_bits(25), dec_to_bits(67)] {
            let x_switch = (prec as f64 + ASYMP_MARGIN_BITS) / (2.0 * std::f64::consts::LOG2_E);
            let x = Float::with_val(prec, x_switch + 0.5);
            let hp = prec + series_guard_bits(&x);
            let xhp = Float::with_val(hp, &x);
            let up = Float::with_val(hp, &xhp).exp();
            let down = Float::with_val(hp, -&xhp).exp();
            let cases = [
                (Float::with_val(hp, besseli0_series(&xhp, hp) * &down), besseli_asymp(&x, 0, prec)),
                (Float::with_val(hp, besseli1_series(&xhp, hp) * &down), besseli_asymp(&x, 1, prec)),
                (Float::with_val(hp, besselk0_series(&xhp, hp) * &up), besselk_asymp(&x, 0, prec)),
                (Float::with_val(hp, besselk1_series(&xhp, hp) * &up), besselk_asymp(&x, 1, prec)),
            ];
            for (i, (series, asymp)) in cases.iter().enumerate() {
                let rel = Float::with_val(prec, Float::with_val(prec, series - asymp) / asymp).abs();
                let bits = -rel.to_f64().log2();
                assert!(bits > prec as f64 - 12.0, "case {} prec {}: only {:.0} bits agree", i, prec, bits);
            }
        }
    }
}
