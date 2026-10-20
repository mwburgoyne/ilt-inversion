//! Arbitrary-precision modified Bessel functions I_n and K_n (`mpfloat::Mp`).
//!
//! Power series for small arguments, asymptotic expansion for large.
//! Guard bits compensate for catastrophic cancellation in K_n series.

use pyo3::prelude::*;
use crate::gwr::dec_to_bits;
use crate::mpfloat::Mp;

// The asymptotic series, optimally truncated, is good to a relative e^(-2x),
// so it is used only once that is below 2^-prec (with a margin). Below the
// switch the K_n series cancels terms of size e^x down to a result of size
// e^-x, so the series runs with 2x*log2(e) guard bits. A fixed x > 25 switch
// with x*log2(e) guard bits capped the result at about 22 digits.
const ASYMP_MARGIN_BITS: f64 = 16.0;
const SERIES_EXTRA_GUARD_BITS: u32 = 20;

/// True when the asymptotic expansion reaches `prec` bits at `x`.
fn use_asymptotic(x: &Mp, prec: u32) -> bool {
    2.0 * x.to_f64() * std::f64::consts::LOG2_E > prec as f64 + ASYMP_MARGIN_BITS
}

/// True when `term` no longer changes `sum` at `prec` bits.
fn series_negligible(term: &Mp, sum: &Mp, prec: u32) -> bool {
    match (term.exponent(), sum.exponent()) {
        (Some(te), Some(se)) => te < se - prec as i32,
        _ => term.is_zero(),
    }
}

/// Loop bound for the asymptotic sum: past the optimal truncation index (~2x).
fn asymptotic_max_terms(x: &Mp) -> u64 {
    (4.0 * x.to_f64()) as u64 + 100
}

/// Guard bits for evaluating a series at `x`.
fn series_guard_bits(x: &Mp) -> u32 {
    (2.0 * x.to_f64() * std::f64::consts::LOG2_E).ceil() as u32 + SERIES_EXTRA_GUARD_BITS
}

// =========================================================================
//  Asymptotic expansions (large x) — scaled forms
// =========================================================================

fn besseli_asymp(x: &Mp, nu: u32, prec: u32) -> Mp {
    let mu = (4 * nu * nu) as f64;
    let one_over_x = &Mp::from_u64(1, prec) / x;

    let mut sum = Mp::from_u64(1, prec);
    let mut a_k = Mp::from_u64(1, prec);
    let mut prev_term_abs = Mp::from_f64(f64::MAX, prec);

    // Optimal truncation sits near k = 2x, so the cap scales with x; a fixed
    // 200 terms fell short once the precision-scaled switch passed x = 100.
    for k in 1u64..asymptotic_max_terms(x) {
        let sq = (2 * k - 1) * (2 * k - 1);
        // The I expansion carries (-1)^k (A&S 9.7.1), folded in here; K (9.7.2) does not
        let numer = sq as f64 - mu;
        a_k = &(&(&a_k * numer) / (8 * k)) * &one_over_x;

        let term_abs = a_k.abs();
        if term_abs > prev_term_abs || series_negligible(&a_k, &sum, prec) { break; }
        prev_term_abs = term_abs;
        sum = &sum + &a_k;
    }

    let two_pi_x = &(x * &Mp::pi(prec)) * 2u64;
    &sum / &two_pi_x.sqrt()
}

fn besselk_asymp(x: &Mp, nu: u32, prec: u32) -> Mp {
    let mu = (4 * nu * nu) as f64;
    let one_over_x = &Mp::from_u64(1, prec) / x;

    let mut sum = Mp::from_u64(1, prec);
    let mut a_k = Mp::from_u64(1, prec);
    let mut prev_term_abs = Mp::from_f64(f64::MAX, prec);

    // Optimal truncation sits near k = 2x, so the cap scales with x; a fixed
    // 200 terms fell short once the precision-scaled switch passed x = 100.
    for k in 1u64..asymptotic_max_terms(x) {
        let sq = (2 * k - 1) * (2 * k - 1);
        let numer = mu - sq as f64;
        a_k = &(&(&a_k * numer) / (8 * k)) * &one_over_x;

        let term_abs = a_k.abs();
        if term_abs > prev_term_abs || series_negligible(&a_k, &sum, prec) { break; }
        prev_term_abs = term_abs;
        sum = &sum + &a_k;
    }

    let pi_over_2x = &(&Mp::pi(prec) / x) / 2u64;
    &sum * &pi_over_2x.sqrt()
}

// =========================================================================
//  Power series (small x)
// =========================================================================

fn besseli0_series(x: &Mp, prec: u32) -> Mp {
    let x2_over4 = &(x * x) / 4u64;
    let mut result = Mp::from_u64(1, prec);
    let mut term = Mp::from_u64(1, prec);
    for k in 1u64..100_000 {
        term = &(&term * &x2_over4) / (k * k);
        result = &result + &term;
        if k > 10 && series_negligible(&term, &result, prec) { break; }
    }
    result
}

fn besseli1_series(x: &Mp, prec: u32) -> Mp {
    let x2_over4 = &(x * x) / 4u64;
    let mut sum = Mp::from_u64(1, prec);
    let mut term = Mp::from_u64(1, prec);
    for k in 1u64..100_000 {
        term = &(&term * &x2_over4) / (k * (k + 1));
        sum = &sum + &term;
        if k > 10 && series_negligible(&term, &sum, prec) { break; }
    }
    &(x / 2u64) * &sum
}

fn besselk0_series(x: &Mp, prec: u32) -> Mp {
    let i0 = besseli0_series(x, prec);
    let gamma = Mp::euler(prec);
    let ln_x_over_2 = (x / 2u64).ln();

    let mut result = -&(&(&ln_x_over_2 + &gamma) * &i0);

    let x2_over4 = &(x * x) / 4u64;
    let mut term = x2_over4.clone();
    let mut h_k = Mp::from_u64(1, prec);
    result = &result + &(&h_k * &term);

    for k in 2u64..100_000 {
        term = &(&term * &x2_over4) / (k * k);
        h_k = &h_k + &(&Mp::from_u64(1, prec) / k);
        let contribution = &h_k * &term;
        result = &result + &contribution;
        if k > 10 && series_negligible(&contribution, &result, prec) { break; }
    }
    result
}

fn besselk1_series(x: &Mp, prec: u32) -> Mp {
    let i0 = besseli0_series(x, prec);
    let i1 = besseli1_series(x, prec);
    let k0 = besselk0_series(x, prec);
    let one_over_x = &Mp::from_u64(1, prec) / x;
    &(&one_over_x - &(&i1 * &k0)) / &i0
}

// =========================================================================
//  Public dispatch: series (with guard bits) or asymptotic
// =========================================================================

pub fn mpfr_besseli_scaled(x: &Mp, nu: u32, prec: u32) -> Mp {
    if !use_asymptotic(x, prec) {
        let guard = series_guard_bits(x);
        let hp = prec + guard;
        let xhp = x.with_prec(hp);
        let val = match nu {
            0 => besseli0_series(&xhp, hp),
            1 => besseli1_series(&xhp, hp),
            _ => panic!("Only nu=0,1 supported"),
        };
        (&val * &(-&xhp).exp()).with_prec(prec)
    } else {
        besseli_asymp(x, nu, prec)
    }
}

pub fn mpfr_besselk_scaled(x: &Mp, nu: u32, prec: u32) -> Mp {
    if !use_asymptotic(x, prec) {
        let guard = series_guard_bits(x);
        let hp = prec + guard;
        let xhp = x.with_prec(hp);
        let val = match nu {
            0 => besselk0_series(&xhp, hp),
            1 => besselk1_series(&xhp, hp),
            _ => panic!("Only nu=0,1 supported"),
        };
        (&val * &xhp.exp()).with_prec(prec)
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
    let xf = Mp::from_f64(x, bits);
    let scaled = mpfr_besseli_scaled(&xf, n, bits);
    // Unscale: I_n(x) = I_ne(x) * exp(x)
    Ok((&scaled * &xf.exp()).to_f64())
}

/// Compute modified Bessel function K_n(x) at arbitrary precision.
/// Returns float. n must be 0 or 1. x must be > 0.
#[pyfunction]
#[pyo3(signature = (n, x, prec=34))]
pub fn besselk_rust(n: u32, x: f64, prec: u32) -> PyResult<f64> {
    let bits = dec_to_bits(prec);
    let xf = Mp::from_f64(x, bits);
    let scaled = mpfr_besselk_scaled(&xf, n, bits);
    Ok((&scaled * &(-&xf).exp()).to_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Series and asymptotic paths agree to the working precision just past
    /// the switch point, for both kinds and orders, at 25 and 67 digits.
    #[test]
    fn series_asymptotic_overlap() {
        for prec in [dec_to_bits(25), dec_to_bits(67), dec_to_bits(134), dec_to_bits(300), dec_to_bits(700)] {
            let x_switch = (prec as f64 + ASYMP_MARGIN_BITS) / (2.0 * std::f64::consts::LOG2_E);
            let x = Mp::from_f64(x_switch + 0.5, prec);
            let hp = prec + series_guard_bits(&x);
            let xhp = x.with_prec(hp);
            let up = xhp.exp();
            let down = (-&xhp).exp();
            let cases = [
                (&besseli0_series(&xhp, hp) * &down, besseli_asymp(&x, 0, prec)),
                (&besseli1_series(&xhp, hp) * &down, besseli_asymp(&x, 1, prec)),
                (&besselk0_series(&xhp, hp) * &up, besselk_asymp(&x, 0, prec)),
                (&besselk1_series(&xhp, hp) * &up, besselk_asymp(&x, 1, prec)),
            ];
            for (i, (series, asymp)) in cases.iter().enumerate() {
                let rel = (&(series - asymp) / asymp).abs();
                let bits = -rel.to_f64().log2();
                assert!(bits > prec as f64 - 12.0, "case {} prec {}: only {:.0} bits agree", i, prec, bits);
            }
        }
    }

    /// Scaled I_0, I_1, K_0, K_1 against mpmath at 100 digits, both sides of
    /// the switch (mpmath 1.3, mp.dps = 110).
    #[test]
    fn matches_mpmath_at_100_digits() {
        let prec = dec_to_bits(100);
        let reference: [(&str, [&str; 4]); 7] = [
            ("0.001", ["0.999000749583515559395046764639456554789316767358788582699072538198566965189609396026082670294922984226577", "0.000499500312354221336983820506497873799262010996972098326455710985892870321187955688287572629959318711877092", "7.03071600237825151848061412739232252500790428381336683573253255296656125348677551295311352625906278416725", "1000.99673455906845243231366241041866787259215450285347225336727836719744625470833853803731885954184949430"]),
            ("0.7", ["0.559305526507068317705986933790815280198531690174559030097844404935123809756358788651982193096431324131093", "0.184669982762747324567156081047686881864007813733768147427686993094025659926677915755811059508623404753047", "1.33012365624205562011222477052427596899064423146508929855201122499549742350878258867622174728343706548649", "2.11501131284805236117467532823326595298245067293900228104109334323400982964559372683239583397523291131970"]),
            ("5", ["0.183540812609328353073650751837396677071552992944265677521533571441920626479281602791151390587113026109164", "0.163972266944542356926122903857531712071570190870036637335381144534989296137234617739505462306638391790581", "0.547807564313518986868201568742870282223396090853118338855275999063337138575248689443904651560470064795655", "0.600273858788312582936045656627059826657768518031746874339422682331827478272917275548189451562351396292891"]),
            ("23.5", ["0.0827442263816562798326606886596376695293161784587809493118689223564767029175891164978921797818622047876046", "0.0809641267899889406869993423786370356555223627963405802267238782263020369249676060992062789711933631337320", "0.257195304388912854075002389137657900517152808756151652500148286648507436462736804368632825720525653953509", "0.262611655280944622628066195808451264007193856613766721226164960614934864897947691084148879845058165988065"]),
            ("60", ["0.0516115491736098409485656502259201219221508814546575798901386054042887740472307177830470426619510462413296", "0.0511796301890287181184593726440717802760518979266730770871607906913936911969875948894481328287740448983037", "0.161468178236293925648388738233593668117934932785352730691549709024851429851092820525228755336937787794627", "0.162808230944044271031125024384568087390406224942514678888854885038237947042159852004306286133587096486519"]),
            ("150", ["0.0326007478839180494850559044868671709225828886732947154619276794412387751432533591411095520922873536005504", "0.0324918963888489424816053537535592922142566371994303674207295638209844866113293981679660327993231806741824", "0.102247711160191052771125668554237816980962388606895066195987797686384586162885850536896951046853796638665", "0.102587972569261209568112767246550147785858847291697931855463198546373962822711937828785867799606616154933"]),
            ("400", ["0.0199533562819399898712967947299769564475028132790480696676537508311637191292937573770298667203716740156511", "0.0199283989589035418522544778426415582911795928512358108996177823935042873514918979013483560060518783539008", "0.0626461512995781138289217039252431519196378162290600113133361443379604466297522051467178182988384697032177", "0.0627244101682771651757802859054632422181692312611018656970814590089716031744019903537757001323105397906361"]),
        ];
        for (xs, vals) in reference.iter() {
            let x = Mp::parse_dec(xs, prec);
            let got = [
                mpfr_besseli_scaled(&x, 0, prec),
                mpfr_besseli_scaled(&x, 1, prec),
                mpfr_besselk_scaled(&x, 0, prec),
                mpfr_besselk_scaled(&x, 1, prec),
            ];
            for (i, (g, v)) in got.iter().zip(vals.iter()).enumerate() {
                let r = Mp::parse_dec(v, prec + 64);
                let rel = (&(g - &r) / &r).abs();
                let bits = if rel.is_zero() { f64::INFINITY } else { -rel.to_f64().log2() };
                assert!(bits > prec as f64 - 12.0, "x {} fn {}: only {:.0} of {} bits", xs, i, bits, prec);
            }
        }
    }
}
