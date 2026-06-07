//! Interpolation error bounds and estimation utilities.
//!
//! Provides theoretical error bounds for common interpolation methods.

/// Estimate the Lagrange interpolation error bound.
///
/// For a polynomial of degree `n` interpolating a function with known
/// `(n+1)`-th derivative bound `M`, the error at `x` satisfies:
/// ```text
/// |f(x) - P(x)| ≤ M / (n+1)! · Πᵢ |x - xᵢ|
/// ```
///
/// # Arguments
///
/// * `x`       — Evaluation point.
/// * `xs`      — Interpolation nodes.
/// * `deriv_bound` — Upper bound on |f^(n+1)(ξ)| for some ξ in the interval.
///
/// # Returns
///
/// Theoretical upper bound on the absolute interpolation error.
pub fn lagrange_error_bound(x: f64, xs: &[f64], deriv_bound: f64) -> f64 {
    let n = xs.len();
    // Actually for degree n-1 poly with n nodes, we need n! not (n+1)!
    let factorial: f64 = (1..=n).fold(1.0, |acc, i| acc * i as f64);
    let product: f64 = xs.iter().map(|xi| (x - xi).abs()).fold(1.0, |acc, v| acc * v);
    deriv_bound * product / factorial
}

/// Estimate the cubic spline interpolation error bound.
///
/// For a natural cubic spline interpolating `f` with `|f⁴| ≤ M₄` on a uniform
/// grid with spacing `h`:
/// ```text
/// |f(x) - S(x)| ≤ (5/384) · M₄ · h⁴
/// ```
pub fn cubic_spline_error_bound(h: f64, fourth_deriv_bound: f64) -> f64 {
    (5.0 / 384.0) * fourth_deriv_bound * h.powi(4)
}

/// Compute the maximum interpolation error on a test grid.
///
/// Compares interpolated values against exact values.
pub fn max_error(interpolated: &[f64], exact: &[f64]) -> f64 {
    interpolated.iter().zip(exact.iter()).map(|(a, b)| (a - b).abs()).fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lagrange::LagrangeInterpolator;
    use crate::cubic_spline::CubicSpline;

    #[test]
    fn lagrange_bound_is_valid() {
        // Interpolate sin(x) on [0, π] with 5 nodes
        let n = 5;
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        // sin has bounded derivatives ≤ 1
        for i in 0..20 {
            let x = std::f64::consts::PI * i as f64 / 20.0;
            let actual_err = (interp.eval(x) - x.sin()).abs();
            let bound = lagrange_error_bound(x, &xs, 1.0);
            if actual_err > 1e-12 {
                assert!(actual_err <= bound + 1e-6,
                    "actual={actual_err}, bound={bound} at x={x}");
            }
        }
    }

    #[test]
    fn spline_bound_is_valid() {
        let n = 10;
        let h = std::f64::consts::PI / n as f64;
        let xs: Vec<f64> = (0..=n).map(|i| h * i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        let bound = cubic_spline_error_bound(h, 1.0);
        for i in 0..50 {
            let x = std::f64::consts::PI * i as f64 / 50.0;
            let actual_err = (spline.eval(x) - x.sin()).abs();
            assert!(actual_err <= bound + 1e-6,
                "actual={actual_err}, bound={bound} at x={x}");
        }
    }

    #[test]
    fn max_error_computation() {
        let interp = [1.0, 2.0, 3.0];
        let exact = [1.1, 2.0, 3.2];
        assert!((max_error(&interp, &exact) - 0.2).abs() < 1e-12);
    }

    #[test]
    fn max_error_zero() {
        let vals = [1.0, 2.0, 3.0];
        assert!(max_error(&vals, &vals) < 1e-15);
    }

    #[test]
    fn spline_error_decreases_with_h() {
        let b1 = cubic_spline_error_bound(0.1, 1.0);
        let b2 = cubic_spline_error_bound(0.05, 1.0);
        assert!(b2 < b1, "b2={b2}, b1={b1}");
        // h halved → error should decrease by ~16x
        let ratio = b1 / b2;
        assert!((ratio - 16.0).abs() < 1.0, "ratio={ratio}");
    }

    #[test]
    fn lagrange_bound_decreases_with_nodes() {
        let xs5: Vec<f64> = (0..=5).map(|i| i as f64).collect();
        let xs10: Vec<f64> = (0..=10).map(|i| i as f64 / 2.0).collect();
        let x = 2.5;
        let b5 = lagrange_error_bound(x, &xs5, 1.0);
        let b10 = lagrange_error_bound(x, &xs10, 1.0);
        assert!(b10 < b5, "b10={b10}, b5={b5}");
    }

    #[test]
    fn lagrange_error_at_node_is_zero() {
        let xs = vec![0.0, 1.0, 2.0];
        let bound = lagrange_error_bound(1.0, &xs, 1.0);
        assert!(bound < 1e-15, "bound={bound}");
    }
}
