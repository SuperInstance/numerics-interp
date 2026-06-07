//! Barycentric interpolation (second form).
//!
//! An O(n) per-evaluation reformulation of Lagrange interpolation using
//! precomputed barycentric weights:
//! ```text
//! wᵢ = 1 / Π_{j≠i} (xᵢ - xⱼ)
//! L(x) = (Σᵢ wᵢ yᵢ / (x - xᵢ)) / (Σᵢ wᵢ / (x - xᵢ))
//! ```
//!
//! Numerically stable and efficient for repeated evaluations.

/// Barycentric interpolator.
pub struct BarycentricInterpolator {
    xs: Vec<f64>,
    ys: Vec<f64>,
    weights: Vec<f64>,
}

impl BarycentricInterpolator {
    /// Create a new barycentric interpolator from distinct nodes.
    ///
    /// # Errors
    ///
    /// Returns `None` if fewer than 2 points, duplicate x values, or length mismatch.
    pub fn new(xs: &[f64], ys: &[f64]) -> Option<Self> {
        let n = xs.len();
        if n < 2 || n != ys.len() {
            return None;
        }
        // Compute barycentric weights
        let mut weights = vec![0.0; n];
        for i in 0..n {
            let mut w = 1.0;
            for j in 0..n {
                if i != j {
                    let diff = xs[i] - xs[j];
                    if diff.abs() < 1e-14 {
                        return None;
                    }
                    w *= diff;
                }
            }
            if w.abs() < 1e-300 {
                return None;
            }
            weights[i] = 1.0 / w;
        }
        Some(Self { xs: xs.to_vec(), ys: ys.to_vec(), weights })
    }

    /// Evaluate the interpolant at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..self.xs.len() {
            let diff = x - self.xs[i];
            if diff.abs() < 1e-14 {
                return self.ys[i];
            }
            let term = self.weights[i] / diff;
            num += term * self.ys[i];
            den += term;
        }
        num / den
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.eval(x)).collect()
    }

    /// Number of nodes.
    pub fn len(&self) -> usize { self.xs.len() }

    /// Returns true if no nodes.
    pub fn is_empty(&self) -> bool { self.xs.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_at_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 3.0, 2.0, 5.0];
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        for i in 0..xs.len() {
            assert!((interp.eval(xs[i]) - ys[i]).abs() < 1e-10, "at node {i}");
        }
    }

    #[test]
    fn matches_lagrange() {
        let xs = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let ys: Vec<f64> = xs.iter().map(|&x: &f64| x.sin()).collect();
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        for x in [0.25, 0.75, 1.25, 1.75] {
            let val = interp.eval(x);
            // Compare with direct Lagrange computation
            let mut lag_val = 0.0;
            for i in 0..xs.len() {
                let mut basis = 1.0;
                for j in 0..xs.len() {
                    if i != j {
                        basis *= (x - xs[j]) / (xs[i] - xs[j]);
                    }
                }
                lag_val += ys[i] * basis;
            }
            assert!((val - lag_val).abs() < 1e-8, "bary={val}, lag={lag_val} at x={x}");
        }
    }

    #[test]
    fn linear_exact() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        for x in [0.5, 1.5, 2.5] {
            assert!((interp.eval(x) - (2.0 * x + 1.0)).abs() < 1e-10);
        }
    }

    #[test]
    fn quadratic_exact() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        for x in [0.5, 1.0, 1.5] {
            assert!((interp.eval(x) - x * x).abs() < 1e-10);
        }
    }

    #[test]
    fn rejects_duplicates() {
        let xs = vec![1.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        assert!(BarycentricInterpolator::new(&xs, &ys).is_none());
    }

    #[test]
    fn rejects_too_few() {
        assert!(BarycentricInterpolator::new(&[1.0], &[2.0]).is_none());
    }

    #[test]
    fn exp_interpolation() {
        let xs = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let ys: Vec<f64> = xs.iter().map(|&x: &f64| x.exp()).collect();
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        let val = interp.eval(0.5);
        let exact = 0.5_f64.exp();
        assert!((val - exact).abs() < 1e-10, "got {val}, exact {exact}");
    }

    #[test]
    fn eval_many() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        let vals = interp.eval_many(&[0.0, 0.5, 1.0, 1.5, 2.0]);
        assert!((vals[0]).abs() < 1e-10);
        assert!((vals[2] - 1.0).abs() < 1e-10);
        assert!((vals[4] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn len_and_empty() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        assert_eq!(interp.len(), 2);
        assert!(!interp.is_empty());
    }

    #[test]
    fn sin_convergence() {
        let err_5 = bary_sin_error(5);
        let err_10 = bary_sin_error(10);
        assert!(err_10 < err_5, "err_10={err_10}, err_5={err_5}");
    }

    fn bary_sin_error(n: usize) -> f64 {
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x: &f64| x.sin()).collect();
        let interp = BarycentricInterpolator::new(&xs, &ys).unwrap();
        let mut max_err: f64 = 0.0;
        for i in 0..50 {
            let x = std::f64::consts::PI * i as f64 / 50.0;
            if !xs.iter().any(|&xi| (xi - x).abs() < 1e-10) {
                max_err = max_err.max((interp.eval(x) - x.sin()).abs());
            }
        }
        max_err
    }
}
