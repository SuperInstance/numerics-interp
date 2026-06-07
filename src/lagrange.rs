//! Lagrange polynomial interpolation.
//!
//! Given nodes `(x₀,y₀),…,(xₙ,yₙ)`, the Lagrange interpolating polynomial is:
//! ```text
//! L(x) = Σᵢ yᵢ · ℓᵢ(x)
//! ℓᵢ(x) = Π_{j≠i} (x - xⱼ) / (xᵢ - xⱼ)
//! ```
//!
//! Degree n polynomial passing through all n+1 nodes.
//! Computational cost: O(n²) per evaluation.

/// Lagrange interpolator.
pub struct LagrangeInterpolator {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl LagrangeInterpolator {
    /// Create a new Lagrange interpolator.
    ///
    /// # Errors
    ///
    /// Returns `None` if fewer than 2 points or `xs` contains duplicates.
    pub fn new(xs: &[f64], ys: &[f64]) -> Option<Self> {
        if xs.len() < 2 || xs.len() != ys.len() {
            return None;
        }
        for i in 0..xs.len() {
            for j in (i + 1)..xs.len() {
                if (xs[i] - xs[j]).abs() < 1e-14 {
                    return None;
                }
            }
        }
        Some(Self { xs: xs.to_vec(), ys: ys.to_vec() })
    }

    /// Evaluate the Lagrange polynomial at `x`.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        let mut result = 0.0;
        for i in 0..n {
            let mut basis = 1.0;
            for j in 0..n {
                if i != j {
                    basis *= (x - self.xs[j]) / (self.xs[i] - self.xs[j]);
                }
            }
            result += self.ys[i] * basis;
        }
        result
    }

    /// Number of interpolation nodes.
    pub fn len(&self) -> usize { self.xs.len() }

    /// Returns true if the interpolator has no nodes.
    pub fn is_empty(&self) -> bool { self.xs.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_at_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 3.0, 2.0, 5.0];
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        for i in 0..xs.len() {
            assert!((interp.eval(xs[i]) - ys[i]).abs() < 1e-10, "at node {i}");
        }
    }

    #[test]
    fn linear_function_exact() {
        // A linear function is exactly reproduced
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| 3.0 * x + 1.0).collect();
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        for x in [0.5, 1.5, 2.5] {
            assert!((interp.eval(x) - (3.0 * x + 1.0)).abs() < 1e-10, "at x={x}");
        }
    }

    #[test]
    fn quadratic_exact() {
        // y = x² through 3 points should be exact
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        for x in [0.5, 1.0, 1.5] {
            assert!((interp.eval(x) - x * x).abs() < 1e-10, "at x={x}");
        }
    }

    #[test]
    fn sin_interpolation_convergence() {
        // More nodes → better approximation of sin(x)
        let err_5 = {
            let xs: Vec<f64> = (0..=5).map(|i| std::f64::consts::PI * i as f64 / 5.0).collect();
            let ys: Vec<f64> = xs.iter().map(|x: &f64| x.sin()).collect();
            let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
            (interp.eval(std::f64::consts::PI / 3.0) - (std::f64::consts::PI / 3.0).sin()).abs()
        };
        let err_10 = {
            let xs: Vec<f64> = (0..=10).map(|i| std::f64::consts::PI * i as f64 / 10.0).collect();
            let ys: Vec<f64> = xs.iter().map(|x: &f64| x.sin()).collect();
            let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
            (interp.eval(std::f64::consts::PI / 3.0) - (std::f64::consts::PI / 3.0).sin()).abs()
        };
        assert!(err_10 < err_5, "err_10={err_10}, err_5={err_5}");
    }

    #[test]
    fn cubic_exact() {
        // y = x³ through 4 points
        let xs = vec![-1.0, 0.0, 1.0, 2.0];
        let ys: Vec<f64> = xs.iter().map(|&x: &f64| x.powi(3)).collect();
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        for x in [-0.5, 0.5, 1.5] {
            assert!((interp.eval(x) - x.powi(3)).abs() < 1e-10, "at x={x}");
        }
    }

    #[test]
    fn rejects_duplicates() {
        let xs = vec![1.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        assert!(LagrangeInterpolator::new(&xs, &ys).is_none());
    }

    #[test]
    fn rejects_too_few() {
        assert!(LagrangeInterpolator::new(&[1.0], &[2.0]).is_none());
    }

    #[test]
    fn rejects_mismatched() {
        assert!(LagrangeInterpolator::new(&[0.0, 1.0], &[0.0]).is_none());
    }

    #[test]
    fn exp_interpolation() {
        let xs = vec![0.0, 0.5, 1.0];
        let ys: Vec<f64> = xs.iter().map(|&x: &f64| x.exp()).collect();
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        // At 0.25, should be close to e^0.25
        let val = interp.eval(0.25);
        let exact = 0.25_f64.exp();
        assert!((val - exact).abs() < 0.05, "got {val}, exact {exact}");
    }

    #[test]
    fn runge_phenomenon_detection() {
        // Runge function: f(x) = 1/(1+25x²) on [-1,1]
        // With many equispaced nodes, interpolation should still match at nodes
        let n = 10;
        let xs: Vec<f64> = (0..=n).map(|i| -1.0 + 2.0 * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| 1.0 / (1.0 + 25.0 * x * x)).collect();
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        for i in 0..xs.len() {
            assert!((interp.eval(xs[i]) - ys[i]).abs() < 1e-8, "at node {i}");
        }
    }

    #[test]
    fn len_and_empty() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let interp = LagrangeInterpolator::new(&xs, &ys).unwrap();
        assert_eq!(interp.len(), 2);
        assert!(!interp.is_empty());
    }
}
