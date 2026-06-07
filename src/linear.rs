//! Piecewise linear interpolation.
//!
//! Given sorted nodes `(x₀,y₀),…,(xₙ,yₙ)`, the interpolant on interval `[xᵢ, xᵢ₊₁]` is:
//! ```text
//! L(x) = yᵢ + (yᵢ₊₁ - yᵢ) · (x - xᵢ) / (xᵢ₊₁ - xᵢ)
//! ```

/// Piecewise linear interpolator.
pub struct LinearInterpolator {
    xs: Vec<f64>,
    ys: Vec<f64>,
}

impl LinearInterpolator {
    /// Create a new linear interpolator from sorted nodes.
    ///
    /// # Errors
    ///
    /// Returns `None` if fewer than 2 points are given or `xs` is not sorted.
    pub fn new(xs: &[f64], ys: &[f64]) -> Option<Self> {
        if xs.len() < 2 || xs.len() != ys.len() {
            return None;
        }
        for w in xs.windows(2) {
            if w[0] >= w[1] {
                return None;
            }
        }
        Some(Self { xs: xs.to_vec(), ys: ys.to_vec() })
    }

    /// Evaluate the piecewise linear interpolant at `x`.
    ///
    /// Clamps to the boundary values if `x` is outside `[x₀, xₙ]`.
    pub fn eval(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= *self.xs.last().unwrap() {
            return *self.ys.last().unwrap();
        }
        // Binary search for interval
        let idx = match self.xs.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(i) => i,
            Err(i) => i,
        };
        // idx is either the exact match or the insertion point
        let i = if idx == 0 { 0 } else { idx - 1 };
        let i = i.min(self.xs.len() - 2);
        let t = (x - self.xs[i]) / (self.xs[i + 1] - self.xs[i]);
        self.ys[i] + t * (self.ys[i + 1] - self.ys[i])
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.eval(x)).collect()
    }

    /// Reference to the x-coordinates.
    pub fn xs(&self) -> &[f64] { &self.xs }
    /// Reference to the y-coordinates.
    pub fn ys(&self) -> &[f64] { &self.ys }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_identity() {
        // y = x
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 2.0, 3.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        for x in [0.0, 0.5, 1.0, 1.5, 2.5, 3.0] {
            assert!((interp.eval(x) - x).abs() < 1e-12, "at x={x}");
        }
    }

    #[test]
    fn linear_quadratic_approx() {
        // y = x² at nodes 0,1,2,3
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        // At x=0.5: linear gives 0.5, exact is 0.25
        let val = interp.eval(0.5);
        assert!((val - 0.5).abs() < 1e-12, "got {val}");
        // At x=1.5: linear gives 2.5, exact is 2.25
        assert!((interp.eval(1.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn linear_exact_at_nodes() {
        let xs = vec![0.0, 2.0, 5.0, 10.0];
        let ys = vec![1.0, 3.0, -1.0, 7.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        for i in 0..xs.len() {
            assert!((interp.eval(xs[i]) - ys[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn linear_clamp_below() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        assert!((interp.eval(0.0) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn linear_clamp_above() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        assert!((interp.eval(5.0) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn rejects_unsorted() {
        let xs = vec![3.0, 1.0, 2.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(LinearInterpolator::new(&xs, &ys).is_none());
    }

    #[test]
    fn rejects_too_few() {
        assert!(LinearInterpolator::new(&[1.0], &[2.0]).is_none());
        assert!(LinearInterpolator::new(&[], &[]).is_none());
    }

    #[test]
    fn rejects_mismatched_lengths() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0];
        assert!(LinearInterpolator::new(&xs, &ys).is_none());
    }

    #[test]
    fn linear_sin_approx() {
        // With many nodes, linear approximates sin well
        let n = 100;
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        for i in 0..50 {
            let x = std::f64::consts::PI * i as f64 / 50.0;
            let err = (interp.eval(x) - x.sin()).abs();
            assert!(err < 0.01, "error {err} at x={x}");
        }
    }

    #[test]
    fn eval_many() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 2.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        let result = interp.eval_many(&[0.5, 1.0, 1.5]);
        assert!((result[0] - 0.5).abs() < 1e-12);
        assert!((result[1] - 1.0).abs() < 1e-12);
        assert!((result[2] - 1.5).abs() < 1e-12);
    }

    #[test]
    fn accessors() {
        let xs = vec![0.0, 1.0];
        let ys = vec![5.0, 10.0];
        let interp = LinearInterpolator::new(&xs, &ys).unwrap();
        assert_eq!(interp.xs(), &[0.0, 1.0]);
        assert_eq!(interp.ys(), &[5.0, 10.0]);
    }
}
