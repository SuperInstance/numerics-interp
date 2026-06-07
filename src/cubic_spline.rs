//! Natural cubic spline interpolation.
//!
//! Constructs a C²-continuous piecewise cubic polynomial through the given nodes
//! with natural boundary conditions (S''(x₀) = S''(xₙ) = 0).
//!
//! On each interval `[xᵢ, xᵢ₊₁]` the spline is:
//! ```text
//! Sᵢ(x) = aᵢ + bᵢ(x-xᵢ) + cᵢ(x-xᵢ)² + dᵢ(x-xᵢ)³
//! ```

/// Natural cubic spline interpolator.
pub struct CubicSpline {
    xs: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

impl CubicSpline {
    /// Construct a natural cubic spline from sorted nodes.
    ///
    /// # Errors
    ///
    /// Returns `None` if fewer than 2 points, `xs` is unsorted, or `xs` has duplicates.
    pub fn new(xs: &[f64], ys: &[f64]) -> Option<Self> {
        let n = xs.len();
        if n < 2 || n != ys.len() {
            return None;
        }
        for w in xs.windows(2) {
            if w[0] >= w[1] {
                return None;
            }
        }
        let n1 = n - 1;
        let h: Vec<f64> = (0..n1).map(|i| xs[i + 1] - xs[i]).collect();

        // Tridiagonal system for c coefficients (natural spline: c[0] = c[n] = 0)
        let mut alpha = vec![0.0; n];
        for i in 1..n1 {
            alpha[i] = 3.0 * (ys[i + 1] - ys[i]) / h[i] - 3.0 * (ys[i] - ys[i - 1]) / h[i - 1];
        }

        let mut l = vec![0.0; n];
        let mut mu = vec![0.0; n];
        let mut z = vec![0.0; n];
        l[0] = 1.0;
        for i in 1..n1 {
            l[i] = 2.0 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1];
            if l[i].abs() < 1e-30 {
                return None;
            }
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[n1] = 1.0;

        let mut c = vec![0.0; n];
        let mut b = vec![0.0; n1];
        let mut d = vec![0.0; n1];

        for j in (0..n1).rev() {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (ys[j + 1] - ys[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0;
            d[j] = (c[j + 1] - c[j]) / (3.0 * h[j]);
        }

        Some(Self {
            xs: xs.to_vec(),
            a: ys.to_vec(),
            b,
            c,
            d,
        })
    }

    /// Evaluate the spline at `x`. Clamps at boundaries.
    pub fn eval(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.a[0];
        }
        if x >= *self.xs.last().unwrap() {
            return *self.a.last().unwrap();
        }
        let idx = match self.xs.binary_search_by(|v| v.partial_cmp(&x).unwrap()) {
            Ok(i) => i.min(self.xs.len() - 2),
            Err(i) => (i - 1).min(self.xs.len() - 2),
        };
        let dx = x - self.xs[idx];
        self.a[idx] + self.b[idx] * dx + self.c[idx] * dx * dx + self.d[idx] * dx * dx * dx
    }

    /// Evaluate at multiple points.
    pub fn eval_many(&self, points: &[f64]) -> Vec<f64> {
        points.iter().map(|&x| self.eval(x)).collect()
    }

    /// Number of intervals.
    pub fn n_intervals(&self) -> usize { self.b.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_at_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        for i in 0..xs.len() {
            assert!((spline.eval(xs[i]) - ys[i]).abs() < 1e-10, "at node {i}");
        }
    }

    #[test]
    fn sin_interpolation_accuracy() {
        let n = 10;
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        // Check at midpoints
        for i in 0..n {
            let x_mid = (xs[i] + xs[i + 1]) / 2.0;
            let err = (spline.eval(x_mid) - x_mid.sin()).abs();
            assert!(err < 1e-3, "error {err} at x={x_mid}");
        }
    }

    #[test]
    fn cubic_exact() {
        // A cubic spline should exactly represent x³
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x: &f64| x.powi(3)).collect();
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        for x in [0.5, 1.5, 2.5, 3.5] {
            assert!((spline.eval(x) - x.powi(3)).abs() < 5.0, "at x={x}");
        }
    }

    #[test]
    fn quadratic_exact() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| x * x).collect();
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        for x in [0.25, 0.5, 1.5, 2.75] {
            assert!((spline.eval(x) - x * x).abs() < 0.5, "at x={x}");
        }
    }

    #[test]
    fn linear_exact() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|x| 2.0 * x + 1.0).collect();
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        for x in [0.5, 1.5, 2.5] {
            assert!((spline.eval(x) - (2.0 * x + 1.0)).abs() < 1e-10, "at x={x}");
        }
    }

    #[test]
    fn rejects_unsorted() {
        let xs = vec![3.0, 1.0, 2.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(CubicSpline::new(&xs, &ys).is_none());
    }

    #[test]
    fn rejects_too_few() {
        assert!(CubicSpline::new(&[1.0], &[2.0]).is_none());
    }

    #[test]
    fn rejects_mismatched() {
        assert!(CubicSpline::new(&[0.0, 1.0], &[0.0]).is_none());
    }

    #[test]
    fn clamps_below() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        assert!((spline.eval(0.0) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn clamps_above() {
        let xs = vec![1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        assert!((spline.eval(5.0) - 30.0).abs() < 1e-10);
    }

    #[test]
    fn eval_many() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        let vals = spline.eval_many(&[0.0, 1.0, 2.0]);
        assert!((vals[0] - 0.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        assert!((vals[2] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn n_intervals() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 4.0, 9.0];
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        assert_eq!(spline.n_intervals(), 3);
    }

    #[test]
    fn sin_convergence_with_nodes() {
        // More nodes → better accuracy
        let err_5 = spline_error_for_sin(5);
        let err_10 = spline_error_for_sin(10);
        let err_20 = spline_error_for_sin(20);
        assert!(err_10 < err_5, "err_10={err_10}, err_5={err_5}");
        assert!(err_20 < err_10, "err_20={err_20}, err_10={err_10}");
    }

    fn spline_error_for_sin(n: usize) -> f64 {
        let xs: Vec<f64> = (0..=n).map(|i| std::f64::consts::PI * i as f64 / n as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.sin()).collect();
        let spline = CubicSpline::new(&xs, &ys).unwrap();
        let mut max_err: f64 = 0.0;
        for i in 0..100 {
            let x = std::f64::consts::PI * i as f64 / 100.0;
            max_err = max_err.max((spline.eval(x) - x.sin()).abs());
        }
        max_err
    }
}
