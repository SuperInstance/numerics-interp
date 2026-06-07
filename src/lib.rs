//! # numerics-interp
//!
//! Research-grade interpolation methods in pure Rust with zero external dependencies.
//!
//! ## Methods
//!
//! - **Linear** — Piecewise linear interpolation. O(n) evaluation.
//! - **Lagrange** — Polynomial interpolation through all nodes. O(n²) evaluation.
//! - **Cubic Spline** — Natural cubic spline with C² continuity. O(n) evaluation.
//! - **Barycentric** — Fast Lagrange evaluation via barycentric weights. O(n) per evaluation.
//!
//! ## Example
//!
//! ```
//! use numerics_interp::cubic_spline;
//!
//! let xs = vec![0.0, 1.0, 2.0, 3.0];
//! let ys = vec![0.0, 1.0, 4.0, 9.0]; // x²
//! let spline = cubic_spline::CubicSpline::new(&xs, &ys).unwrap();
//! assert!((spline.eval(1.5) - 2.25).abs() < 0.01);
//! ```

pub mod linear;
pub mod lagrange;
pub mod cubic_spline;
pub mod barycentric;
pub mod error;
