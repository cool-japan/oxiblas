//! Advanced convergence monitoring for iterative solvers.
//!
//! This module provides flexible stopping criteria and convergence monitoring
//! for iterative linear solvers. Features include:
//!
//! - Multiple stopping criteria (relative, absolute, mixed)
//! - Stagnation detection
//! - Divergence detection
//! - Convergence rate estimation
//! - Residual smoothing for noisy iterations
//!
//! # Example
//!
//! ```ignore
//! use oxiblas_sparse::linalg::convergence::{ConvergenceMonitor, StoppingCriteria};
//!
//! // Create a monitor with relative tolerance
//! let criteria = StoppingCriteria::relative(1e-10);
//! let mut monitor = ConvergenceMonitor::new(criteria);
//!
//! // During iteration, update and check
//! for iter in 0..max_iter {
//!     let residual = compute_residual();
//!     monitor.update(residual, iter);
//!
//!     if monitor.has_converged() {
//!         break;
//!     }
//!     if monitor.has_stagnated() {
//!         // Handle stagnation
//!     }
//! }
//!
//! // Get detailed convergence info
//! let info = monitor.convergence_info();
//! ```

use oxiblas_core::scalar::{Field, Real, Scalar};

/// Stopping criteria for iterative solvers.
#[derive(Debug, Clone)]
pub enum StoppingCriteria<T> {
    /// Relative tolerance: ||r|| / ||b|| < tol
    Relative {
        /// Tolerance value
        tol: T,
    },
    /// Absolute tolerance: ||r|| < tol
    Absolute {
        /// Tolerance value
        tol: T,
    },
    /// Mixed tolerance: ||r|| < atol + rtol * ||b||
    Mixed {
        /// Absolute tolerance
        atol: T,
        /// Relative tolerance
        rtol: T,
    },
    /// Relative residual decrease: ||r_k|| / ||r_0|| < tol
    RelativeResidualDecrease {
        /// Tolerance value
        tol: T,
    },
    /// Energy norm for SPD systems: sqrt(r^T * A^{-1} * r) < tol
    /// (approximated using preconditioner if available)
    EnergyNorm {
        /// Tolerance value
        tol: T,
    },
    /// Combined criteria with logical AND
    And(Box<StoppingCriteria<T>>, Box<StoppingCriteria<T>>),
    /// Combined criteria with logical OR
    Or(Box<StoppingCriteria<T>>, Box<StoppingCriteria<T>>),
}

impl<T: Scalar<Real = T> + Clone> StoppingCriteria<T> {
    /// Creates relative tolerance criterion.
    pub fn relative(tol: T) -> Self {
        Self::Relative { tol }
    }

    /// Creates absolute tolerance criterion.
    pub fn absolute(tol: T) -> Self {
        Self::Absolute { tol }
    }

    /// Creates mixed tolerance criterion.
    pub fn mixed(atol: T, rtol: T) -> Self {
        Self::Mixed { atol, rtol }
    }

    /// Creates relative residual decrease criterion.
    pub fn relative_decrease(tol: T) -> Self {
        Self::RelativeResidualDecrease { tol }
    }

    /// Creates energy norm criterion.
    pub fn energy_norm(tol: T) -> Self {
        Self::EnergyNorm { tol }
    }

    /// Combines two criteria with AND.
    pub fn and(self, other: Self) -> Self {
        Self::And(Box::new(self), Box::new(other))
    }

    /// Combines two criteria with OR.
    pub fn or(self, other: Self) -> Self {
        Self::Or(Box::new(self), Box::new(other))
    }

    /// Checks if convergence criteria is satisfied.
    pub fn is_satisfied(&self, residual: T, rhs_norm: T, initial_residual: Option<T>) -> bool
    where
        T: PartialOrd + Clone + Field,
    {
        match self {
            Self::Relative { tol } => {
                if rhs_norm <= <T as Scalar>::epsilon() {
                    residual <= *tol
                } else {
                    residual / rhs_norm.clone() <= *tol
                }
            }
            Self::Absolute { tol } => residual <= *tol,
            Self::Mixed { atol, rtol } => residual <= atol.clone() + rtol.clone() * rhs_norm,
            Self::RelativeResidualDecrease { tol } => {
                if let Some(init_res) = initial_residual {
                    if init_res <= <T as Scalar>::epsilon() {
                        true
                    } else {
                        residual / init_res <= *tol
                    }
                } else {
                    false
                }
            }
            Self::EnergyNorm { tol } => {
                // For energy norm, the caller should provide the energy norm as residual
                residual <= *tol
            }
            Self::And(a, b) => {
                a.is_satisfied(residual.clone(), rhs_norm.clone(), initial_residual.clone())
                    && b.is_satisfied(residual, rhs_norm, initial_residual)
            }
            Self::Or(a, b) => {
                a.is_satisfied(residual.clone(), rhs_norm.clone(), initial_residual.clone())
                    || b.is_satisfied(residual, rhs_norm, initial_residual)
            }
        }
    }
}

/// Convergence status of an iterative solver.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// Still iterating
    Iterating,
    /// Converged to desired tolerance
    Converged,
    /// Stagnated (residual not decreasing significantly)
    Stagnated,
    /// Diverged (residual increasing)
    Diverged,
    /// Breakdown detected
    Breakdown,
    /// Maximum iterations reached
    MaxIterationsReached,
}

/// Detailed information about convergence behavior.
#[derive(Debug, Clone)]
pub struct ConvergenceInfo<T> {
    /// Current status
    pub status: ConvergenceStatus,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm
    pub final_residual: T,
    /// Initial residual norm
    pub initial_residual: T,
    /// RHS norm (||b||)
    pub rhs_norm: T,
    /// Estimated convergence rate (geometric mean of residual ratios)
    pub convergence_rate: Option<T>,
    /// Number of stagnation detections
    pub stagnation_count: usize,
    /// Full residual history
    pub residual_history: Vec<T>,
    /// Smoothed residual history (for noisy iterations)
    pub smoothed_history: Vec<T>,
}

/// Configuration for convergence monitoring.
#[derive(Debug, Clone)]
pub struct ConvergenceConfig<T> {
    /// Stopping criteria
    pub criteria: StoppingCriteria<T>,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Stagnation tolerance (relative residual change below this is considered stagnation)
    pub stagnation_tol: T,
    /// Number of consecutive stagnation iterations before declaring stagnation
    pub stagnation_window: usize,
    /// Divergence factor (if residual increases by this factor, declare divergence)
    pub divergence_factor: T,
    /// Smoothing window for residual averaging
    pub smoothing_window: usize,
    /// Enable detailed logging
    pub verbose: bool,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> Default for ConvergenceConfig<T> {
    fn default() -> Self {
        Self {
            criteria: StoppingCriteria::relative(
                T::from_f64(1e-10).unwrap_or_else(<T as Scalar>::epsilon),
            ),
            max_iterations: 1000,
            stagnation_tol: T::from_f64(1e-12).unwrap_or_else(<T as Scalar>::epsilon),
            stagnation_window: 5,
            divergence_factor: T::from_f64(1e6).unwrap_or_else(|| {
                let mut v = T::one();
                for _ in 0..6 {
                    v = v.clone() * T::from_f64(10.0).unwrap_or_else(T::one);
                }
                v
            }),
            smoothing_window: 3,
            verbose: false,
        }
    }
}

/// Convergence monitor for iterative solvers.
///
/// Tracks residual history, detects stagnation and divergence,
/// and computes convergence rate estimates.
#[derive(Debug, Clone)]
pub struct ConvergenceMonitor<T> {
    config: ConvergenceConfig<T>,
    residual_history: Vec<T>,
    smoothed_history: Vec<T>,
    rhs_norm: Option<T>,
    stagnation_count: usize,
    status: ConvergenceStatus,
    iteration: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real + PartialOrd> ConvergenceMonitor<T> {
    /// Creates a new convergence monitor with the given stopping criteria.
    pub fn new(criteria: StoppingCriteria<T>) -> Self {
        Self::with_config(ConvergenceConfig {
            criteria,
            ..Default::default()
        })
    }

    /// Creates a new convergence monitor with full configuration.
    pub fn with_config(config: ConvergenceConfig<T>) -> Self {
        Self {
            config,
            residual_history: Vec::new(),
            smoothed_history: Vec::new(),
            rhs_norm: None,
            stagnation_count: 0,
            status: ConvergenceStatus::Iterating,
            iteration: 0,
        }
    }

    /// Sets the RHS norm (||b||) for relative tolerance calculation.
    pub fn set_rhs_norm(&mut self, norm: T) {
        self.rhs_norm = Some(norm);
    }

    /// Updates the monitor with a new residual value.
    ///
    /// Returns true if the solver should continue iterating.
    pub fn update(&mut self, residual: T, iteration: usize) -> bool {
        self.iteration = iteration;
        self.residual_history.push(residual.clone());

        // Update smoothed history
        let smoothed = self.compute_smoothed_residual();
        self.smoothed_history.push(smoothed.clone());

        // Check for convergence
        let rhs_norm = self.rhs_norm.clone().unwrap_or_else(T::one);
        let initial_residual = self.residual_history.first().cloned();

        if self
            .config
            .criteria
            .is_satisfied(residual.clone(), rhs_norm, initial_residual)
        {
            self.status = ConvergenceStatus::Converged;
            return false;
        }

        // Check for divergence
        if self.check_divergence() {
            self.status = ConvergenceStatus::Diverged;
            return false;
        }

        // Check for stagnation
        if self.check_stagnation() {
            self.stagnation_count += 1;
            if self.stagnation_count >= self.config.stagnation_window {
                self.status = ConvergenceStatus::Stagnated;
                return false;
            }
        } else {
            self.stagnation_count = 0;
        }

        // Check for max iterations
        if iteration >= self.config.max_iterations {
            self.status = ConvergenceStatus::MaxIterationsReached;
            return false;
        }

        true
    }

    /// Records a breakdown event.
    pub fn record_breakdown(&mut self) {
        self.status = ConvergenceStatus::Breakdown;
    }

    /// Returns true if convergence has been achieved.
    pub fn has_converged(&self) -> bool {
        self.status == ConvergenceStatus::Converged
    }

    /// Returns true if the solver has stagnated.
    pub fn has_stagnated(&self) -> bool {
        self.status == ConvergenceStatus::Stagnated
    }

    /// Returns true if the solver has diverged.
    pub fn has_diverged(&self) -> bool {
        self.status == ConvergenceStatus::Diverged
    }

    /// Returns the current status.
    pub fn status(&self) -> &ConvergenceStatus {
        &self.status
    }

    /// Returns the current iteration count.
    pub fn iterations(&self) -> usize {
        self.iteration
    }

    /// Returns the current residual norm.
    pub fn current_residual(&self) -> Option<&T> {
        self.residual_history.last()
    }

    /// Returns the full residual history.
    pub fn residual_history(&self) -> &[T] {
        &self.residual_history
    }

    /// Returns the smoothed residual history.
    pub fn smoothed_history(&self) -> &[T] {
        &self.smoothed_history
    }

    /// Computes the estimated convergence rate.
    ///
    /// Returns the geometric mean of consecutive residual ratios.
    pub fn convergence_rate(&self) -> Option<T> {
        if self.residual_history.len() < 2 {
            return None;
        }

        let _n = self.residual_history.len() - 1;
        let mut product = T::one();
        let mut count = 0;

        for i in 1..self.residual_history.len() {
            let prev = &self.residual_history[i - 1];
            let curr = &self.residual_history[i];

            if Scalar::abs(prev.clone()) > <T as Scalar>::epsilon() {
                let ratio = Scalar::abs(curr.clone()) / Scalar::abs(prev.clone());
                product = product * ratio;
                count += 1;
            }
        }

        if count > 0 {
            // Geometric mean: product^(1/count)
            let exp = T::one() / T::from_usize(count).unwrap_or_else(T::one);
            Some(Real::powf(product, exp))
        } else {
            None
        }
    }

    /// Returns detailed convergence information.
    pub fn convergence_info(&self) -> ConvergenceInfo<T> {
        ConvergenceInfo {
            status: self.status.clone(),
            iterations: self.iteration,
            final_residual: self
                .residual_history
                .last()
                .cloned()
                .unwrap_or_else(T::zero),
            initial_residual: self
                .residual_history
                .first()
                .cloned()
                .unwrap_or_else(T::zero),
            rhs_norm: self.rhs_norm.clone().unwrap_or_else(T::one),
            convergence_rate: self.convergence_rate(),
            stagnation_count: self.stagnation_count,
            residual_history: self.residual_history.clone(),
            smoothed_history: self.smoothed_history.clone(),
        }
    }

    /// Computes smoothed residual using moving average.
    fn compute_smoothed_residual(&self) -> T {
        let window = self
            .config
            .smoothing_window
            .min(self.residual_history.len());
        if window == 0 {
            return T::zero();
        }

        let start = self.residual_history.len() - window;
        let mut sum = T::zero();
        for i in start..self.residual_history.len() {
            sum = sum + self.residual_history[i].clone();
        }

        sum / T::from_usize(window).unwrap_or_else(T::one)
    }

    /// Checks if the solver has diverged.
    fn check_divergence(&self) -> bool {
        if self.residual_history.len() < 2 {
            return false;
        }

        let initial = &self.residual_history[0];
        let current = self
            .residual_history
            .last()
            .expect("collection should be non-empty");

        if Scalar::abs(initial.clone()) > <T as Scalar>::epsilon() {
            Scalar::abs(current.clone()) / Scalar::abs(initial.clone())
                > self.config.divergence_factor
        } else {
            Scalar::abs(current.clone()) > self.config.divergence_factor
        }
    }

    /// Checks if the solver is stagnating.
    fn check_stagnation(&self) -> bool {
        if self.residual_history.len() < 2 {
            return false;
        }

        let prev = &self.residual_history[self.residual_history.len() - 2];
        let curr = self
            .residual_history
            .last()
            .expect("collection should be non-empty");

        if Scalar::abs(prev.clone()) > <T as Scalar>::epsilon() {
            let relative_change =
                Scalar::abs(curr.clone() - prev.clone()) / Scalar::abs(prev.clone());
            relative_change <= self.config.stagnation_tol
        } else {
            Scalar::abs(curr.clone() - prev.clone()) <= self.config.stagnation_tol
        }
    }

    /// Resets the monitor for a new solve.
    pub fn reset(&mut self) {
        self.residual_history.clear();
        self.smoothed_history.clear();
        self.rhs_norm = None;
        self.stagnation_count = 0;
        self.status = ConvergenceStatus::Iterating;
        self.iteration = 0;
    }
}

/// Estimates the number of iterations to reach target tolerance.
///
/// Based on current convergence rate, estimates how many more iterations
/// are needed to reduce residual to target tolerance.
pub fn estimate_iterations_to_convergence<T>(
    current_residual: T,
    target_residual: T,
    convergence_rate: T,
) -> Option<usize>
where
    T: Scalar<Real = T> + Clone + Field + Real + PartialOrd,
{
    if convergence_rate >= T::one() {
        // Not converging
        return None;
    }

    if current_residual <= target_residual {
        return Some(0);
    }

    if Scalar::abs(convergence_rate.clone()) <= <T as Scalar>::epsilon() {
        return None;
    }

    // residual_k = residual_0 * rate^k
    // target = current * rate^k
    // k = log(target/current) / log(rate)
    let ratio = target_residual / current_residual;
    let log_ratio = Real::ln(ratio);
    let log_rate = Real::ln(convergence_rate);

    if Scalar::abs(log_rate.clone()) <= <T as Scalar>::epsilon() {
        return None;
    }

    let k = log_ratio / log_rate;
    if k < T::zero() {
        None
    } else {
        Some(Real::ceil(k).to_usize().unwrap_or(usize::MAX))
    }
}

/// Computes the asymptotic convergence factor from residual history.
///
/// Uses the last few iterations to estimate the convergence factor
/// more accurately than using all iterations.
pub fn asymptotic_convergence_factor<T>(residual_history: &[T], window: usize) -> Option<T>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    if residual_history.len() < 2 {
        return None;
    }

    let window = window.min(residual_history.len() - 1);
    let start = residual_history.len() - 1 - window;

    let mut product = T::one();
    let mut count = 0;

    for i in (start + 1)..residual_history.len() {
        let prev = &residual_history[i - 1];
        let curr = &residual_history[i];

        if Scalar::abs(prev.clone()) > <T as Scalar>::epsilon() {
            let ratio = Scalar::abs(curr.clone()) / Scalar::abs(prev.clone());
            product = product * ratio;
            count += 1;
        }
    }

    if count > 0 {
        let exp = T::one() / T::from_usize(count).unwrap_or_else(T::one);
        Some(Real::powf(product, exp))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stopping_criteria_relative() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::relative(1e-6);

        // Should converge when relative residual is small enough
        assert!(criteria.is_satisfied(1e-7, 1.0, None));
        assert!(!criteria.is_satisfied(1e-5, 1.0, None));

        // Scale with RHS norm
        assert!(criteria.is_satisfied(1e-4, 100.0, None));
        assert!(!criteria.is_satisfied(1e-3, 100.0, None));
    }

    #[test]
    fn test_stopping_criteria_absolute() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::absolute(1e-8);

        assert!(criteria.is_satisfied(1e-9, 1.0, None));
        assert!(!criteria.is_satisfied(1e-7, 1.0, None));

        // RHS norm shouldn't matter for absolute
        assert!(criteria.is_satisfied(1e-9, 100.0, None));
    }

    #[test]
    fn test_stopping_criteria_mixed() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::mixed(1e-10, 1e-6);

        // atol + rtol * ||b|| = 1e-10 + 1e-6 * 1.0 ≈ 1e-6
        assert!(criteria.is_satisfied(1e-7, 1.0, None));
        assert!(!criteria.is_satisfied(1e-5, 1.0, None));
    }

    #[test]
    fn test_stopping_criteria_relative_decrease() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::relative_decrease(1e-6);

        assert!(criteria.is_satisfied(1e-7, 1.0, Some(1.0)));
        assert!(!criteria.is_satisfied(1e-5, 1.0, Some(1.0)));

        // Without initial residual, should not converge
        assert!(!criteria.is_satisfied(1e-10, 1.0, None));
    }

    #[test]
    fn test_stopping_criteria_combined() {
        let rel: StoppingCriteria<f64> = StoppingCriteria::relative(1e-6);
        let abs: StoppingCriteria<f64> = StoppingCriteria::absolute(1e-10);

        // AND: both must be satisfied
        let and_criteria = rel.clone().and(abs.clone());
        assert!(and_criteria.is_satisfied(1e-11, 1.0, None)); // Both satisfied
        assert!(!and_criteria.is_satisfied(1e-7, 1.0, None)); // Only relative satisfied
        assert!(!and_criteria.is_satisfied(1e-9, 1e10, None)); // Only absolute satisfied (rel fails)

        // OR: at least one must be satisfied
        let or_criteria = StoppingCriteria::relative(1e-6).or(StoppingCriteria::absolute(1e-10));
        assert!(or_criteria.is_satisfied(1e-7, 1.0, None)); // Relative satisfied
        assert!(or_criteria.is_satisfied(1e-11, 1e10, None)); // Absolute satisfied
        // Note: 1e-5 / 1e10 = 1e-15 <= 1e-6, so relative IS satisfied
        // Use residual 0.1 with rhs_norm 100: 0.1 / 100 = 1e-3 > 1e-6 (relative fails)
        // and 0.1 > 1e-10 (absolute fails)
        assert!(!or_criteria.is_satisfied(0.1, 100.0, None)); // Neither satisfied
    }

    #[test]
    fn test_convergence_monitor_basic() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::absolute(1e-8);
        let mut monitor = ConvergenceMonitor::new(criteria);

        // Simulate converging iteration
        assert!(monitor.update(1.0, 0));
        assert!(monitor.update(0.1, 1));
        assert!(monitor.update(0.01, 2));
        assert!(monitor.update(1e-5, 3));
        assert!(!monitor.update(1e-9, 4)); // Should converge

        assert!(monitor.has_converged());
        assert_eq!(monitor.iterations(), 4);
    }

    #[test]
    fn test_convergence_monitor_divergence() {
        let config = ConvergenceConfig {
            criteria: StoppingCriteria::absolute(1e-10),
            divergence_factor: 1e3,
            ..Default::default()
        };
        let mut monitor = ConvergenceMonitor::with_config(config);

        // Simulate diverging iteration
        assert!(monitor.update(1.0, 0));
        assert!(monitor.update(10.0, 1));
        assert!(monitor.update(100.0, 2));
        assert!(!monitor.update(1e4, 3)); // Should diverge

        assert!(monitor.has_diverged());
    }

    #[test]
    fn test_convergence_monitor_stagnation() {
        let config = ConvergenceConfig {
            criteria: StoppingCriteria::absolute(1e-10),
            stagnation_tol: 1e-12,
            stagnation_window: 3,
            ..Default::default()
        };
        let mut monitor = ConvergenceMonitor::with_config(config);

        // Simulate stagnating iteration
        monitor.update(1.0, 0);
        monitor.update(0.5, 1);
        monitor.update(0.5, 2); // Stagnation starts
        monitor.update(0.5, 3);
        monitor.update(0.5, 4);
        let should_stop = !monitor.update(0.5, 5); // Should stagnate

        assert!(should_stop);
        assert!(monitor.has_stagnated());
    }

    #[test]
    fn test_convergence_rate_estimation() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::absolute(1e-10);
        let mut monitor = ConvergenceMonitor::new(criteria);

        // Linear convergence with rate 0.5
        for i in 0..10 {
            let residual = 0.5f64.powi(i);
            monitor.update(residual, i as usize);
        }

        let rate = monitor.convergence_rate().unwrap();
        assert!((rate - 0.5).abs() < 0.01, "Expected rate ~0.5, got {rate}");
    }

    #[test]
    fn test_estimate_iterations() {
        let estimate = estimate_iterations_to_convergence(1.0, 1e-10, 0.5);
        assert!(estimate.is_some());

        // 0.5^k < 1e-10 => k > log(1e-10)/log(0.5) ≈ 33.2
        let k = estimate.unwrap();
        assert!((33..=35).contains(&k), "Expected ~34 iterations, got {k}");

        // Non-converging case
        let no_converge = estimate_iterations_to_convergence(1.0, 1e-10, 1.1);
        assert!(no_converge.is_none());
    }

    #[test]
    fn test_asymptotic_convergence_factor() {
        let history: Vec<f64> = (0..20).map(|i| 0.9f64.powi(i)).collect();

        let factor = asymptotic_convergence_factor(&history, 5).unwrap();
        assert!(
            (factor - 0.9).abs() < 0.01,
            "Expected factor ~0.9, got {factor}"
        );
    }

    #[test]
    fn test_convergence_info() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::absolute(1e-8);
        let mut monitor = ConvergenceMonitor::new(criteria);
        monitor.set_rhs_norm(10.0);

        for i in 0..5 {
            monitor.update(10.0 * 0.1f64.powi(i), i as usize);
        }

        let info = monitor.convergence_info();
        assert_eq!(info.iterations, 4);
        assert_eq!(info.residual_history.len(), 5);
        assert!((info.rhs_norm - 10.0).abs() < 1e-10);
        assert!(info.convergence_rate.is_some());
    }

    #[test]
    fn test_monitor_reset() {
        let criteria: StoppingCriteria<f64> = StoppingCriteria::absolute(1e-8);
        let mut monitor = ConvergenceMonitor::new(criteria);

        monitor.update(1.0, 0);
        monitor.update(0.5, 1);
        assert_eq!(monitor.residual_history().len(), 2);

        monitor.reset();
        assert_eq!(monitor.residual_history().len(), 0);
        assert_eq!(*monitor.status(), ConvergenceStatus::Iterating);
    }
}
