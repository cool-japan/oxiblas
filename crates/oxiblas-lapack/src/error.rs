//! Standardized error codes for LAPACK operations.
//!
//! This module provides a unified error system compatible with LAPACK's INFO
//! parameter conventions while providing rich Rust error handling.
//!
//! # LAPACK INFO Convention
//!
//! In LAPACK, the INFO parameter indicates the operation status:
//! - `INFO = 0`: Success
//! - `INFO > 0`: Algorithmic failure (e.g., singular matrix, non-convergence)
//! - `INFO < 0`: Invalid argument (`-INFO` is the argument number)
//!
//! This module maps Rust errors to these conventions and back.
//!
//! # Usage
//!
//! ```ignore
//! use oxiblas_lapack::error::{LapackError, ErrorCode, InfoCode};
//!
//! // Create from info code
//! let err = LapackError::from_info(-3);  // Invalid argument 3
//!
//! // Get info code
//! let info = err.to_info();  // Returns -3
//!
//! // Rich error information
//! println!("Error: {}", err);
//! ```

use std::fmt;

// ============================================================================
// INFO Code Type
// ============================================================================

/// LAPACK-style INFO code.
///
/// Represents the status of a LAPACK operation:
/// - Zero indicates success
/// - Positive indicates algorithmic failure (index of problem)
/// - Negative indicates invalid argument (-value is argument number)
pub type InfoCode = i32;

/// Success info code.
pub const INFO_SUCCESS: InfoCode = 0;

// ============================================================================
// Error Categories
// ============================================================================

/// High-level error categories for LAPACK operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// Operation completed successfully.
    Success,

    /// Input validation error (dimensions, values, etc.).
    InvalidInput,

    /// Matrix is singular or nearly singular.
    Singular,

    /// Algorithm did not converge.
    NotConverged,

    /// Numerical issues (overflow, underflow, NaN).
    Numerical,

    /// Memory allocation failure.
    Allocation,

    /// Workspace too small.
    InsufficientWorkspace,

    /// Matrix not positive definite.
    NotPositiveDefinite,

    /// Internal error (should not happen).
    Internal,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::InvalidInput => write!(f, "invalid input"),
            Self::Singular => write!(f, "singular matrix"),
            Self::NotConverged => write!(f, "algorithm did not converge"),
            Self::Numerical => write!(f, "numerical issue"),
            Self::Allocation => write!(f, "memory allocation failed"),
            Self::InsufficientWorkspace => write!(f, "insufficient workspace"),
            Self::NotPositiveDefinite => write!(f, "matrix not positive definite"),
            Self::Internal => write!(f, "internal error"),
        }
    }
}

// ============================================================================
// Error Codes
// ============================================================================

/// Specific error codes for LAPACK operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ErrorCode {
    // Success
    /// Operation completed successfully.
    Success,

    // Input Errors (-1 to -99 in LAPACK convention)
    /// Matrix dimensions are invalid (zero, negative, or mismatched).
    InvalidDimension {
        /// Which argument had the invalid dimension.
        argument: usize,
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },

    /// Leading dimension is too small.
    InvalidLeadingDimension {
        /// The argument number.
        argument: usize,
        /// Minimum required value.
        minimum: usize,
        /// Actual value.
        actual: usize,
    },

    /// Invalid argument value.
    InvalidArgument {
        /// The argument number.
        argument: usize,
        /// Description of the issue.
        reason: &'static str,
    },

    /// Matrix is not square when square is required.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },

    /// Workspace is too small.
    WorkspaceTooSmall {
        /// Required workspace size.
        required: usize,
        /// Provided workspace size.
        provided: usize,
    },

    // Algorithmic Errors (positive INFO)
    /// Matrix is exactly singular.
    Singular {
        /// Index where singularity was detected (0-based).
        index: usize,
    },

    /// Matrix is nearly singular (ill-conditioned).
    NearlySingular {
        /// Estimated reciprocal condition number.
        rcond: f64,
    },

    /// Iterative algorithm did not converge.
    NotConverged {
        /// Number of iterations performed.
        iterations: usize,
        /// Index where convergence failed (if applicable).
        index: Option<usize>,
    },

    /// Matrix is not positive definite.
    NotPositiveDefinite {
        /// Index where failure was detected.
        index: usize,
    },

    /// Eigenvalue computation failed.
    EigenvalueFailure {
        /// Number of eigenvalues that failed to converge.
        num_failed: usize,
    },

    /// SVD computation failed.
    SvdFailure {
        /// Number of superdiagonals that did not converge.
        num_failed: usize,
    },

    // Numerical Errors
    /// Overflow occurred during computation.
    Overflow,

    /// Underflow occurred during computation.
    Underflow,

    /// NaN or Inf encountered in input or output.
    NaNOrInf,

    // Resource Errors
    /// Memory allocation failed.
    AllocationFailed,

    // Internal Errors
    /// Internal error (bug in implementation).
    Internal {
        /// Description of the error.
        description: &'static str,
    },
}

impl ErrorCode {
    /// Returns the error category.
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::Success => ErrorCategory::Success,
            Self::InvalidDimension { .. }
            | Self::InvalidLeadingDimension { .. }
            | Self::InvalidArgument { .. }
            | Self::NotSquare { .. } => ErrorCategory::InvalidInput,
            Self::WorkspaceTooSmall { .. } => ErrorCategory::InsufficientWorkspace,
            Self::Singular { .. } | Self::NearlySingular { .. } => ErrorCategory::Singular,
            Self::NotConverged { .. }
            | Self::EigenvalueFailure { .. }
            | Self::SvdFailure { .. } => ErrorCategory::NotConverged,
            Self::NotPositiveDefinite { .. } => ErrorCategory::NotPositiveDefinite,
            Self::Overflow | Self::Underflow | Self::NaNOrInf => ErrorCategory::Numerical,
            Self::AllocationFailed => ErrorCategory::Allocation,
            Self::Internal { .. } => ErrorCategory::Internal,
        }
    }

    /// Returns true if this is a success code.
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Returns true if this is an input validation error.
    pub fn is_input_error(&self) -> bool {
        self.category() == ErrorCategory::InvalidInput
    }

    /// Returns true if this indicates a singular matrix.
    pub fn is_singular(&self) -> bool {
        self.category() == ErrorCategory::Singular
    }

    /// Converts to LAPACK-style INFO code.
    pub fn to_info(&self) -> InfoCode {
        match self {
            Self::Success => 0,
            // Input errors are negative (argument number)
            Self::InvalidDimension { argument, .. } => -(*argument as i32),
            Self::InvalidLeadingDimension { argument, .. } => -(*argument as i32),
            Self::InvalidArgument { argument, .. } => -(*argument as i32),
            Self::NotSquare { .. } => -1, // Usually first matrix argument
            Self::WorkspaceTooSmall { .. } => -100, // LAPACK uses large negative for workspace
            // Algorithmic errors are positive
            Self::Singular { index } => (*index as i32) + 1,
            Self::NearlySingular { .. } => 1,
            Self::NotConverged { index, .. } => index.map(|i| i as i32 + 1).unwrap_or(1),
            Self::NotPositiveDefinite { index } => (*index as i32) + 1,
            Self::EigenvalueFailure { num_failed } => *num_failed as i32,
            Self::SvdFailure { num_failed } => *num_failed as i32,
            // Numerical errors
            Self::Overflow => i32::MAX - 1,
            Self::Underflow => i32::MAX - 2,
            Self::NaNOrInf => i32::MAX - 3,
            // Resource errors
            Self::AllocationFailed => i32::MIN + 1,
            // Internal errors
            Self::Internal { .. } => i32::MIN,
        }
    }

    /// Creates an error code from LAPACK-style INFO.
    pub fn from_info(info: InfoCode) -> Self {
        if info == 0 {
            Self::Success
        } else if info < 0 {
            Self::InvalidArgument {
                argument: (-info) as usize,
                reason: "invalid argument",
            }
        } else {
            Self::Singular {
                index: (info - 1) as usize,
            }
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::InvalidDimension {
                argument,
                expected,
                actual,
            } => write!(
                f,
                "invalid dimension for argument {}: expected {}, got {}",
                argument, expected, actual
            ),
            Self::InvalidLeadingDimension {
                argument,
                minimum,
                actual,
            } => write!(
                f,
                "leading dimension {} for argument {} is less than minimum {}",
                actual, argument, minimum
            ),
            Self::InvalidArgument { argument, reason } => {
                write!(f, "invalid argument {}: {}", argument, reason)
            }
            Self::NotSquare { nrows, ncols } => {
                write!(f, "matrix is not square: {}×{}", nrows, ncols)
            }
            Self::WorkspaceTooSmall { required, provided } => {
                write!(
                    f,
                    "workspace too small: required {}, provided {}",
                    required, provided
                )
            }
            Self::Singular { index } => {
                write!(f, "matrix is singular at index {}", index)
            }
            Self::NearlySingular { rcond } => {
                write!(f, "matrix is nearly singular (rcond = {:.2e})", rcond)
            }
            Self::NotConverged { iterations, index } => {
                if let Some(idx) = index {
                    write!(
                        f,
                        "algorithm did not converge after {} iterations at index {}",
                        iterations, idx
                    )
                } else {
                    write!(
                        f,
                        "algorithm did not converge after {} iterations",
                        iterations
                    )
                }
            }
            Self::NotPositiveDefinite { index } => {
                write!(f, "matrix is not positive definite at index {}", index)
            }
            Self::EigenvalueFailure { num_failed } => {
                write!(f, "{} eigenvalue(s) failed to converge", num_failed)
            }
            Self::SvdFailure { num_failed } => {
                write!(f, "{} superdiagonal(s) did not converge", num_failed)
            }
            Self::Overflow => write!(f, "overflow occurred"),
            Self::Underflow => write!(f, "underflow occurred"),
            Self::NaNOrInf => write!(f, "NaN or infinity encountered"),
            Self::AllocationFailed => write!(f, "memory allocation failed"),
            Self::Internal { description } => {
                write!(f, "internal error: {}", description)
            }
        }
    }
}

// ============================================================================
// LapackError Type
// ============================================================================

/// Unified error type for all LAPACK operations.
///
/// This type provides:
/// - LAPACK-compatible INFO codes
/// - Rich error information with context
/// - Conversion from specific operation errors
#[derive(Debug, Clone)]
pub struct LapackError {
    /// The specific error code.
    code: ErrorCode,

    /// The operation that failed.
    operation: &'static str,

    /// Additional context message.
    context: Option<String>,
}

impl LapackError {
    /// Creates a new LapackError.
    pub fn new(code: ErrorCode, operation: &'static str) -> Self {
        Self {
            code,
            operation,
            context: None,
        }
    }

    /// Creates an error with additional context.
    pub fn with_context(
        code: ErrorCode,
        operation: &'static str,
        context: impl Into<String>,
    ) -> Self {
        Self {
            code,
            operation,
            context: Some(context.into()),
        }
    }

    /// Creates an error from LAPACK INFO code.
    pub fn from_info(info: InfoCode, operation: &'static str) -> Self {
        Self::new(ErrorCode::from_info(info), operation)
    }

    /// Returns the error code.
    pub fn code(&self) -> &ErrorCode {
        &self.code
    }

    /// Returns the error category.
    pub fn category(&self) -> ErrorCategory {
        self.code.category()
    }

    /// Returns the operation name.
    pub fn operation(&self) -> &'static str {
        self.operation
    }

    /// Returns the LAPACK-style INFO code.
    pub fn to_info(&self) -> InfoCode {
        self.code.to_info()
    }

    /// Returns true if this is a success (should not normally be an error).
    pub fn is_success(&self) -> bool {
        self.code.is_success()
    }

    /// Adds context to this error.
    pub fn add_context(mut self, context: impl Into<String>) -> Self {
        self.context = Some(context.into());
        self
    }

    // Convenience constructors for common errors

    /// Creates a singular matrix error.
    pub fn singular(operation: &'static str, index: usize) -> Self {
        Self::new(ErrorCode::Singular { index }, operation)
    }

    /// Creates a not positive definite error.
    pub fn not_positive_definite(operation: &'static str, index: usize) -> Self {
        Self::new(ErrorCode::NotPositiveDefinite { index }, operation)
    }

    /// Creates an invalid dimension error.
    pub fn invalid_dimension(
        operation: &'static str,
        argument: usize,
        expected: usize,
        actual: usize,
    ) -> Self {
        Self::new(
            ErrorCode::InvalidDimension {
                argument,
                expected,
                actual,
            },
            operation,
        )
    }

    /// Creates a not square error.
    pub fn not_square(operation: &'static str, nrows: usize, ncols: usize) -> Self {
        Self::new(ErrorCode::NotSquare { nrows, ncols }, operation)
    }

    /// Creates a not converged error.
    pub fn not_converged(operation: &'static str, iterations: usize, index: Option<usize>) -> Self {
        Self::new(ErrorCode::NotConverged { iterations, index }, operation)
    }

    /// Creates a workspace too small error.
    pub fn workspace_too_small(operation: &'static str, required: usize, provided: usize) -> Self {
        Self::new(
            ErrorCode::WorkspaceTooSmall { required, provided },
            operation,
        )
    }
}

impl fmt::Display for LapackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.operation, self.code)?;
        if let Some(ref ctx) = self.context {
            write!(f, " ({})", ctx)?;
        }
        Ok(())
    }
}

impl std::error::Error for LapackError {}

// ============================================================================
// Result Type Alias
// ============================================================================

/// Result type for LAPACK operations.
pub type LapackResult<T> = Result<T, LapackError>;

// ============================================================================
// Conversion Traits
// ============================================================================

/// Trait for converting operation-specific errors to LapackError.
pub trait IntoLapackError {
    /// Converts this error to a LapackError.
    fn into_lapack_error(self, operation: &'static str) -> LapackError;
}

/// Trait for types that can report LAPACK-style INFO codes.
pub trait HasInfoCode {
    /// Returns the INFO code for this result/error.
    fn info_code(&self) -> InfoCode;
}

impl HasInfoCode for LapackError {
    fn info_code(&self) -> InfoCode {
        self.to_info()
    }
}

impl<T> HasInfoCode for LapackResult<T> {
    fn info_code(&self) -> InfoCode {
        match self {
            Ok(_) => INFO_SUCCESS,
            Err(e) => e.to_info(),
        }
    }
}

// ============================================================================
// Conversions from Existing Error Types
// ============================================================================

// Import existing error types
use crate::cholesky::CholeskyError;
use crate::evd::SymmetricEvdError;
use crate::lu::LuError;
use crate::qr::QrError;
use crate::svd::SvdError;

impl IntoLapackError for LuError {
    fn into_lapack_error(self, operation: &'static str) -> LapackError {
        let code = match self {
            LuError::Singular { index } => ErrorCode::Singular { index },
            LuError::NotSquare { nrows, ncols } => ErrorCode::NotSquare { nrows, ncols },
            LuError::DimensionMismatch { expected, actual } => ErrorCode::InvalidDimension {
                argument: 2,
                expected,
                actual,
            },
        };
        LapackError::new(code, operation)
    }
}

impl IntoLapackError for CholeskyError {
    fn into_lapack_error(self, operation: &'static str) -> LapackError {
        let code = match self {
            CholeskyError::NotPositiveDefinite { index } => {
                ErrorCode::NotPositiveDefinite { index }
            }
            CholeskyError::NotSquare { nrows, ncols } => ErrorCode::NotSquare { nrows, ncols },
            CholeskyError::DimensionMismatch { expected, actual } => ErrorCode::InvalidDimension {
                argument: 2,
                expected,
                actual,
            },
        };
        LapackError::new(code, operation)
    }
}

impl IntoLapackError for QrError {
    fn into_lapack_error(self, operation: &'static str) -> LapackError {
        let code = match self {
            QrError::EmptyMatrix => ErrorCode::InvalidDimension {
                argument: 1,
                expected: 1,
                actual: 0,
            },
        };
        LapackError::new(code, operation)
    }
}

impl IntoLapackError for SvdError {
    fn into_lapack_error(self, operation: &'static str) -> LapackError {
        let code = match self {
            SvdError::EmptyMatrix => ErrorCode::InvalidDimension {
                argument: 1,
                expected: 1,
                actual: 0,
            },
            SvdError::NotConverged => ErrorCode::NotConverged {
                iterations: 0,
                index: None,
            },
        };
        LapackError::new(code, operation)
    }
}

impl IntoLapackError for SymmetricEvdError {
    fn into_lapack_error(self, operation: &'static str) -> LapackError {
        let code = match self {
            SymmetricEvdError::EmptyMatrix => ErrorCode::InvalidDimension {
                argument: 1,
                expected: 1,
                actual: 0,
            },
            SymmetricEvdError::NotConverged => ErrorCode::NotConverged {
                iterations: 0,
                index: None,
            },
            SymmetricEvdError::NotSquare => ErrorCode::NotSquare { nrows: 0, ncols: 0 },
        };
        LapackError::new(code, operation)
    }
}

// From implementations for convenience
impl From<(LuError, &'static str)> for LapackError {
    fn from((error, operation): (LuError, &'static str)) -> Self {
        error.into_lapack_error(operation)
    }
}

impl From<(CholeskyError, &'static str)> for LapackError {
    fn from((error, operation): (CholeskyError, &'static str)) -> Self {
        error.into_lapack_error(operation)
    }
}

impl From<(QrError, &'static str)> for LapackError {
    fn from((error, operation): (QrError, &'static str)) -> Self {
        error.into_lapack_error(operation)
    }
}

impl From<(SvdError, &'static str)> for LapackError {
    fn from((error, operation): (SvdError, &'static str)) -> Self {
        error.into_lapack_error(operation)
    }
}

impl From<(SymmetricEvdError, &'static str)> for LapackError {
    fn from((error, operation): (SymmetricEvdError, &'static str)) -> Self {
        error.into_lapack_error(operation)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_info_success() {
        assert_eq!(ErrorCode::Success.to_info(), 0);
        assert!(ErrorCode::from_info(0).is_success());
    }

    #[test]
    fn test_info_singular() {
        let code = ErrorCode::Singular { index: 3 };
        assert_eq!(code.to_info(), 4); // 1-based in LAPACK

        let reconstructed = ErrorCode::from_info(4);
        assert!(matches!(reconstructed, ErrorCode::Singular { index: 3 }));
    }

    #[test]
    fn test_info_invalid_argument() {
        let code = ErrorCode::InvalidArgument {
            argument: 5,
            reason: "too small",
        };
        assert_eq!(code.to_info(), -5);
    }

    #[test]
    fn test_error_categories() {
        assert_eq!(ErrorCode::Success.category(), ErrorCategory::Success);
        assert_eq!(
            ErrorCode::Singular { index: 0 }.category(),
            ErrorCategory::Singular
        );
        assert_eq!(
            ErrorCode::NotPositiveDefinite { index: 0 }.category(),
            ErrorCategory::NotPositiveDefinite
        );
        assert_eq!(
            ErrorCode::NotConverged {
                iterations: 100,
                index: None
            }
            .category(),
            ErrorCategory::NotConverged
        );
    }

    #[test]
    fn test_lapack_error_display() {
        let err = LapackError::singular("DGETRF", 5);
        let msg = format!("{}", err);
        assert!(msg.contains("DGETRF"));
        assert!(msg.contains("singular"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_lapack_error_with_context() {
        let err = LapackError::with_context(
            ErrorCode::Singular { index: 0 },
            "DGETRF",
            "diagonal element too small",
        );
        let msg = format!("{}", err);
        assert!(msg.contains("diagonal element"));
    }

    #[test]
    fn test_lapack_error_from_info() {
        let err = LapackError::from_info(-3, "DGESV");
        assert_eq!(err.to_info(), -3);
        assert!(err.code().is_input_error());
    }

    #[test]
    fn test_lu_error_conversion() {
        let lu_err = LuError::Singular { index: 2 };
        let lapack_err = lu_err.into_lapack_error("DGETRF");

        assert_eq!(lapack_err.category(), ErrorCategory::Singular);
        assert_eq!(lapack_err.to_info(), 3); // 1-based
    }

    #[test]
    fn test_cholesky_error_conversion() {
        let chol_err = CholeskyError::NotPositiveDefinite { index: 4 };
        let lapack_err = chol_err.into_lapack_error("DPOTRF");

        assert_eq!(lapack_err.category(), ErrorCategory::NotPositiveDefinite);
    }

    #[test]
    fn test_from_tuple() {
        let lu_err = LuError::NotSquare { nrows: 3, ncols: 4 };
        let lapack_err: LapackError = (lu_err, "DGETRF").into();

        assert!(lapack_err.code().is_input_error());
    }

    #[test]
    fn test_has_info_code_result() {
        let ok_result: LapackResult<i32> = Ok(42);
        assert_eq!(ok_result.info_code(), 0);

        let err_result: LapackResult<i32> = Err(LapackError::singular("TEST", 1));
        assert_eq!(err_result.info_code(), 2); // 1-based
    }

    #[test]
    fn test_workspace_error() {
        let err = LapackError::workspace_too_small("DGESVD", 1000, 100);
        assert_eq!(err.category(), ErrorCategory::InsufficientWorkspace);

        let msg = format!("{}", err);
        assert!(msg.contains("1000"));
        assert!(msg.contains("100"));
    }

    #[test]
    fn test_not_converged() {
        let err = LapackError::not_converged("DGEEV", 30, Some(5));
        assert_eq!(err.category(), ErrorCategory::NotConverged);

        let msg = format!("{}", err);
        assert!(msg.contains("30"));
        assert!(msg.contains("5"));
    }

    #[test]
    fn test_error_code_display() {
        let code = ErrorCode::EigenvalueFailure { num_failed: 3 };
        let msg = format!("{}", code);
        assert!(msg.contains("3"));
        assert!(msg.contains("eigenvalue"));
    }

    #[test]
    fn test_category_display() {
        assert_eq!(format!("{}", ErrorCategory::Singular), "singular matrix");
        assert_eq!(format!("{}", ErrorCategory::Success), "success");
    }

    #[test]
    fn test_add_context() {
        let err = LapackError::singular("DGETRF", 0).add_context("pivot element was exactly zero");

        let msg = format!("{}", err);
        assert!(msg.contains("pivot element"));
    }
}
