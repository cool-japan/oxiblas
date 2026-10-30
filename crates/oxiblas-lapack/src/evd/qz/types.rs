//! Public value types of the QZ decomposition: [`QzError`] and
//! [`GeneralizedEigenvalue`].

use oxiblas_core::scalar::{Field, Real, Scalar};

/// Error type for QZ decomposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QzError {
    /// Matrices are empty.
    EmptyMatrix,
    /// Matrices have incompatible dimensions.
    DimensionMismatch {
        /// Rows of A.
        nrows_a: usize,
        /// Columns of A.
        ncols_a: usize,
        /// Rows of B.
        nrows_b: usize,
        /// Columns of B.
        ncols_b: usize,
    },
    /// Matrix A is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Algorithm did not converge.
    ///
    /// Returned when the QZ iteration exhausts its budget of
    /// `30 · (ihi − ilo + 1)` passes (LAPACK `DGGES` `INFO = 1..N` or
    /// `INFO = N+1`; `iterations` is that budget), and with `iterations: 0`
    /// when an input entry is not finite (NaN or ±∞).
    NotConverged {
        /// Iteration count when failure occurred.
        iterations: usize,
    },
}

impl core::fmt::Display for QzError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyMatrix => write!(f, "Empty matrix"),
            Self::DimensionMismatch {
                nrows_a,
                ncols_a,
                nrows_b,
                ncols_b,
            } => {
                write!(
                    f,
                    "Dimension mismatch: A is {nrows_a}x{ncols_a}, B is {nrows_b}x{ncols_b}"
                )
            }
            Self::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows} x {ncols}")
            }
            Self::NotConverged { iterations } => {
                write!(f, "Did not converge after {iterations} iterations")
            }
        }
    }
}

impl std::error::Error for QzError {}

/// Generalized eigenvalue `λ = (alpha_real + i·alpha_imag) / beta` from the
/// QZ decomposition, in LAPACK `DGGES` (`ALPHAR`, `ALPHAI`, `BETA`) form.
///
/// * `beta` is always `≥ 0`; `beta == 0` is an infinite eigenvalue (and
///   `alpha == beta == 0` signals a singular pencil, also reported as not
///   finite).
/// * For a 1×1 diagonal block of the Schur form, `alpha_real = S[j,j]`,
///   `alpha_imag = 0` and `beta = T[j,j]` exactly.
/// * A complex conjugate pair comes from a 2×2 block and occupies two
///   consecutive entries: the first has `alpha_imag > 0`, the second
///   `alpha_imag < 0`. The two `beta`s generally differ (they are not the
///   diagonal of `T`), so `alpha_imag` of the second member is not the
///   exact negation of the first; the *ratios* `alpha/beta` are conjugate.
///   (After a reordering the triples of a pair come from `DLAG2` and share
///   `beta`, with exactly negated `alpha_imag`, unless an unscale guard of
///   `DGGES` rescaled one member.)
#[derive(Debug, Clone, Copy)]
pub struct GeneralizedEigenvalue<T: Scalar> {
    /// Real part of the numerator `α` (`ALPHAR`).
    pub alpha_real: T,
    /// Imaginary part of the numerator `α` (`ALPHAI`; zero for real eigenvalues).
    pub alpha_imag: T,
    /// Denominator `β ≥ 0` (`BETA`; zero for infinite eigenvalues).
    pub beta: T,
}

impl<T: Field + Real> GeneralizedEigenvalue<T> {
    /// Creates a real generalized eigenvalue.
    pub fn real(alpha: T, beta: T) -> Self {
        Self {
            alpha_real: alpha,
            alpha_imag: T::zero(),
            beta,
        }
    }

    /// Creates a complex generalized eigenvalue.
    pub fn complex(alpha_real: T, alpha_imag: T, beta: T) -> Self {
        Self {
            alpha_real,
            alpha_imag,
            beta,
        }
    }

    /// Returns true if the eigenvalue is real (`alpha_imag == 0` exactly, as
    /// written by the QZ iteration for 1×1 blocks).
    pub fn is_real(&self) -> bool {
        self.alpha_imag == T::zero()
    }

    /// Returns true if the eigenvalue is finite (`beta != 0`).
    ///
    /// Deflation of an infinite eigenvalue in [`Qz::compute`] writes an
    /// exact zero into `beta`, so no tolerance is involved there. A singular
    /// pencil (`alpha == beta == 0`) is reported as not finite.
    ///
    /// After a reordering ([`Qz::compute_sorted`], [`Qz::reorder`]) an
    /// infinite eigenvalue that took part in a swap of diagonal blocks is in
    /// general represented by a rounding-level `beta` instead of an exact
    /// zero; test finiteness relative to `|alpha|` for such factorizations.
    ///
    /// [`Qz::compute`]: super::Qz::compute
    /// [`Qz::compute_sorted`]: super::Qz::compute_sorted
    /// [`Qz::reorder`]: super::Qz::reorder
    pub fn is_finite(&self) -> bool {
        self.beta != T::zero()
    }

    /// Computes the eigenvalue as α/β.
    /// Returns None if β = 0 (infinite eigenvalue) or the eigenvalue is complex.
    pub fn value(&self) -> Option<T> {
        if self.is_finite() && self.is_real() {
            Some(self.alpha_real / self.beta)
        } else {
            None
        }
    }

    /// Computes the real and imaginary parts of α/β.
    /// Returns None if β = 0 (infinite eigenvalue).
    pub fn value_complex(&self) -> Option<(T, T)> {
        if !self.is_finite() {
            return None;
        }
        Some((self.alpha_real / self.beta, self.alpha_imag / self.beta))
    }
}
