//! LU decomposition with full (complete) pivoting.
//!
//! Full pivoting selects the largest element from the entire remaining
//! submatrix as the pivot, providing maximum numerical stability at the
//! cost of additional computation.
//!
//! PAQ = LU where P is the row permutation and Q is the column permutation.

use num_traits::{FromPrimitive, One};
use oxiblas_core::scalar::{Field, Scalar};
use oxiblas_matrix::{Mat, MatRef};

/// Error returned when LU full pivoting decomposition fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LuFullPivError {
    /// The matrix is singular (has a zero or near-zero pivot).
    Singular {
        /// The index where the singularity was detected.
        index: usize,
    },
    /// The matrix is not square.
    NotSquare {
        /// Number of rows.
        nrows: usize,
        /// Number of columns.
        ncols: usize,
    },
    /// Dimension mismatch in solve operation.
    DimensionMismatch {
        /// Expected dimension.
        expected: usize,
        /// Actual dimension.
        actual: usize,
    },
}

impl core::fmt::Display for LuFullPivError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            LuFullPivError::Singular { index } => {
                write!(f, "Matrix is singular at index {index}")
            }
            LuFullPivError::NotSquare { nrows, ncols } => {
                write!(f, "Matrix is not square: {nrows}×{ncols}")
            }
            LuFullPivError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for LuFullPivError {}

/// LU decomposition with full (row and column) pivoting.
///
/// Stores the factorization PAQ = LU where:
/// - P is a row permutation matrix (stored as pivot indices)
/// - Q is a column permutation matrix (stored as pivot indices)
/// - L is lower triangular with unit diagonal
/// - U is upper triangular
///
/// Full pivoting provides maximum numerical stability by selecting the
/// largest element in the remaining submatrix as the pivot.
#[derive(Clone, Debug)]
pub struct LuFullPiv<T: Scalar> {
    /// Combined L and U factors.
    /// L is stored below the diagonal (with implicit unit diagonal).
    /// U is stored on and above the diagonal.
    lu: Mat<T>,
    /// Row pivot indices: row i was swapped with row row_pivot[i].
    row_pivot: Vec<usize>,
    /// Column pivot indices: column j was swapped with column col_pivot[j].
    col_pivot: Vec<usize>,
    /// Number of row swaps (for determinant sign).
    num_row_swaps: usize,
    /// Number of column swaps (for determinant sign).
    num_col_swaps: usize,
    /// The numerical rank detected during factorization.
    rank: usize,
}

impl<T: Field + bytemuck::Zeroable> LuFullPiv<T> {
    /// Computes the LU decomposition with full pivoting.
    ///
    /// Uses both row and column permutations for maximum numerical stability.
    ///
    /// # Errors
    ///
    /// Returns `LuFullPivError::NotSquare` if the matrix is not square.
    /// Returns `LuFullPivError::Singular` if the matrix is singular.
    pub fn compute(a: MatRef<'_, T>) -> Result<Self, LuFullPivError> {
        Self::compute_with_tol(a, None)
    }

    /// Computes the LU decomposition with full pivoting and custom tolerance.
    ///
    /// # Arguments
    ///
    /// * `a` - The matrix to decompose
    /// * `tol` - Optional tolerance for detecting singularity. If None, uses
    ///           machine epsilon scaled by matrix size.
    ///
    /// # Errors
    ///
    /// Returns `LuFullPivError::NotSquare` if the matrix is not square.
    /// Returns `LuFullPivError::Singular` if the matrix is singular.
    pub fn compute_with_tol(
        a: MatRef<'_, T>,
        tol: Option<T::Real>,
    ) -> Result<Self, LuFullPivError> {
        let n = a.nrows();

        if n != a.ncols() {
            return Err(LuFullPivError::NotSquare {
                nrows: n,
                ncols: a.ncols(),
            });
        }

        if n == 0 {
            return Ok(LuFullPiv {
                lu: Mat::zeros(0, 0),
                row_pivot: Vec::new(),
                col_pivot: Vec::new(),
                num_row_swaps: 0,
                num_col_swaps: 0,
                rank: 0,
            });
        }

        // Copy A into LU matrix
        let mut lu = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                lu[(i, j)] = a[(i, j)];
            }
        }

        let mut row_pivot = vec![0usize; n];
        let mut col_pivot = vec![0usize; n];
        let mut num_row_swaps = 0;
        let mut num_col_swaps = 0;

        // Default tolerance
        let default_tol = T::epsilon()
            * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one());
        let tolerance = tol.unwrap_or(default_tol);

        let rank = n;

        // Doolittle algorithm with full pivoting
        for k in 0..n {
            // Find pivot: largest absolute value in submatrix lu[k..n, k..n]
            let mut pivot_row = k;
            let mut pivot_col = k;
            let mut pivot_val = Scalar::abs(lu[(k, k)]);

            for i in k..n {
                for j in k..n {
                    let val = Scalar::abs(lu[(i, j)]);
                    if val > pivot_val {
                        pivot_val = val;
                        pivot_row = i;
                        pivot_col = j;
                    }
                }
            }

            // Check for singularity
            if pivot_val <= tolerance {
                // Matrix is numerically singular
                return Err(LuFullPivError::Singular { index: k });
            }

            // Store pivots
            row_pivot[k] = pivot_row;
            col_pivot[k] = pivot_col;

            // Swap rows if needed
            if pivot_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                num_row_swaps += 1;
            }

            // Swap columns if needed
            if pivot_col != k {
                for i in 0..n {
                    let tmp = lu[(i, k)];
                    lu[(i, k)] = lu[(i, pivot_col)];
                    lu[(i, pivot_col)] = tmp;
                }
                num_col_swaps += 1;
            }

            // Compute multipliers (L's subdiagonal entries) and update
            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                // Multiplier (stored in L)
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;

                // Update remaining submatrix
                for j in (k + 1)..n {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }

        Ok(LuFullPiv {
            lu,
            row_pivot,
            col_pivot,
            num_row_swaps,
            num_col_swaps,
            rank,
        })
    }

    /// Returns the size of the matrix (n for an n×n matrix).
    #[inline]
    pub fn size(&self) -> usize {
        self.lu.nrows()
    }

    /// Returns the numerical rank detected during factorization.
    #[inline]
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Returns a reference to the combined LU matrix.
    ///
    /// L is stored below the diagonal, U is on and above the diagonal.
    pub fn lu_matrix(&self) -> MatRef<'_, T> {
        self.lu.as_ref()
    }

    /// Returns the row pivot indices.
    pub fn row_pivot(&self) -> &[usize] {
        &self.row_pivot
    }

    /// Returns the column pivot indices.
    pub fn col_pivot(&self) -> &[usize] {
        &self.col_pivot
    }

    /// Computes the determinant of the original matrix.
    ///
    /// The determinant is the product of U's diagonal elements,
    /// with sign determined by the total number of permutations.
    pub fn determinant(&self) -> T {
        let n = self.size();
        if n == 0 {
            return T::one();
        }

        let total_swaps = self.num_row_swaps + self.num_col_swaps;
        let mut det = if total_swaps % 2 == 0 {
            T::one()
        } else {
            -T::one()
        };

        // Product of U's diagonal
        for i in 0..n {
            det = det * self.lu[(i, i)];
        }

        det
    }

    /// Solves the system Ax = b.
    ///
    /// Given the LU factorization PAQ = LU, solves Ax = b:
    /// 1. PAQ = LU, so A = P^(-1) L U Q^(-1)
    /// 2. Ax = b means P^(-1) L U Q^(-1) x = b
    /// 3. Let y = Q^(-1) x, then P^(-1) L U y = b
    /// 4. So LU y = Pb (apply row permutation)
    /// 5. Forward solve: Lz = Pb
    /// 6. Backward solve: Uy = z
    /// 7. Apply inverse column permutation: x = Qy
    ///
    /// # Arguments
    ///
    /// * `b` - The right-hand side matrix (n × m for multiple RHS)
    ///
    /// # Errors
    ///
    /// Returns `LuFullPivError::DimensionMismatch` if b has wrong number of rows.
    pub fn solve(&self, b: MatRef<'_, T>) -> Result<Mat<T>, LuFullPivError> {
        let n = self.size();

        if b.nrows() != n {
            return Err(LuFullPivError::DimensionMismatch {
                expected: n,
                actual: b.nrows(),
            });
        }

        let m = b.ncols();
        let mut x = Mat::zeros(n, m);
        let mut work = Mat::zeros(n, m);

        // Copy b to work, applying row permutation
        for j in 0..m {
            for i in 0..n {
                work[(i, j)] = b[(i, j)];
            }
        }

        // Apply row permutations in order
        for k in 0..n {
            let pk = self.row_pivot[k];
            if k != pk {
                for j in 0..m {
                    let tmp = work[(k, j)];
                    work[(k, j)] = work[(pk, j)];
                    work[(pk, j)] = tmp;
                }
            }
        }

        // Forward substitution: Lz = Pb (L has unit diagonal)
        for k in 0..n {
            for i in (k + 1)..n {
                let mult = self.lu[(i, k)];
                for j in 0..m {
                    let val = work[(i, j)] - mult * work[(k, j)];
                    work[(i, j)] = val;
                }
            }
        }

        // Back substitution: Uy = z
        for k in (0..n).rev() {
            let diag = self.lu[(k, k)];
            for j in 0..m {
                work[(k, j)] = work[(k, j)] / diag;
            }

            for i in 0..k {
                let mult = self.lu[(i, k)];
                for j in 0..m {
                    let val = work[(i, j)] - mult * work[(k, j)];
                    work[(i, j)] = val;
                }
            }
        }

        // Apply inverse column permutation: x = Qy
        // We need to apply the permutations in reverse order
        // First copy work to x
        for j in 0..m {
            for i in 0..n {
                x[(i, j)] = work[(i, j)];
            }
        }

        // Apply column permutations in reverse
        for k in (0..n).rev() {
            let pk = self.col_pivot[k];
            if k != pk {
                for j in 0..m {
                    let tmp = x[(k, j)];
                    x[(k, j)] = x[(pk, j)];
                    x[(pk, j)] = tmp;
                }
            }
        }

        Ok(x)
    }

    /// Computes the inverse of the original matrix.
    ///
    /// Solves AX = I to find A^(-1).
    pub fn inverse(&self) -> Result<Mat<T>, LuFullPivError> {
        let n = self.size();
        let identity = Mat::<T>::eye(n);
        self.solve(identity.as_ref())
    }

    /// Extracts the L factor (lower triangular with unit diagonal).
    pub fn l_factor(&self) -> Mat<T> {
        let n = self.size();
        let mut l = Mat::zeros(n, n);

        for i in 0..n {
            // Unit diagonal
            l[(i, i)] = T::one();
            // Below diagonal
            for j in 0..i {
                l[(i, j)] = self.lu[(i, j)];
            }
        }

        l
    }

    /// Extracts the U factor (upper triangular).
    pub fn u_factor(&self) -> Mat<T> {
        let n = self.size();
        let mut u = Mat::zeros(n, n);

        for i in 0..n {
            // On and above diagonal
            for j in i..n {
                u[(i, j)] = self.lu[(i, j)];
            }
        }

        u
    }

    /// Constructs the row permutation matrix P.
    ///
    /// P is such that PAQ = LU (P acts on rows).
    pub fn row_permutation_matrix(&self) -> Mat<T> {
        let n = self.size();
        let mut p = Mat::eye(n);

        for k in 0..n {
            let pk = self.row_pivot[k];
            if k != pk {
                // Swap rows k and pk
                for j in 0..n {
                    let tmp = p[(k, j)];
                    p[(k, j)] = p[(pk, j)];
                    p[(pk, j)] = tmp;
                }
            }
        }

        p
    }

    /// Constructs the column permutation matrix Q.
    ///
    /// Q is such that PAQ = LU (Q acts on columns).
    pub fn col_permutation_matrix(&self) -> Mat<T> {
        let n = self.size();
        let mut q = Mat::eye(n);

        for k in 0..n {
            let pk = self.col_pivot[k];
            if k != pk {
                // Swap columns k and pk
                for i in 0..n {
                    let tmp = q[(i, k)];
                    q[(i, k)] = q[(i, pk)];
                    q[(i, pk)] = tmp;
                }
            }
        }

        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lu_full_piv_simple() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 3.0], &[6.0, 3.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");

        // det(A) = 4*3 - 3*6 = 12 - 18 = -6
        let det = lu.determinant();
        assert!((det.abs() - 6.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_lu_full_piv_solve() {
        // A = [2 1; 4 3]
        // b = [3; 7]
        // x = [1; 1] (since 2*1 + 1*1 = 3, 4*1 + 3*1 = 7)
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0], &[4.0, 3.0]]);
        let b: Mat<f64> = Mat::from_rows(&[&[3.0], &[7.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-10, "x[1] = {}", x[(1, 0)]);
    }

    #[test]
    fn test_lu_full_piv_singular() {
        // Singular matrix (second row is 2x first row)
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);

        let result = LuFullPiv::compute(a.as_ref());
        assert!(result.is_err());
    }

    #[test]
    fn test_lu_full_piv_3x3() {
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0, 1.0], &[4.0, 3.0, 3.0], &[8.0, 7.0, 9.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");

        // Test solve: Ax = b where b = [4, 10, 24]
        // Solution should be x = [1, 1, 1]
        let b: Mat<f64> = Mat::from_rows(&[&[4.0], &[10.0], &[24.0]]);
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-10, "x[1] = {}", x[(1, 0)]);
        assert!((x[(2, 0)] - 1.0).abs() < 1e-10, "x[2] = {}", x[(2, 0)]);
    }

    #[test]
    fn test_lu_full_piv_determinant() {
        // A = [1 2 3; 4 5 6; 7 8 10]
        // det = 1*(5*10 - 6*8) - 2*(4*10 - 6*7) + 3*(4*8 - 5*7)
        //     = 1*(50 - 48) - 2*(40 - 42) + 3*(32 - 35)
        //     = 1*2 - 2*(-2) + 3*(-3)
        //     = 2 + 4 - 9 = -3
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");
        let det = lu.determinant();

        assert!((det + 3.0).abs() < 1e-10, "det = {}", det);
    }

    #[test]
    fn test_lu_full_piv_inverse() {
        let a: Mat<f64> = Mat::from_rows(&[&[4.0, 7.0], &[2.0, 6.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");
        let a_inv = lu.inverse().expect("Should invert");

        // A * A^-1 should be identity
        // det(A) = 24 - 14 = 10
        // A^-1 = [6/10 -7/10; -2/10 4/10] = [0.6 -0.7; -0.2 0.4]
        assert!((a_inv[(0, 0)] - 0.6).abs() < 1e-10);
        assert!((a_inv[(0, 1)] + 0.7).abs() < 1e-10);
        assert!((a_inv[(1, 0)] + 0.2).abs() < 1e-10);
        assert!((a_inv[(1, 1)] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_lu_full_piv_paq_lu() {
        // Test that P * A * Q = L * U
        let a: Mat<f64> = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");

        let l = lu.l_factor();
        let u = lu.u_factor();
        let p = lu.row_permutation_matrix();
        let q = lu.col_permutation_matrix();

        // Compute L * U
        let n = a.nrows();
        let mut lu_prod = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += l[(i, k)] * u[(k, j)];
                }
                lu_prod[(i, j)] = sum;
            }
        }

        // Compute P * A * Q
        let mut pa = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += p[(i, k)] * a[(k, j)];
                }
                pa[(i, j)] = sum;
            }
        }

        let mut paq = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += pa[(i, k)] * q[(k, j)];
                }
                paq[(i, j)] = sum;
            }
        }

        // Check P * A * Q ≈ L * U
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (paq[(i, j)] - lu_prod[(i, j)]).abs() < 1e-10,
                    "PAQ[{},{}] = {}, LU[{},{}] = {}",
                    i,
                    j,
                    paq[(i, j)],
                    i,
                    j,
                    lu_prod[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_lu_full_piv_f32() {
        let a: Mat<f32> = Mat::from_rows(&[&[2.0f32, 1.0], &[4.0, 3.0]]);
        let b: Mat<f32> = Mat::from_rows(&[&[3.0f32], &[7.0]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");
        let x = lu.solve(b.as_ref()).expect("Should solve");

        assert!((x[(0, 0)] - 1.0).abs() < 1e-5, "x[0] = {}", x[(0, 0)]);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-5, "x[1] = {}", x[(1, 0)]);
    }

    #[test]
    fn test_lu_full_piv_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Empty should succeed");
        assert_eq!(lu.size(), 0);
        assert_eq!(lu.rank(), 0);
    }

    #[test]
    fn test_lu_full_piv_not_square() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let result = LuFullPiv::compute(a.as_ref());
        assert!(matches!(result, Err(LuFullPivError::NotSquare { .. })));
    }

    #[test]
    fn test_lu_full_piv_identity() {
        let eye: Mat<f64> = Mat::eye(3);

        let lu = LuFullPiv::compute(eye.as_ref()).expect("Identity should not be singular");

        // Determinant should be 1
        let det = lu.determinant();
        assert!((det - 1.0).abs() < 1e-10);

        // Inverse should be identity
        let inv = lu.inverse().expect("Should invert");
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[(i, j)] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_lu_full_piv_better_stability() {
        // This matrix has small diagonal elements that would cause issues
        // with partial pivoting, but full pivoting handles it well
        let a: Mat<f64> =
            Mat::from_rows(&[&[1e-10, 1.0, 2.0], &[1.0, 1e-10, 3.0], &[2.0, 3.0, 1e-10]]);

        let lu = LuFullPiv::compute(a.as_ref()).expect("Should handle ill-conditioned");

        // Should be able to solve
        let b: Mat<f64> = Mat::from_rows(&[&[1.0], &[1.0], &[1.0]]);
        let x = lu.solve(b.as_ref()).expect("Should solve");

        // Verify Ax ≈ b
        for i in 0..3 {
            let mut sum = 0.0;
            for j in 0..3 {
                sum += a[(i, j)] * x[(j, 0)];
            }
            assert!((sum - b[(i, 0)]).abs() < 1e-5, "Ax[{}] = {}", i, sum);
        }
    }
}
