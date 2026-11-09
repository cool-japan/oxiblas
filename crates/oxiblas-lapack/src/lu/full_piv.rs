//! LU decomposition with full (complete) pivoting.
//!
//! Full pivoting selects the largest element from the entire remaining
//! submatrix as the pivot, providing maximum numerical stability at the
//! cost of additional computation.
//!
//! PAQ = LU where P is the row permutation and Q is the column permutation.

use super::partial_piv::eliminate_column;
use super::tolerance::lu_pivot_tolerances;
use num_traits::{FromPrimitive, One, Zero};
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

/// Pivot of elimination step `k`: the row, column and magnitude of the entry
/// of largest magnitude in `lu[k.., k..]`.
///
/// Among entries of equal magnitude the first in row-major order wins, as in
/// a search that walks the rows. This search walks the contiguous columns of
/// the column-major storage instead, so on a tie it keeps the entry with the
/// smaller row (an equal entry in the same row has a larger column, and was
/// therefore met later by a row-major walk). Starting from `lu[k, k]` and
/// replacing the candidate only on a strictly larger or tied magnitude, a NaN
/// entry is never chosen unless it is `lu[k, k]` itself, which is then kept.
/// The pivot is therefore exactly that of the row-major search.
fn full_pivot_search<T: Field>(lu: &Mat<T>, k: usize) -> (usize, usize, T::Real) {
    let n = lu.nrows();
    let ld = lu.row_stride();
    let data = lu.raw_data();
    let (mut pivot_row, mut pivot_col) = (k, k);
    let mut pivot_val = Scalar::abs(data[k * ld + k]);
    for j in k..n {
        for (i, &x) in (k..n).zip(&data[j * ld + k..j * ld + n]) {
            let val = Scalar::abs(x);
            if val > pivot_val || (val == pivot_val && i < pivot_row) {
                pivot_val = val;
                pivot_row = i;
                pivot_col = j;
            }
        }
    }
    (pivot_row, pivot_col, pivot_val)
}

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
///
/// By default a pivot is treated as zero, and reported as
/// [`LuFullPivError::Singular`], when its magnitude is at most `eps · n · s_c`,
/// where `s_c = max_i |a_ic|` is taken over the finite entries of the column
/// `c` of the input that the pivot comes from (the column interchanges are
/// tracked). [`Self::rank`] instead counts the diagonal entries of U above
/// the global threshold `n · eps · max |a_ij|`, so a badly scaled matrix such
/// as `diag(1e20, 1)` factors successfully but has numerical rank 1.
#[derive(Clone, Debug)]
pub struct LuFullPiv<T: Scalar> {
    /// Combined L and U factors.
    /// L is stored below the diagonal (with implicit unit diagonal).
    /// U is stored on and above the diagonal.
    lu: Mat<T>,
    /// Row pivot indices: row i was swapped with row `row_pivot[i]`.
    row_pivot: Vec<usize>,
    /// Column pivot indices: column j was swapped with column `col_pivot[j]`.
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
    /// * `tol` - Optional absolute tolerance for detecting singularity: a pivot
    ///           whose magnitude is at most `tol` is treated as zero. If None,
    ///           a pivot taken from column `c` of `a` is compared with
    ///           `eps · n · max_i |a_ic|` (the maximum over the finite entries
    ///           of that column), which does not depend on the scale of `a`
    ///           or of its columns.
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

        // Singularity threshold of every column of A: an explicit tolerance is
        // used as given; the default is `eps · n · max_i |a_ij|` over the
        // finite entries of each column.
        let col_tol = match tol {
            Some(tol) => vec![tol; n],
            None => lu_pivot_tolerances(a),
        };
        // `orig_col[j]` is the column of A currently held at position `j`;
        // the column interchanges below keep it up to date.
        let mut orig_col: Vec<usize> = (0..n).collect();

        // Norm of the original matrix (max absolute entry), captured before `lu`
        // is overwritten by the elimination. Used below to build a *relative*
        // tolerance for rank detection, since an absolute epsilon threshold is
        // meaningless once A is scaled away from O(1) magnitude.
        let mut norm_a = <T::Real as Zero>::zero();
        for j in 0..n {
            for i in 0..n {
                let val = Scalar::abs(a[(i, j)]);
                if val > norm_a {
                    norm_a = val;
                }
            }
        }

        // Doolittle algorithm with full pivoting
        for k in 0..n {
            // Find pivot: largest absolute value in submatrix lu[k..n, k..n]
            let (pivot_row, pivot_col, pivot_val) = full_pivot_search(&lu, k);

            // Check for singularity against the threshold of the column of A
            // the pivot comes from. A NaN pivot fails the comparison.
            if pivot_val <= col_tol[orig_col[pivot_col]] {
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
                orig_col.swap(k, pivot_col);
                num_col_swaps += 1;
            }

            // Compute multipliers (L's subdiagonal entries) and update the
            // remaining submatrix, column by column (same arithmetic as the
            // row-by-row loops, so the factors are bit-identical)
            eliminate_column(&mut lu, k, n);
        }

        // Numerical rank: count diagonal entries of the completed U factor whose
        // magnitude exceeds a tolerance relative to the norm of the original
        // matrix (not a bare absolute epsilon, which would be meaningless for
        // matrices scaled far away from O(1) magnitude). This mirrors the
        // classic LU-based numerical rank estimate `tol = n * eps * ||A||`.
        let rank_tolerance = T::epsilon()
            * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one())
            * norm_a;
        let mut rank = 0usize;
        for i in 0..n {
            if Scalar::abs(lu[(i, i)]) > rank_tolerance {
                rank += 1;
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
    ///
    /// The rank is the number of diagonal entries of the completed U factor
    /// whose magnitude exceeds a tolerance relative to the norm of the
    /// original matrix (`n * eps * ||A||`), not merely the matrix dimension.
    /// For a rank-deficient matrix this is strictly less than [`Self::size`].
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
        let mut det = if total_swaps.is_multiple_of(2) {
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
    fn test_lu_full_piv_rank_deficient_duplicate_rows() {
        // Row 3 is an exact duplicate of row 0, so the true rank is 3, not 4
        // (verified independently: the 3x3 minor formed by rows 0..2 and
        // columns 0..2 has determinant -25, so rows 0..2 are independent).
        //
        // Under exact arithmetic, full pivoting drives the pivot at the
        // dependent step to *exactly* zero (dividing a row by an identical
        // row's pivot gives a multiplier of exactly 1.0, and subtracting two
        // bit-identical rows gives exactly 0.0 in IEEE-754). The default
        // tolerance treats that as a hard singularity (a separate, existing
        // contract this fix does not change — LuFullPivError::Singular must
        // keep firing for `compute()`'s default tolerance, since the FFI
        // layer (oblas_dgetc2/sgetc2) relies on it to report LAPACK-style
        // singular-matrix INFO codes). Here we pass a permissive tolerance
        // purely to obtain the completed factorization so the *rank
        // computation itself* (the thing this test targets) can be checked
        // against the diagonal of U.
        let a: Mat<f64> = Mat::from_rows(&[
            &[4.0, 2.0, 7.0, 1.0],
            &[1.0, 5.0, 2.0, 3.0],
            &[6.0, 1.0, 9.0, 4.0],
            &[4.0, 2.0, 7.0, 1.0],
        ]);

        // Sanity: the default (strict) tolerance still reports singularity,
        // exactly as before this fix.
        let default_result = LuFullPiv::compute(a.as_ref());
        assert!(
            default_result.is_err(),
            "default tolerance must still hard-fail on this singular matrix"
        );

        let lu = LuFullPiv::compute_with_tol(a.as_ref(), Some(-1.0))
            .expect("permissive tolerance should bypass the hard singularity check");

        assert_eq!(lu.size(), 4);
        assert_eq!(
            lu.rank(),
            3,
            "a matrix with a duplicated row has rank 3, not the fabricated n=4"
        );
    }

    #[test]
    fn test_lu_full_piv_rank_deficient_outer_product() {
        // A = u * v^T is a rank-1 matrix by construction. Both u and v are
        // chosen as exact powers of two so every entry, and every ratio
        // formed during elimination, is exactly representable in binary
        // floating point -- guaranteeing the sub-diagonal pivots collapse to
        // *exactly* zero rather than some ambiguous near-zero noise-floor
        // value, making the expected rank deterministic across platforms.
        let u = [1.0f64, 2.0, 4.0, 8.0];
        let v = [1.0f64, 2.0, 4.0, 8.0];
        let rows: Vec<Vec<f64>> = u
            .iter()
            .map(|ui| v.iter().map(|vj| ui * vj).collect())
            .collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        let a: Mat<f64> = Mat::from_rows(&row_refs);

        let lu = LuFullPiv::compute_with_tol(a.as_ref(), Some(-1.0))
            .expect("permissive tolerance should bypass the hard singularity check");

        assert_eq!(lu.size(), 4);
        assert_eq!(lu.rank(), 1, "rank-1 outer product must report rank 1");
    }

    #[test]
    fn test_lu_full_piv_rank_full_rank_unaffected() {
        // Sanity guard: well-conditioned, genuinely full-rank matrices must
        // still report rank() == n after this fix (i.e. the new relative
        // tolerance must not be so aggressive that it flags healthy pivots).
        let a: Mat<f64> = Mat::from_rows(&[&[2.0, 1.0, 1.0], &[4.0, 3.0, 3.0], &[8.0, 7.0, 9.0]]);
        let lu = LuFullPiv::compute(a.as_ref()).expect("Should not be singular");
        assert_eq!(lu.rank(), 3);

        let eye: Mat<f64> = Mat::eye(5);
        let lu_eye = LuFullPiv::compute(eye.as_ref()).expect("Identity should not be singular");
        assert_eq!(lu_eye.rank(), 5);
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

/// Bit-exactness of the column-oriented pivot search and elimination against
/// the previous row-by-row loops.
#[cfg(test)]
mod oracle_tests {
    use super::*;
    use num_complex::Complex64;

    /// The previous `LuFullPiv::compute_with_tol`, with its row-major pivot
    /// search and row-order elimination, kept verbatim as the reference.
    fn oracle<T: Field + bytemuck::Zeroable>(
        a: MatRef<'_, T>,
        tol: Option<T::Real>,
    ) -> Result<LuFullPiv<T>, LuFullPivError> {
        let n = a.nrows();
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
        let col_tol = match tol {
            Some(tol) => vec![tol; n],
            None => lu_pivot_tolerances(a),
        };
        let mut orig_col: Vec<usize> = (0..n).collect();
        let mut norm_a = <T::Real as Zero>::zero();
        for j in 0..n {
            for i in 0..n {
                let val = Scalar::abs(a[(i, j)]);
                if val > norm_a {
                    norm_a = val;
                }
            }
        }

        for k in 0..n {
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
            if pivot_val <= col_tol[orig_col[pivot_col]] {
                return Err(LuFullPivError::Singular { index: k });
            }
            row_pivot[k] = pivot_row;
            col_pivot[k] = pivot_col;
            if pivot_row != k {
                for j in 0..n {
                    let tmp = lu[(k, j)];
                    lu[(k, j)] = lu[(pivot_row, j)];
                    lu[(pivot_row, j)] = tmp;
                }
                num_row_swaps += 1;
            }
            if pivot_col != k {
                for i in 0..n {
                    let tmp = lu[(i, k)];
                    lu[(i, k)] = lu[(i, pivot_col)];
                    lu[(i, pivot_col)] = tmp;
                }
                orig_col.swap(k, pivot_col);
                num_col_swaps += 1;
            }
            let pivot_inv = T::one() / lu[(k, k)];
            for i in (k + 1)..n {
                let mult = lu[(i, k)] * pivot_inv;
                lu[(i, k)] = mult;
                for j in (k + 1)..n {
                    let val = lu[(i, j)] - mult * lu[(k, j)];
                    lu[(i, j)] = val;
                }
            }
        }

        let rank_tolerance = T::epsilon()
            * <T::Real as FromPrimitive>::from_usize(n).unwrap_or(<T::Real as One>::one())
            * norm_a;
        let mut rank = 0usize;
        for i in 0..n {
            if Scalar::abs(lu[(i, i)]) > rank_tolerance {
                rank += 1;
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

    /// Exact bit pattern of a scalar.
    trait Bits: Field + bytemuck::Zeroable {
        fn bits(self) -> (u64, u64);
    }

    impl Bits for f64 {
        fn bits(self) -> (u64, u64) {
            (self.to_bits(), 0)
        }
    }

    impl Bits for f32 {
        fn bits(self) -> (u64, u64) {
            (u64::from(self.to_bits()), 0)
        }
    }

    impl Bits for Complex64 {
        fn bits(self) -> (u64, u64) {
            (self.re.to_bits(), self.im.to_bits())
        }
    }

    fn assert_same<T: Bits>(label: &str, a: &Mat<T>, tol: Option<T::Real>) {
        match (
            LuFullPiv::compute_with_tol(a.as_ref(), tol),
            oracle(a.as_ref(), tol),
        ) {
            (Ok(got), Ok(want)) => {
                assert_eq!(got.row_pivot, want.row_pivot, "{label}: row pivots differ");
                assert_eq!(got.col_pivot, want.col_pivot, "{label}: col pivots differ");
                assert_eq!(got.num_row_swaps, want.num_row_swaps, "{label}: row swaps");
                assert_eq!(got.num_col_swaps, want.num_col_swaps, "{label}: col swaps");
                assert_eq!(got.rank, want.rank, "{label}: ranks differ");
                let n = a.nrows();
                for j in 0..n {
                    for i in 0..n {
                        assert_eq!(
                            got.lu[(i, j)].bits(),
                            want.lu[(i, j)].bits(),
                            "{label}: LU({i}, {j}) differs: {} vs {}",
                            got.lu[(i, j)],
                            want.lu[(i, j)]
                        );
                    }
                }
            }
            (Err(got), Err(want)) => assert_eq!(got, want, "{label}: errors differ"),
            (got, want) => panic!(
                "{label}: outcome differs: got {:?}, oracle {:?}",
                got.map(|_| ()),
                want.map(|_| ())
            ),
        }
    }

    /// Deterministic matrix filled by `f(state)` from a xorshift64 state.
    fn matrix_from<T: Bits>(n: usize, seed: u64, f: impl Fn(u64) -> T) -> Mat<T> {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut a = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                a[(i, j)] = f(x);
            }
        }
        a
    }

    /// Uniform sample in `[-1, 1)` from a xorshift64 state.
    fn unit(x: u64) -> f64 {
        2.0 * ((x >> 11) as f64 / (1u64 << 53) as f64) - 1.0
    }

    #[test]
    fn test_lu_full_piv_bit_identical_random() {
        for &n in &[0usize, 1, 2, 3, 5, 8, 17, 33, 64, 65, 100] {
            let seed = 900 + n as u64;
            assert_same(&format!("f64 n={n}"), &matrix_from(n, seed, unit), None);
            let a32 = matrix_from(n, seed, |x| unit(x) as f32);
            assert_same(&format!("f32 n={n}"), &a32, None);
            let c = matrix_from(n, seed, |x| {
                Complex64::new(unit(x), unit(x.rotate_left(17)))
            });
            assert_same(&format!("c64 n={n}"), &c, None);
        }
    }

    #[test]
    fn test_lu_full_piv_bit_identical_ties_and_singular() {
        // Entries in {-2, ..., 2}: many pivot candidates of equal magnitude,
        // so the tie rule of the column-major search decides; some of these
        // matrices are singular, which must be reported at the same step.
        for &n in &[2usize, 5, 8, 17, 40] {
            for seed in 0..4u64 {
                let a = matrix_from(n, 1000 + 10 * n as u64 + seed, |x| (x % 5) as f64 - 2.0);
                assert_same(&format!("ties n={n} seed={seed}"), &a, None);
            }
        }
        // Rank-deficient with ties, and an explicit tolerance.
        let a: Mat<f64> =
            Mat::from_rows(&[&[1.0, 2.0, -2.0], &[2.0, -2.0, 1.0], &[3.0, 0.0, -1.0]]);
        assert_same("rank-deficient 3x3", &a, None);
        assert_same("explicit tolerance", &a, Some(0.5));
        // A zero row and column.
        let mut z = matrix_from(30, 77, unit);
        for i in 0..30 {
            z[(i, 11)] = 0.0;
            z[(4, i)] = 0.0;
        }
        assert_same("zero row and column", &z, None);
    }

    #[test]
    fn test_lu_full_piv_bit_identical_non_finite() {
        // NaN and infinite entries: the search never picks a NaN unless it is
        // the leading entry, exactly as the row-major search.
        let mut a = matrix_from(12, 5, unit);
        a[(7, 3)] = f64::NAN;
        a[(2, 9)] = f64::INFINITY;
        assert_same("nan and inf", &a, None);
        let mut b = matrix_from(9, 6, unit);
        b[(0, 0)] = f64::NAN;
        assert_same("leading nan", &b, None);
    }
}
