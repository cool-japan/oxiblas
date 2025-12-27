//! Complete orthogonal decomposition (COD).
//!
//! The complete orthogonal decomposition of an m×n matrix A with rank r is:
//! A = Q * [ T  0 ] * Z^T
//!         [ 0  0 ]
//!
//! where:
//! - Q is m×m orthogonal
//! - Z is n×n orthogonal
//! - T is r×r upper triangular and non-singular
//!
//! This is computed via QR factorization with column pivoting followed by
//! an RQ factorization of the R factor.

use crate::error::LapackError;
use crate::qr::{QrPivot, Rq};
use oxiblas_core::scalar::{Field, Real};
use oxiblas_matrix::{Mat, MatRef};

/// Complete orthogonal decomposition result.
#[derive(Debug, Clone)]
pub struct CompleteOrthogonalDecomp<T: Field> {
    /// The Q factor (m×m orthogonal matrix).
    q: Mat<T>,
    /// The triangular factor T (r×r).
    t: Mat<T>,
    /// The Z factor (n×n orthogonal matrix).
    z: Mat<T>,
    /// The numerical rank.
    rank: usize,
    /// Column permutation from pivoting.
    perm: Vec<usize>,
}

impl<T: Field + Real + bytemuck::Zeroable> CompleteOrthogonalDecomp<T> {
    /// Computes the complete orthogonal decomposition.
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix (m×n)
    /// * `tol` - Tolerance for rank determination (singular values below this are treated as zero)
    ///
    /// # Returns
    ///
    /// Complete orthogonal decomposition.
    ///
    /// # Errors
    ///
    /// Returns error if the factorization fails.
    ///
    /// # Example
    ///
    /// ```
    /// use oxiblas_lapack::qr::CompleteOrthogonalDecomp;
    /// use oxiblas_matrix::Mat;
    ///
    /// let a = Mat::from_rows(&[
    ///     &[1.0f64, 2.0, 3.0],
    ///     &[4.0, 5.0, 6.0],
    ///     &[7.0, 8.0, 9.0],
    /// ]);
    ///
    /// let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
    /// let rank = cod.rank();
    /// ```
    pub fn compute(a: MatRef<T>, tol: T) -> Result<Self, LapackError> {
        let m = a.nrows();
        let n = a.ncols();

        // Step 1: QR with column pivoting to reveal rank
        let qr_piv = QrPivot::compute(a).map_err(|_e| {
            LapackError::new(
                crate::error::ErrorCode::Internal {
                    description: "QR pivot failed",
                },
                "complete_orthogonal",
            )
        })?;
        let q = qr_piv.q();
        let r = qr_piv.r();
        let perm = qr_piv.column_permutation().to_vec();

        // Step 2: Determine numerical rank
        let k = m.min(n);
        let mut rank = 0;
        for i in 0..k {
            if oxiblas_core::scalar::Scalar::abs(r[(i, i)]) > tol {
                rank += 1;
            } else {
                break;
            }
        }

        // Step 3: Extract the rank-revealing part of R
        let r_rank = if rank > 0 {
            let mut r_sub = Mat::zeros(rank, n);
            for i in 0..rank {
                for j in 0..n {
                    r_sub[(i, j)] = r[(i, j)];
                }
            }
            r_sub
        } else {
            Mat::zeros(1, n)
        };

        // Step 4: Apply RQ to the rank-revealing part to get T and Z
        let (t, z) = if rank > 0 {
            let rq = Rq::compute(r_rank.as_ref())?;
            let t_mat = rq.r_factor();
            let z_rq = rq.q_factor();

            // Extract just the r×r triangular part from T
            // For RQ of r×n matrix, R has the triangular part in the right columns
            let mut t_small = Mat::zeros(rank, rank);
            for i in 0..rank {
                // The triangular part starts at column (n - rank + i)
                let start_col = n - rank + i;
                for j in start_col..n {
                    let local_j = j - (n - rank);
                    t_small[(i, local_j)] = t_mat[(i, j)];
                }
            }

            // Z is the Q factor from RQ (n×n orthogonal matrix)
            (t_small, z_rq)
        } else {
            // For zero rank, T is empty and Z is identity
            let mut z_id: Mat<T> = Mat::zeros(n, n);
            for i in 0..n {
                z_id[(i, i)] = T::one();
            }
            (Mat::<T>::zeros(0, 0), z_id)
        };

        Ok(Self {
            q,
            t,
            z,
            rank,
            perm,
        })
    }

    /// Returns the numerical rank of the matrix.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Returns the Q factor.
    #[must_use]
    pub fn q(&self) -> &Mat<T> {
        &self.q
    }

    /// Returns the triangular factor T.
    #[must_use]
    pub fn t(&self) -> &Mat<T> {
        &self.t
    }

    /// Returns the Z factor.
    #[must_use]
    pub fn z(&self) -> &Mat<T> {
        &self.z
    }

    /// Returns the column permutation.
    #[must_use]
    pub fn permutation(&self) -> &[usize] {
        &self.perm
    }

    /// Returns the original matrix dimensions.
    pub fn dims(&self) -> (usize, usize) {
        (self.q.nrows(), self.z.nrows())
    }

    /// Reconstructs the original matrix.
    ///
    /// The decomposition is: A * P = Q * R_rq * Z
    /// where R_rq has the form [0 | T] with T being the r×r triangular factor.
    ///
    /// So A = Q * R_rq * Z * P^{-1}
    ///
    /// This is useful for verifying the decomposition.
    pub fn reconstruct(&self) -> Mat<T> {
        let (m, n) = self.dims();
        let r = self.rank;

        // Build the extended R matrix (m × n) with T in the rightmost r columns of first r rows
        // R_ext has the form: [0_{r×(n-r)} | T]
        //                     [0_{(m-r)×n}    ]
        let mut r_ext = Mat::zeros(m, n);
        for i in 0..r {
            for j in 0..r {
                r_ext[(i, n - r + j)] = self.t[(i, j)];
            }
        }

        // Compute Q * R_ext
        let mut qr = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..m {
                    sum = sum + self.q[(i, k)] * r_ext[(k, j)];
                }
                qr[(i, j)] = sum;
            }
        }

        // Compute (Q * R_ext) * Z
        let mut qrz = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for k in 0..n {
                    sum = sum + qr[(i, k)] * self.z[(k, j)];
                }
                qrz[(i, j)] = sum;
            }
        }

        // Apply inverse permutation to columns (P^{-1})
        // perm[j] = original column that moved to position j
        // So result[:, perm[j]] = qrz[:, j]
        let mut result = Mat::zeros(m, n);
        for j in 0..n {
            let orig_col = self.perm[j];
            for i in 0..m {
                result[(i, orig_col)] = qrz[(i, j)];
            }
        }

        result
    }

    /// Solves the minimum-norm least squares problem.
    ///
    /// For the system A*x = b, computes the solution x that:
    /// - Minimizes ||A*x - b||_2 if the system is overdetermined
    /// - Minimizes ||x||_2 among all solutions if the system is underdetermined
    ///
    /// This uses the COD: A = Q * [T 0; 0 0] * Z^T * P^T
    pub fn solve(&self, b: MatRef<T>) -> Result<Mat<T>, LapackError> {
        let (m, n) = self.dims();
        let r = self.rank;
        let nrhs = b.ncols();

        if b.nrows() != m {
            return Err(LapackError::new(
                crate::error::ErrorCode::InvalidDimension {
                    argument: 1,
                    expected: m,
                    actual: b.nrows(),
                },
                "complete_orthogonal_solve",
            ));
        }

        if r == 0 {
            // Zero matrix: minimum norm solution is zero
            return Ok(Mat::zeros(n, nrhs));
        }

        // Step 1: Compute c = Q^T * b (m × nrhs)
        let mut c = Mat::zeros(m, nrhs);
        for j in 0..nrhs {
            for i in 0..m {
                let mut sum = T::zero();
                for k in 0..m {
                    sum = sum + self.q[(k, i)] * b[(k, j)]; // Q^T: take row i of Q^T = column i of Q
                }
                c[(i, j)] = sum;
            }
        }

        // Step 2: Extract c1 = c[0:r, :] and solve T * y = c1
        // Use back substitution since T is upper triangular
        let mut y = Mat::zeros(r, nrhs);
        for col in 0..nrhs {
            for i in (0..r).rev() {
                let mut sum = c[(i, col)];
                for j in (i + 1)..r {
                    sum = sum - self.t[(i, j)] * y[(j, col)];
                }
                y[(i, col)] = sum / self.t[(i, i)];
            }
        }

        // Step 3: Compute z = Z^T * [0; y] where we put y in the last r positions
        // Since T is in columns (n-r) to (n-1), we use Z[:, n-r:n]
        // z = Z[:, n-r:n] * y = sum_{k=0}^{r-1} Z[:, n-r+k] * y[k, :]
        let mut z_result = Mat::zeros(n, nrhs);
        for j in 0..nrhs {
            for i in 0..n {
                let mut sum = T::zero();
                for k in 0..r {
                    sum = sum + self.z[(i, n - r + k)] * y[(k, j)];
                }
                z_result[(i, j)] = sum;
            }
        }

        // Step 4: Apply permutation P: x = P * z
        // P is the column permutation, so x[perm[i]] = z[i]
        let mut x = Mat::zeros(n, nrhs);
        for i in 0..n {
            let orig_idx = self.perm[i];
            for j in 0..nrhs {
                x[(orig_idx, j)] = z_result[(i, j)];
            }
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_cod_full_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 0.0], &[0.0, 1.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();

        assert_eq!(cod.rank(), 2);
    }

    #[test]
    fn test_cod_rank_deficient() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[2.0, 4.0, 6.0], &[3.0, 6.0, 9.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-8).unwrap();

        // All rows are proportional, rank = 1
        assert_eq!(cod.rank(), 1);
    }

    #[test]
    fn test_cod_rectangular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();

        assert_eq!(cod.rank(), 2);
    }

    #[test]
    fn test_cod_zero_matrix() {
        let a = Mat::zeros(3, 3);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();

        assert_eq!(cod.rank(), 0);
    }

    #[test]
    fn test_cod_q_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        let q = cod.q();

        // Q^T * Q should be identity
        let m = q.nrows();
        for i in 0..m {
            for j in 0..m {
                let mut dot = 0.0;
                for k in 0..m {
                    dot += q[(k, i)] * q[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Q^T*Q[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_cod_z_orthogonal() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        let z = cod.z();

        // Z^T * Z should be identity
        let n = z.nrows();
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += z[(k, i)] * z[(k, j)];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    approx_eq(dot, expected, 1e-10),
                    "Z^T*Z[{},{}] = {}, expected {}",
                    i,
                    j,
                    dot,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_cod_t_upper_triangular() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 10.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        let t = cod.t();

        // T should be upper triangular
        let r = cod.rank();
        for i in 0..r {
            for j in 0..i {
                assert!(
                    approx_eq(t[(i, j)], 0.0, 1e-10),
                    "T[{},{}] = {}, expected 0",
                    i,
                    j,
                    t[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cod_reconstruction_full_rank() {
        let a = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        let reconstructed = cod.reconstruct();

        // Verify reconstruction matches original
        let m = a.nrows();
        let n = a.ncols();
        for i in 0..m {
            for j in 0..n {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-10),
                    "Reconstruction[{},{}] = {}, expected {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cod_reconstruction_rank_deficient() {
        // Rank-2 matrix in 3x3
        let a = Mat::from_rows(&[
            &[1.0f64, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[5.0, 7.0, 9.0], // Row 3 = Row 1 + Row 2
        ]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        assert_eq!(cod.rank(), 2);

        let reconstructed = cod.reconstruct();

        // Verify reconstruction matches original
        let m = a.nrows();
        let n = a.ncols();
        for i in 0..m {
            for j in 0..n {
                assert!(
                    approx_eq(reconstructed[(i, j)], a[(i, j)], 1e-9),
                    "Reconstruction[{},{}] = {}, expected {}",
                    i,
                    j,
                    reconstructed[(i, j)],
                    a[(i, j)]
                );
            }
        }
    }

    #[test]
    fn test_cod_solve_full_rank() {
        // Full rank 2x2 system
        let a = Mat::from_rows(&[&[2.0f64, 1.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[4.0f64], &[5.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        let x = cod.solve(b.as_ref()).unwrap();

        // Verify A*x = b
        for i in 0..2 {
            let mut ax = 0.0;
            for j in 0..2 {
                ax += a[(i, j)] * x[(j, 0)];
            }
            assert!(
                approx_eq(ax, b[(i, 0)], 1e-10),
                "Ax[{}] = {}, expected {}",
                i,
                ax,
                b[(i, 0)]
            );
        }
    }

    #[test]
    fn test_cod_solve_overdetermined() {
        // Overdetermined system (least squares)
        let a = Mat::from_rows(&[&[1.0f64, 1.0], &[1.0, 2.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[1.0f64], &[2.0], &[3.0]]);

        let cod = CompleteOrthogonalDecomp::compute(a.as_ref(), 1e-10).unwrap();
        let x = cod.solve(b.as_ref()).unwrap();

        // For least squares, A^T * (A*x - b) should be approximately zero
        let mut residual = [0.0; 3];
        for i in 0..3 {
            let mut ax = 0.0;
            for j in 0..2 {
                ax += a[(i, j)] * x[(j, 0)];
            }
            residual[i] = ax - b[(i, 0)];
        }

        // A^T * residual should be close to zero
        for j in 0..2 {
            let mut atr = 0.0;
            for i in 0..3 {
                atr += a[(i, j)] * residual[i];
            }
            assert!(
                approx_eq(atr, 0.0, 1e-9),
                "A^T*r[{}] = {}, expected 0",
                j,
                atr
            );
        }
    }
}
