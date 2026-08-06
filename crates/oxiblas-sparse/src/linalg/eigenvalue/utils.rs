//! Utility functions for eigenvalue computations.
//!
//! This module provides helper functions for vector and matrix operations
//! used by the eigenvalue solvers.

use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use num_traits::FromPrimitive;
use oxiblas_core::scalar::{Field, Real, Scalar};

use super::error::EigenvalueError;

// =============================================================================
// Vector Operations
// =============================================================================

/// Compute Givens rotation parameters.
///
/// Returns (c, s, r) such that:
/// [c  s][a] = [r]
/// [-s c][b]   [0]
pub fn givens_rotation<T: Scalar<Real = T> + Clone + Field + Real>(a: T, b: T) -> (T, T, T) {
    if Scalar::abs(b.clone()) <= <T as Scalar>::epsilon() {
        return (T::one(), T::zero(), a);
    }

    if Scalar::abs(a.clone()) <= <T as Scalar>::epsilon() {
        return (
            T::zero(),
            if b >= T::zero() {
                T::one()
            } else {
                T::zero() - T::one()
            },
            Scalar::abs(b),
        );
    }

    let r = Real::sqrt(a.clone() * a.clone() + b.clone() * b.clone());
    let c = a / r.clone();
    let s = b / r.clone();

    (c, s, r)
}

/// Computes the dot product of two vectors.
pub fn dot<T: Scalar + Clone + Field>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i].clone() * b[i].clone();
    }
    sum
}

/// Computes the 2-norm of a vector.
pub fn norm<T: Scalar<Real = T> + Clone + Field + Real>(v: &[T]) -> T {
    Real::sqrt(dot(v, v))
}

// =============================================================================
// Matrix Operations
// =============================================================================

/// Compute C = A - sigma * B (sparse matrix subtraction).
///
/// # Errors
///
/// Returns [`EigenvalueError::NotSquare`] if `a` is not square, or
/// [`EigenvalueError::DimensionMismatch`] if `b`'s dimensions do not match
/// `a`'s. Returns [`EigenvalueError::ComputationError`] if the resulting
/// sparsity pattern fails to form a valid CSR matrix (should not occur for
/// well-formed inputs, but is surfaced rather than panicking).
pub fn subtract_scaled_matrices<T: Scalar + Clone>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    sigma: T,
) -> Result<CsrMatrix<T>, EigenvalueError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(EigenvalueError::NotSquare {
            nrows: n,
            ncols: a.ncols(),
        });
    }
    if b.nrows() != n {
        return Err(EigenvalueError::DimensionMismatch {
            expected: n,
            actual: b.nrows(),
        });
    }
    if b.ncols() != n {
        return Err(EigenvalueError::DimensionMismatch {
            expected: n,
            actual: b.ncols(),
        });
    }

    // Use symbolic addition to find sparsity pattern
    let mut row_ptrs = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        let b_start = b.row_ptrs()[i];
        let b_end = b.row_ptrs()[i + 1];

        // Merge sorted column indices
        let mut a_ptr = a_start;
        let mut b_ptr = b_start;

        while a_ptr < a_end || b_ptr < b_end {
            let a_col = if a_ptr < a_end {
                Some(a.col_indices()[a_ptr])
            } else {
                None
            };
            let b_col = if b_ptr < b_end {
                Some(b.col_indices()[b_ptr])
            } else {
                None
            };

            match (a_col, b_col) {
                (Some(ac), Some(bc)) if ac < bc => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (Some(ac), Some(bc)) if ac > bc => {
                    col_indices.push(bc);
                    values.push(T::zero() - sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (Some(ac), Some(_bc)) => {
                    // ac == bc
                    let val = a.values()[a_ptr].clone() - sigma.clone() * b.values()[b_ptr].clone();
                    if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_ptr += 1;
                    b_ptr += 1;
                }
                (Some(ac), None) => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (None, Some(bc)) => {
                    col_indices.push(bc);
                    values.push(T::zero() - sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (None, None) => break,
            }
        }

        row_ptrs[i + 1] = col_indices.len();
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).map_err(|e| {
        EigenvalueError::ComputationError(format!(
            "failed to construct CSR matrix in subtract_scaled_matrices: {e}"
        ))
    })
}

/// Compute C = A + sigma * B (sparse matrix addition).
///
/// # Errors
///
/// Returns [`EigenvalueError::NotSquare`] if `a` is not square, or
/// [`EigenvalueError::DimensionMismatch`] if `b`'s dimensions do not match
/// `a`'s. Returns [`EigenvalueError::ComputationError`] if the resulting
/// sparsity pattern fails to form a valid CSR matrix (should not occur for
/// well-formed inputs, but is surfaced rather than panicking).
pub fn add_scaled_matrices<T: Scalar + Clone>(
    a: &CsrMatrix<T>,
    b: &CsrMatrix<T>,
    sigma: T,
) -> Result<CsrMatrix<T>, EigenvalueError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(EigenvalueError::NotSquare {
            nrows: n,
            ncols: a.ncols(),
        });
    }
    if b.nrows() != n {
        return Err(EigenvalueError::DimensionMismatch {
            expected: n,
            actual: b.nrows(),
        });
    }
    if b.ncols() != n {
        return Err(EigenvalueError::DimensionMismatch {
            expected: n,
            actual: b.ncols(),
        });
    }

    let mut row_ptrs = vec![0usize; n + 1];
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    for i in 0..n {
        let a_start = a.row_ptrs()[i];
        let a_end = a.row_ptrs()[i + 1];
        let b_start = b.row_ptrs()[i];
        let b_end = b.row_ptrs()[i + 1];

        let mut a_ptr = a_start;
        let mut b_ptr = b_start;

        while a_ptr < a_end || b_ptr < b_end {
            let a_col = if a_ptr < a_end {
                Some(a.col_indices()[a_ptr])
            } else {
                None
            };
            let b_col = if b_ptr < b_end {
                Some(b.col_indices()[b_ptr])
            } else {
                None
            };

            match (a_col, b_col) {
                (Some(ac), Some(bc)) if ac < bc => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (Some(ac), Some(bc)) if ac > bc => {
                    col_indices.push(bc);
                    values.push(sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (Some(ac), Some(_bc)) => {
                    let val = a.values()[a_ptr].clone() + sigma.clone() * b.values()[b_ptr].clone();
                    if Scalar::abs(val.clone()) > <T as Scalar>::epsilon() {
                        col_indices.push(ac);
                        values.push(val);
                    }
                    a_ptr += 1;
                    b_ptr += 1;
                }
                (Some(ac), None) => {
                    col_indices.push(ac);
                    values.push(a.values()[a_ptr].clone());
                    a_ptr += 1;
                }
                (None, Some(bc)) => {
                    col_indices.push(bc);
                    values.push(sigma.clone() * b.values()[b_ptr].clone());
                    b_ptr += 1;
                }
                (None, None) => break,
            }
        }

        row_ptrs[i + 1] = col_indices.len();
    }

    CsrMatrix::new(n, n, row_ptrs, col_indices, values).map_err(|e| {
        EigenvalueError::ComputationError(format!(
            "failed to construct CSR matrix in add_scaled_matrices: {e}"
        ))
    })
}

/// Convert CSR matrix to CSC format.
///
/// # Errors
///
/// Returns [`EigenvalueError::ComputationError`] if the resulting sparsity
/// pattern fails to form a valid CSC matrix (should not occur for a
/// well-formed `csr` input, but is surfaced rather than panicking).
pub fn csr_to_csc<T: Scalar + Clone>(csr: &CsrMatrix<T>) -> Result<CscMatrix<T>, EigenvalueError> {
    let nrows = csr.nrows();
    let ncols = csr.ncols();
    let nnz = csr.nnz();

    if nnz == 0 {
        return CscMatrix::new(nrows, ncols, vec![0; ncols + 1], vec![], vec![]).map_err(|e| {
            EigenvalueError::ComputationError(format!(
                "failed to construct empty CSC matrix in csr_to_csc: {e}"
            ))
        });
    }

    // Count entries per column
    let mut col_counts = vec![0usize; ncols];
    for &col in csr.col_indices() {
        col_counts[col] += 1;
    }

    // Build column pointers
    let mut col_ptrs = vec![0usize; ncols + 1];
    for j in 0..ncols {
        col_ptrs[j + 1] = col_ptrs[j] + col_counts[j];
    }

    // Fill row indices and values
    let mut row_indices = vec![0usize; nnz];
    let mut values = vec![T::zero(); nnz];
    let mut col_pos = col_ptrs[..ncols].to_vec();

    for i in 0..nrows {
        let row_start = csr.row_ptrs()[i];
        let row_end = csr.row_ptrs()[i + 1];
        for k in row_start..row_end {
            let j = csr.col_indices()[k];
            let pos = col_pos[j];
            row_indices[pos] = i;
            values[pos] = csr.values()[k].clone();
            col_pos[j] += 1;
        }
    }

    CscMatrix::new(nrows, ncols, col_ptrs, row_indices, values).map_err(|e| {
        EigenvalueError::ComputationError(format!(
            "failed to construct CSC matrix in csr_to_csc: {e}"
        ))
    })
}

// =============================================================================
// Dense symmetric eigenvalue problem (Jacobi)
// =============================================================================

/// Compute all eigenpairs of a small dense symmetric matrix via the cyclic
/// Jacobi rotation method.
///
/// `h` is an `m x m` symmetric matrix stored row-major (`h[i][j]`). The routine
/// applies two-sided Jacobi rotations `A <- J^T A J` to drive the off-diagonal
/// to zero, accumulating the rotations to build the eigenvectors. It returns
/// `(eigenvalues, eigenvectors)` sorted ascending by eigenvalue, where
/// `eigenvectors[k]` is the length-`m` (orthonormal) eigenvector associated
/// with `eigenvalues[k]`.
///
/// Jacobi is backward stable and computes even tiny/clustered eigenvalues to
/// high relative accuracy, which is exactly what a Rayleigh-Ritz projection
/// needs; it is intended for the small projected matrices (`m` at most a few
/// hundred) that arise there.
pub fn dense_symmetric_jacobi_evd<T>(h: &[Vec<T>]) -> (Vec<T>, Vec<Vec<T>>)
where
    T: Scalar<Real = T> + Clone + Field + Real + FromPrimitive,
{
    let m = h.len();
    if m == 0 {
        return (vec![], vec![]);
    }

    // Working copy of the matrix and the eigenvector accumulator (identity).
    let mut a: Vec<Vec<T>> = h.to_vec();
    let mut v: Vec<Vec<T>> = (0..m)
        .map(|i| {
            (0..m)
                .map(|j| if i == j { T::one() } else { T::zero() })
                .collect()
        })
        .collect();

    let eps = <T as Scalar>::epsilon();
    let two = T::from_f64(2.0).unwrap_or_else(|| T::one() + T::one());
    // A generous sweep budget; a symmetric matrix converges quadratically and
    // in practice needs far fewer than this many sweeps.
    let max_sweeps = 100usize;

    for _sweep in 0..max_sweeps {
        // Off-diagonal Frobenius norm; stop once it is negligible.
        let mut off = T::zero();
        for p in 0..m {
            for q in (p + 1)..m {
                off = off + a[p][q].clone() * a[p][q].clone();
            }
        }
        if Real::sqrt(off) <= eps {
            break;
        }

        for p in 0..m {
            for q in (p + 1)..m {
                let apq = a[p][q].clone();
                if Scalar::abs(apq.clone()) <= eps {
                    continue;
                }
                let app = a[p][p].clone();
                let aqq = a[q][q].clone();

                // Rotation that annihilates a[p][q]:
                //   theta = (aqq - app) / (2 * apq)
                //   t     = sign(theta) / (|theta| + sqrt(theta^2 + 1))
                //   c     = 1 / sqrt(t^2 + 1),  s = t * c
                let theta = (aqq - app) / (two.clone() * apq.clone());
                let sign = if theta >= T::zero() {
                    T::one()
                } else {
                    T::zero() - T::one()
                };
                let t = sign
                    / (Scalar::abs(theta.clone())
                        + Real::sqrt(theta.clone() * theta.clone() + T::one()));
                let c = T::one() / Real::sqrt(t.clone() * t.clone() + T::one());
                let s = t * c.clone();

                // Right-multiply by J (update columns p and q).
                for row in a.iter_mut() {
                    let akp = row[p].clone();
                    let akq = row[q].clone();
                    row[p] = c.clone() * akp.clone() - s.clone() * akq.clone();
                    row[q] = s.clone() * akp + c.clone() * akq;
                }
                // Left-multiply by J^T (update rows p and q).
                for k in 0..m {
                    let apk = a[p][k].clone();
                    let aqk = a[q][k].clone();
                    a[p][k] = c.clone() * apk.clone() - s.clone() * aqk.clone();
                    a[q][k] = s.clone() * apk + c.clone() * aqk;
                }
                // Accumulate the rotation into the eigenvector matrix.
                for row in v.iter_mut() {
                    let vkp = row[p].clone();
                    let vkq = row[q].clone();
                    row[p] = c.clone() * vkp.clone() - s.clone() * vkq.clone();
                    row[q] = s.clone() * vkp + c.clone() * vkq;
                }
            }
        }
    }

    // Eigenvalues sit on the diagonal; eigenvectors are the columns of `v`.
    let mut pairs: Vec<(T, Vec<T>)> = (0..m)
        .map(|j| {
            let eval = a[j][j].clone();
            let evec: Vec<T> = (0..m).map(|i| v[i][j].clone()).collect();
            (eval, evec)
        })
        .collect();
    pairs.sort_by(|(a_val, _), (b_val, _)| {
        a_val
            .partial_cmp(b_val)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let eigenvalues = pairs.iter().map(|(e, _)| e.clone()).collect();
    let eigenvectors = pairs.into_iter().map(|(_, vec)| vec).collect();
    (eigenvalues, eigenvectors)
}
