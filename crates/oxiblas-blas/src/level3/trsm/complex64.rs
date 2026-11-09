//! Complex64 TRSM entry points.
//!
//! These solve through the same blocked driver as the generic
//! [`super::trsm_in_place`] (register-tiled diagonal blocks, in-place updates
//! on views); for complex element types the off-diagonal updates use the 3M
//! GEMM, which is markedly faster here than the generic complex GEMM kernel.

use super::types::{Diag, Side, Trans, TrsmError, Uplo};
use num_complex::Complex64;
use oxiblas_matrix::{Mat, MatMut, MatRef};

/// Complex64 TRSM: solves `op(A)·X = alpha·B` (Left) or `X·op(A) = alpha·B`
/// (Right) and returns X.
///
/// `op(A)` is `A`, `Aᵀ` or `Aᴴ`. Sequential; the generic
/// [`super::trsm_with_par`] accepts `Complex64` as well and can run in parallel.
pub fn trsm_c64(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    alpha: Complex64,
    a: MatRef<'_, Complex64>,
    b: MatRef<'_, Complex64>,
) -> Result<Mat<Complex64>, TrsmError> {
    let m = b.nrows();
    let n = b.ncols();

    // Reference ?TRSM validates the shapes first and, for alpha == 0, sets X := 0 without
    // reading A or B (a singular A is accepted; NaN/Inf in A or B cannot leak into X).
    super::check_trsm_dims(side, a.nrows(), a.ncols(), m, n)?;
    if alpha == Complex64::new(0.0, 0.0) {
        return Ok(Mat::zeros(m, n));
    }

    // X := alpha·B; alpha == 1 skips the scaling as in reference ?TRSM (a
    // complex multiply by 1 + 0i is not an identity for infinite entries).
    let mut x: Mat<Complex64> = Mat::zeros(m, n);
    let scale = alpha != Complex64::new(1.0, 0.0);
    for j in 0..n {
        for i in 0..m {
            x[(i, j)] = if scale { alpha * b[(i, j)] } else { b[(i, j)] };
        }
    }

    trsm_c64_in_place(side, uplo, trans, diag, a, x.as_mut())?;
    Ok(x)
}

/// Complex64 TRSM in place: overwrites B with the solution of `op(A)·X = B`
/// (Left) or `X·op(A) = B` (Right).
///
/// Sequential. Errors and the reading of A are as for
/// [`super::trsm_in_place`] (B is unchanged when A is singular).
pub fn trsm_c64_in_place(
    side: Side,
    uplo: Uplo,
    trans: Trans,
    diag: Diag,
    a: MatRef<'_, Complex64>,
    b: MatMut<'_, Complex64>,
) -> Result<(), TrsmError> {
    super::generic::trsm_in_place(side, uplo, trans, diag, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Computes `A^H * X` explicitly, used both to build right-hand sides
    /// and as an independent residual check for Side::Left ConjTrans solves.
    fn conj_transpose_mul(a: &Mat<Complex64>, x: &Mat<Complex64>) -> Mat<Complex64> {
        let n = a.nrows();
        let ncols = x.ncols();
        let mut result: Mat<Complex64> = Mat::zeros(n, ncols);
        for j in 0..ncols {
            for i in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    // (A^H)[i, k] = conj(A[k, i])
                    sum += a[(k, i)].conj() * x[(k, j)];
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    /// Computes `X * A^H` explicitly, used both to build right-hand sides
    /// and as an independent residual check for Side::Right ConjTrans solves.
    fn mul_conj_transpose(x: &Mat<Complex64>, a: &Mat<Complex64>) -> Mat<Complex64> {
        let m = x.nrows();
        let n = a.nrows();
        let mut result: Mat<Complex64> = Mat::zeros(m, n);
        for i in 0..m {
            for j in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..n {
                    // (A^H)[k, j] = conj(A[j, k])
                    sum += x[(i, k)] * a[(j, k)].conj();
                }
                result[(i, j)] = sum;
            }
        }
        result
    }

    fn assert_mat_close(actual: &Mat<Complex64>, expected: &Mat<Complex64>, tol: f64, label: &str) {
        assert_eq!(actual.nrows(), expected.nrows());
        assert_eq!(actual.ncols(), expected.ncols());
        for i in 0..actual.nrows() {
            for j in 0..actual.ncols() {
                let diff = actual[(i, j)] - expected[(i, j)];
                assert!(
                    diff.norm() < tol,
                    "{label} mismatch at ({i}, {j}): got {:?}, expected {:?}",
                    actual[(i, j)],
                    expected[(i, j)]
                );
            }
        }
    }

    /// A lower-triangular matrix with genuinely complex entries both on and
    /// off the diagonal (Im != 0 everywhere non-zero), used across the
    /// ConjTrans regression tests below.
    fn complex_lower_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[
                Complex64::new(2.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(1.0, -2.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            &[
                Complex64::new(0.0, 3.0),
                Complex64::new(-1.0, 2.0),
                Complex64::new(4.0, -1.0),
            ],
        ])
    }

    /// An upper-triangular counterpart with genuinely complex entries both
    /// on and off the diagonal.
    fn complex_upper_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[
                Complex64::new(2.0, 1.0),
                Complex64::new(1.0, -2.0),
                Complex64::new(0.0, 3.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(-1.0, 2.0),
            ],
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, -1.0),
            ],
        ])
    }

    /// Arbitrary genuinely-complex 3x2 solution used for Side::Left tests.
    fn x_left_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[Complex64::new(1.0, 1.0), Complex64::new(0.5, -0.5)],
            &[Complex64::new(2.0, -1.0), Complex64::new(-1.0, 0.0)],
            &[Complex64::new(0.0, 2.0), Complex64::new(3.0, 1.0)],
        ])
    }

    /// Arbitrary genuinely-complex 2x3 solution used for Side::Right tests.
    fn x_right_fixture() -> Mat<Complex64> {
        Mat::from_rows(&[
            &[
                Complex64::new(1.0, 1.0),
                Complex64::new(2.0, -1.0),
                Complex64::new(0.0, 2.0),
            ],
            &[
                Complex64::new(0.5, -0.5),
                Complex64::new(-1.0, 0.0),
                Complex64::new(3.0, 1.0),
            ],
        ])
    }

    // ------------------------------------------------------------------
    // Regression tests for Finding #1 (complex64.rs:352): every diagonal
    // read in this file divided by the UNCONJUGATED diagonal for
    // Trans::ConjTrans. All four Left/Right x Lower/Upper branches are
    // covered (they route through trsm_c64_naive_submatrix and
    // trsm_c64_naive_submatrix_right respectively), each with genuinely
    // complex on- and off-diagonal entries so a real-only or
    // diagonal-only-complex matrix could not have caught the bug.
    // ------------------------------------------------------------------

    #[test]
    fn test_trsm_c64_conjtrans_left_lower_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_left_fixture();
        // Solve A^H * X = B, so B = A^H * X_expected.
        let b = conj_transpose_mul(&a, &x_expected);

        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        // Independent residual check: reconstruct A^H * X_solved and
        // compare it against the original right-hand side B.
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_c64_conjtrans_left_upper_complex() {
        let a = complex_upper_fixture();
        let x_expected = x_left_fixture();
        let b = conj_transpose_mul(&a, &x_expected);

        let x = trsm_c64(
            Side::Left,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-9, "A^H * X_solved vs B");
    }

    #[test]
    fn test_trsm_c64_conjtrans_right_lower_complex() {
        let a = complex_lower_fixture();
        let x_expected = x_right_fixture();
        // Solve X * A^H = B, so B = X_expected * A^H.
        let b = mul_conj_transpose(&x_expected, &a);

        let x = trsm_c64(
            Side::Right,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }

    #[test]
    fn test_trsm_c64_conjtrans_right_upper_complex() {
        let a = complex_upper_fixture();
        let x_expected = x_right_fixture();
        let b = mul_conj_transpose(&x_expected, &a);

        let x = trsm_c64(
            Side::Right,
            Uplo::Upper,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-9, "solved X vs expected X");
        let reconstructed = mul_conj_transpose(&x, &a);
        assert_mat_close(&reconstructed, &b, 1e-9, "X_solved * A^H vs B");
    }

    /// Exercises the blocked path (order 140 spans two 128-row diagonal
    /// blocks) with a genuinely complex ConjTrans lower-triangular solve, so
    /// the fix is verified through the diagonal-block solver AND the 3M-GEMM
    /// off-diagonal update working together.
    #[test]
    fn test_trsm_c64_conjtrans_blocked_large_complex() {
        let n = 140;
        let rhs_cols = 5;

        // Lower-triangular A with a genuinely complex diagonal and
        // genuinely complex sub-diagonal band (off-diagonal terms).
        let mut a: Mat<Complex64> = Mat::zeros(n, n);
        for i in 0..n {
            a[(i, i)] = Complex64::new(3.0, 0.7);
            for j in 0..i {
                // Small-magnitude complex off-diagonal entries keep the
                // matrix well-conditioned while still exercising both the
                // real and imaginary parts of the ConjTrans update.
                a[(i, j)] = Complex64::new(0.01, -0.02);
            }
        }

        let mut x_expected: Mat<Complex64> = Mat::zeros(n, rhs_cols);
        for i in 0..n {
            for j in 0..rhs_cols {
                x_expected[(i, j)] = Complex64::new(1.0 + i as f64 * 0.01, 0.5 - j as f64 * 0.02);
            }
        }

        let b = conj_transpose_mul(&a, &x_expected);

        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            a.as_ref(),
            b.as_ref(),
        )
        .expect("diagonal entries are all non-zero");

        assert_mat_close(&x, &x_expected, 1e-6, "solved X vs expected X");
        let reconstructed = conj_transpose_mul(&a, &x);
        assert_mat_close(&reconstructed, &b, 1e-6, "A^H * X_solved vs B");
    }
}
