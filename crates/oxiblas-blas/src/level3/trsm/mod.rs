//! TRSM: Triangular solve for matrices.
//!
//! Solves A·X = α·B or X·A = α·B where A is triangular.

pub mod complex32;
pub mod complex64;
pub mod generic;
pub mod types;

pub use complex32::{trsm_c32, trsm_c32_in_place};
pub use complex64::{trsm_c64, trsm_c64_in_place};
pub use generic::{trsm, trsm_in_place};
pub use types::{Diag, Side, Trans, TrsmError, Uplo};

/// Validates the shapes of a TRSM call: `A` must be square and match the relevant
/// dimension of the `m x n` right-hand side (`m` for `Side::Left`, `n` for
/// `Side::Right`). Reference ?TRSM checks this before anything else, so it also
/// applies when `alpha == 0` and the solve itself is skipped.
fn check_trsm_dims(
    side: Side,
    a_rows: usize,
    a_cols: usize,
    m: usize,
    n: usize,
) -> Result<(), TrsmError> {
    if a_rows != a_cols {
        return Err(TrsmError::NotSquare);
    }
    let expected = match side {
        Side::Left => m,
        Side::Right => n,
    };
    if a_rows != expected {
        return Err(TrsmError::DimensionMismatch);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::{Complex32, Complex64};
    use oxiblas_matrix::Mat;

    #[test]
    fn test_trsm_left_lower() {
        // Solve L·X = B
        // L = [[2, 0, 0], [1, 3, 0], [2, 1, 4]]
        // X = [[2, 4], [1, 2], [2, 4]]
        // B = L·X = [[4, 8], [5, 10], [13, 26]]
        let l = Mat::from_rows(&[&[2.0f64, 0.0, 0.0], &[1.0, 3.0, 0.0], &[2.0, 1.0, 4.0]]);
        let b = Mat::from_rows(&[&[4.0f64, 8.0], &[5.0, 10.0], &[13.0, 26.0]]);

        let x = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)] - 2.0).abs() < 1e-10);
        assert!((x[(0, 1)] - 4.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(1, 1)] - 2.0).abs() < 1e-10);
        assert!((x[(2, 0)] - 2.0).abs() < 1e-10);
        assert!((x[(2, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_left_upper() {
        // Solve U·X = B
        // U = [[2, 1, 2], [0, 3, 1], [0, 0, 4]]
        // X = [[1, 2], [2, 3], [3, 4]]
        // B = U·X = [[2+2+6, 4+3+8], [6+3, 9+4], [12, 16]] = [[10, 15], [9, 13], [12, 16]]
        let u = Mat::from_rows(&[&[2.0f64, 1.0, 2.0], &[0.0, 3.0, 1.0], &[0.0, 0.0, 4.0]]);
        let b = Mat::from_rows(&[&[10.0f64, 15.0], &[9.0, 13.0], &[12.0, 16.0]]);

        let x = trsm(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            u.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 2.0).abs() < 1e-10);
        assert!((x[(1, 1)] - 3.0).abs() < 1e-10);
        assert!((x[(2, 0)] - 3.0).abs() < 1e-10);
        assert!((x[(2, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_right_lower() {
        // Solve X·L = B
        // L = [[2, 0], [1, 3]]
        // X = [[1, 2], [3, 4]]
        // B = X·L = [[2+2, 6], [6+4, 12]] = [[4, 6], [10, 12]]
        let l = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[&[4.0f64, 6.0], &[10.0, 12.0]]);

        let x = trsm(
            Side::Right,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((x[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_right_upper() {
        // Solve X·U = B
        // U = [[2, 1], [0, 3]]
        // X = [[1, 2], [3, 4]]
        // B = X·U = [[2, 1+6], [6, 3+12]] = [[2, 7], [6, 15]]
        let u = Mat::from_rows(&[&[2.0f64, 1.0], &[0.0, 3.0]]);
        let b = Mat::from_rows(&[&[2.0f64, 7.0], &[6.0, 15.0]]);

        let x = trsm(
            Side::Right,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            u.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(0, 1)] - 2.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 3.0).abs() < 1e-10);
        assert!((x[(1, 1)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_unit_diagonal() {
        // Solve L·X = B with unit diagonal
        // L = [[1, 0], [2, 1]] (unit diagonal)
        // X = [[3], [4]]
        // B = L·X = [[3], [6+4]] = [[3], [10]]
        let l = Mat::from_rows(&[
            &[999.0f64, 0.0], // diagonal value ignored
            &[2.0, 999.0],
        ]);
        let b = Mat::from_rows(&[&[3.0f64], &[10.0]]);

        let x = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::Unit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)] - 3.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_alpha() {
        // Solve L·X = α·B with alpha = 2
        let l = Mat::from_rows(&[&[2.0f64, 0.0], &[0.0, 2.0]]);
        let b = Mat::from_rows(&[&[4.0f64], &[6.0]]);

        let x = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            2.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // X should be [4, 6] (B scaled by 2, then divided by 2)
        assert!((x[(0, 0)] - 4.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_transpose() {
        // Solve L^T·X = B where L^T is upper triangular
        let l = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        // L^T = [[2, 1], [0, 3]]
        // X = [[1], [2]]
        // B = L^T·X = [[2+2], [6]] = [[4], [6]]
        let b = Mat::from_rows(&[&[4.0f64], &[6.0]]);

        let x = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::Trans,
            Diag::NonUnit,
            1.0,
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)] - 1.0).abs() < 1e-10);
        assert!((x[(1, 0)] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_singular() {
        let a = Mat::from_rows(&[
            &[2.0f64, 0.0],
            &[1.0, 0.0], // Zero on diagonal
        ]);
        let b = Mat::from_rows(&[&[4.0f64], &[5.0]]);

        let result = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            a.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(result, Err(TrsmError::Singular)));
    }

    #[test]
    fn test_trsm_dimension_mismatch() {
        let a = Mat::from_rows(&[&[2.0f64, 0.0], &[1.0, 3.0]]);
        let b = Mat::from_rows(&[
            &[4.0f64],
            &[5.0],
            &[6.0], // Wrong size
        ]);

        let result = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            a.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(result, Err(TrsmError::DimensionMismatch)));
    }

    #[test]
    fn test_trsm_identity() {
        let eye = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let b = Mat::from_rows(&[&[1.0f64, 2.0], &[3.0, 4.0], &[5.0, 6.0]]);

        let x = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            eye.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // X should equal B
        for i in 0..3 {
            for j in 0..2 {
                assert!((x[(i, j)] - b[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_trsm_empty() {
        let a: Mat<f64> = Mat::zeros(0, 0);
        let b: Mat<f64> = Mat::zeros(0, 2);

        let x = trsm(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            a.as_ref(),
            b.as_ref(),
        )
        .unwrap();
        assert_eq!(x.nrows(), 0);
        assert_eq!(x.ncols(), 2);
    }

    // ========================================================================
    // Complex TRSM tests
    // ========================================================================

    #[test]
    fn test_trsm_c64_left_lower() {
        // Solve L·X = B for complex matrices
        // L = [[2, 0], [1+i, 3]]
        // X = [[1+i], [2]]
        // B = L·X = [[2*(1+i)], [(1+i)*(1+i) + 3*2]]
        //         = [[2+2i], [(1+2i-1) + 6]] = [[2+2i], [6+2i]]
        let l = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);
        let b = Mat::from_rows(&[&[Complex64::new(2.0, 2.0)], &[Complex64::new(6.0, 2.0)]]);

        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((x[(0, 0)].im - 1.0).abs() < 1e-10);
        assert!((x[(1, 0)].re - 2.0).abs() < 1e-10);
        assert!((x[(1, 0)].im).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_c64_left_upper() {
        // Solve U·X = B
        // U = [[2, 1], [0, 3]]
        // X = [[1], [2]]
        // B = U·X = [[2+2], [6]] = [[4], [6]]
        let u = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            &[Complex64::new(0.0, 0.0), Complex64::new(3.0, 0.0)],
        ]);
        let b = Mat::from_rows(&[&[Complex64::new(4.0, 0.0)], &[Complex64::new(6.0, 0.0)]]);

        let x = trsm_c64(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            u.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((x[(1, 0)].re - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_c64_conjtrans() {
        // Solve L^H·X = B (conjugate transpose)
        // L = [[2, 0], [1+i, 3]]
        // L^H = [[2, 1-i], [0, 3]]
        let l = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(1.0, 1.0), Complex64::new(3.0, 0.0)],
        ]);
        // X = [[1], [1]]
        // B = L^H · X = [[2*1 + (1-i)*1], [0*1 + 3*1]] = [[3-i], [3]]
        let b = Mat::from_rows(&[&[Complex64::new(3.0, -1.0)], &[Complex64::new(3.0, 0.0)]]);

        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((x[(0, 0)].im).abs() < 1e-10);
        assert!((x[(1, 0)].re - 1.0).abs() < 1e-10);
        assert!((x[(1, 0)].im).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_c64_right() {
        // Solve X·L = B
        // L = [[2, 0], [1, 3]]
        // X = [[1, 2]]
        // B = X·L = [[1*2 + 2*1, 1*0 + 2*3]] = [[4, 6]]
        let l = Mat::from_rows(&[
            &[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
            &[Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ]);
        let b = Mat::from_rows(&[&[Complex64::new(4.0, 0.0), Complex64::new(6.0, 0.0)]]);

        let x = trsm_c64(
            Side::Right,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        assert!((x[(0, 0)].re - 1.0).abs() < 1e-10);
        assert!((x[(0, 1)].re - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_trsm_c32_basic() {
        // Basic Complex32 test
        let l = Mat::from_rows(&[
            &[Complex32::new(2.0, 0.0), Complex32::new(0.0, 0.0)],
            &[Complex32::new(1.0, 0.0), Complex32::new(3.0, 0.0)],
        ]);
        let b = Mat::from_rows(&[&[Complex32::new(4.0, 0.0)], &[Complex32::new(8.0, 0.0)]]);

        let x = trsm_c32(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex32::new(1.0, 0.0),
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // x[0] = 4/2 = 2
        // x[1] = (8 - 1*2)/3 = 6/3 = 2
        assert!((x[(0, 0)].re - 2.0).abs() < 1e-5);
        assert!((x[(1, 0)].re - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_trsm_c64_large_blocked() {
        // Test blocked algorithm with larger matrix (> 64)
        let n = 100;
        let m = 80;

        // Create a lower triangular matrix with 2 on diagonal
        let mut l: Mat<Complex64> = Mat::zeros(n, n);
        for i in 0..n {
            l[(i, i)] = Complex64::new(2.0, 0.0);
            for j in 0..i {
                l[(i, j)] = Complex64::new(0.1, 0.0);
            }
        }

        // Create B
        let b: Mat<Complex64> = Mat::filled(n, m, Complex64::new(1.0, 0.0));

        // Solve L·X = B
        let x = trsm_c64(
            Side::Left,
            Uplo::Lower,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex64::new(1.0, 0.0),
            l.as_ref(),
            b.as_ref(),
        )
        .unwrap();

        // Verify: L·X should equal B
        for j in 0..m {
            for i in 0..n {
                let mut sum = Complex64::new(0.0, 0.0);
                for k in 0..=i {
                    sum = sum + l[(i, k)] * x[(k, j)];
                }
                assert!(
                    (sum.re - b[(i, j)].re).abs() < 1e-8,
                    "Mismatch at ({}, {}): got {}, expected {}",
                    i,
                    j,
                    sum.re,
                    b[(i, j)].re
                );
            }
        }
    }

    // Issue #9 class: reference ?TRSM with alpha == 0 sets B := 0 without reading A or B
    // (so NaN / Inf in either operand cannot leak, and even a singular A is accepted), after
    // the shapes have been validated.

    #[test]
    fn test_issue_9_trsm_alpha_zero_zero_fills_without_reading_a_or_b() {
        for side in [Side::Left, Side::Right] {
            for uplo in [Uplo::Lower, Uplo::Upper] {
                for trans in [Trans::NoTrans, Trans::Trans] {
                    for diag in [Diag::NonUnit, Diag::Unit] {
                        // B is 2x3; A is 2x2 (Left) or 3x3 (Right).
                        let ka = if side == Side::Left { 2 } else { 3 };
                        let a: Mat<f64> = Mat::filled(ka, ka, f64::NAN);
                        let b: Mat<f64> = Mat::filled(2, 3, f64::INFINITY);
                        let x = trsm(side, uplo, trans, diag, 0.0, a.as_ref(), b.as_ref())
                            .expect("alpha == 0 never inspects A");
                        for j in 0..3 {
                            for i in 0..2 {
                                assert!(
                                    x[(i, j)] == 0.0,
                                    "{side:?}/{uplo:?}/{trans:?}/{diag:?}: x[{i},{j}] = {}",
                                    x[(i, j)]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_issue_9_trsm_alpha_zero_accepts_singular_a_but_validates_shapes() {
        let singular = Mat::from_rows(&[&[0.0f64, 1.0], &[0.0, 0.0]]);
        let b = Mat::from_rows(&[&[f64::NAN, 1.0], &[2.0, f64::NAN]]);
        let x = trsm(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            0.0,
            singular.as_ref(),
            b.as_ref(),
        )
        .expect("A is never read for alpha == 0, so singularity is irrelevant");
        assert!(x[(0, 0)] == 0.0 && x[(0, 1)] == 0.0 && x[(1, 0)] == 0.0 && x[(1, 1)] == 0.0);

        // Shapes are validated first, as in the reference.
        let wrong = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0], &[0.0, 0.0, 1.0]]);
        let r = trsm(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            0.0,
            wrong.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(r, Err(TrsmError::DimensionMismatch)));
        let rect = Mat::from_rows(&[&[1.0f64, 0.0, 0.0], &[0.0, 1.0, 0.0]]);
        let r = trsm(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            0.0,
            rect.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(r, Err(TrsmError::NotSquare)));
    }

    #[test]
    fn test_issue_9_trsm_c64_alpha_zero_zero_fills_without_reading_a_or_b() {
        let nan = Complex64::new(f64::NAN, f64::NAN);
        let inf = Complex64::new(f64::INFINITY, f64::NEG_INFINITY);
        for side in [Side::Left, Side::Right] {
            for trans in [Trans::NoTrans, Trans::ConjTrans] {
                let ka = if side == Side::Left { 2 } else { 3 };
                let a: Mat<Complex64> = Mat::filled(ka, ka, nan);
                let b: Mat<Complex64> = Mat::filled(2, 3, inf);
                let x = trsm_c64(
                    side,
                    Uplo::Lower,
                    trans,
                    Diag::NonUnit,
                    Complex64::new(0.0, 0.0),
                    a.as_ref(),
                    b.as_ref(),
                )
                .expect("alpha == 0 never inspects A");
                for j in 0..3 {
                    for i in 0..2 {
                        assert!(x[(i, j)] == Complex64::new(0.0, 0.0), "{:?}", x[(i, j)]);
                    }
                }
            }
        }
        let singular: Mat<Complex64> = Mat::zeros(2, 2);
        let b: Mat<Complex64> = Mat::filled(2, 3, inf);
        let x = trsm_c64(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex64::new(0.0, 0.0),
            singular.as_ref(),
            b.as_ref(),
        );
        assert!(x.is_ok(), "singular A must be accepted for alpha == 0");
        let wrong: Mat<Complex64> = Mat::zeros(3, 3);
        let r = trsm_c64(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex64::new(0.0, 0.0),
            wrong.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(r, Err(TrsmError::DimensionMismatch)));
    }

    #[test]
    fn test_issue_9_trsm_c32_alpha_zero_zero_fills_without_reading_a_or_b() {
        let nan = Complex32::new(f32::NAN, f32::NAN);
        let inf = Complex32::new(f32::INFINITY, f32::NEG_INFINITY);
        for side in [Side::Left, Side::Right] {
            for trans in [Trans::NoTrans, Trans::ConjTrans] {
                let ka = if side == Side::Left { 2 } else { 3 };
                let a: Mat<Complex32> = Mat::filled(ka, ka, nan);
                let b: Mat<Complex32> = Mat::filled(2, 3, inf);
                let x = trsm_c32(
                    side,
                    Uplo::Lower,
                    trans,
                    Diag::NonUnit,
                    Complex32::new(0.0, 0.0),
                    a.as_ref(),
                    b.as_ref(),
                )
                .expect("alpha == 0 never inspects A");
                for j in 0..3 {
                    for i in 0..2 {
                        assert!(x[(i, j)] == Complex32::new(0.0, 0.0), "{:?}", x[(i, j)]);
                    }
                }
            }
        }
        let singular: Mat<Complex32> = Mat::zeros(2, 2);
        let b: Mat<Complex32> = Mat::filled(2, 3, inf);
        let x = trsm_c32(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex32::new(0.0, 0.0),
            singular.as_ref(),
            b.as_ref(),
        );
        assert!(x.is_ok(), "singular A must be accepted for alpha == 0");
        let wrong: Mat<Complex32> = Mat::zeros(3, 3);
        let r = trsm_c32(
            Side::Left,
            Uplo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            Complex32::new(0.0, 0.0),
            wrong.as_ref(),
            b.as_ref(),
        );
        assert!(matches!(r, Err(TrsmError::DimensionMismatch)));
    }
}
