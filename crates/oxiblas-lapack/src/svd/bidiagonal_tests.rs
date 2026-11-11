//! Tall-shape and equivalence tests of the one-sided Jacobi [`Svd`].
//!
//! `Svd::compute` used to complete `U` to an `m x m` orthogonal matrix by
//! retrying the standard basis vectors `e_0, e_1, ...` from scratch for every
//! new column, each retry a full Gram-Schmidt pass against all previous
//! columns: `O(m^4)` work for an `m x n` matrix with `m >> n` (about two
//! minutes for 300 x 31 in a debug build, hours for 1000 x 20), and a single
//! pass that left the completed columns orthogonal to the leading ones only to
//! about `1e-11`. The tall-shape tests below finish quickly in a debug build
//! and check orthogonality to `1e-12` (every entry of `U^T U - I` for
//! `m <= 400`; for 1000 x 20 the entries listed on [`assert_orthogonal`],
//! plus probes); the oracle tests check that the singular values, `V` and the
//! leading columns of `U` are bit-identical to the previous release and that
//! the completed columns are the same vectors up to rounding.

use super::*;

/// Deterministic `m x n` matrix with entries in `[-1, 1)`.
fn pseudo_random(m: usize, n: usize, seed: u64) -> Mat<f64> {
    let mut state = seed;
    let mut a = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            a[(i, j)] = ((state >> 11) as f64) / ((1u64 << 53) as f64) * 2.0 - 1.0;
        }
    }
    a
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// `max |Q^T Q - I|` over the columns of `q` (contiguous column slices).
fn orthogonality_error(q: MatRef<'_, f64>) -> f64 {
    let cols: Vec<&[f64]> = (0..q.ncols())
        .map(|j| q.col_as_slice(j).expect("contiguous column"))
        .collect();
    let mut worst = 0.0_f64;
    for i in 0..cols.len() {
        for j in 0..=i {
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((dot(cols[i], cols[j]) - expected).abs());
        }
    }
    worst
}

/// Deterministic probe vectors of length `m` with entries in `[-1, 1)`.
fn probes(m: usize, count: usize) -> Vec<Vec<f64>> {
    let a = pseudo_random(m, count, 99);
    (0..count)
        .map(|j| {
            a.as_ref()
                .col_as_slice(j)
                .expect("contiguous column")
                .to_vec()
        })
        .collect()
}

/// Number of trailing columns checked exactly against all columns when the
/// full product is not formed: the accepted residuals of the completion are
/// smallest, and orthogonality most at risk, in the last columns.
const EXACT_TAIL: usize = 64;

/// Orthogonality of the `m x m` matrix `u` whose first `k` columns come from
/// the Jacobi iteration.
///
/// For `m <= 400` every entry of `U^T U - I` is checked against `tol`. For
/// larger `m` (where the full `O(m^3)` product would dominate a debug run) the
/// entries checked exactly against `tol` are: every diagonal entry (unit
/// norms), every product of one of the first `k` columns with any column, and
/// every product of one of the last [`EXACT_TAIL`] columns with any column.
/// The products among the other completed columns are covered by probes:
/// `max_p |(U_c^T U_c x - x)_p| <= tol * max_q |x_q|` for deterministic `x`
/// with entries in `[-1, 1)`, which a single entry of `U_c^T U_c - I` of a few
/// times `tol` fails unless the probes' matching entries are all small.
fn assert_orthogonal(u: MatRef<'_, f64>, k: usize, tol: f64, label: &str) {
    let m = u.ncols();
    if m <= 400 {
        let err = orthogonality_error(u);
        assert!(err <= tol, "{label}: max |U^T U - I| = {err:e}");
        return;
    }
    let cols: Vec<&[f64]> = (0..m)
        .map(|j| u.col_as_slice(j).expect("contiguous column"))
        .collect();
    let tail_start = m - EXACT_TAIL.min(m - k);
    let mut worst = 0.0_f64;
    for (i, ci) in cols.iter().enumerate() {
        worst = worst.max((dot(ci, ci) - 1.0).abs());
        let exact = if i >= tail_start { i } else { k.min(i) };
        for cj in cols.iter().take(exact) {
            worst = worst.max(dot(ci, cj).abs());
        }
    }
    assert!(
        worst <= tol,
        "{label}: max |U^T U - I| over the exactly checked entries = {worst:e}"
    );
    let completed = &cols[k..];
    for x in probes(completed.len(), 4) {
        // y = U_c x, then compare U_c^T y with x.
        let mut y = vec![0.0; m];
        for (c, &xj) in completed.iter().zip(&x) {
            for (yi, &ci) in y.iter_mut().zip(c.iter()) {
                *yi += ci * xj;
            }
        }
        let x_max = x.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let err = completed
            .iter()
            .zip(&x)
            .map(|(c, &xj)| (dot(c, &y) - xj).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            err <= tol * x_max,
            "{label}: probe max |U_c^T U_c x - x| = {err:e} (max |x| = {x_max:e})"
        );
    }
}

/// `U` is `m x m` orthogonal, `V` is `n x n` orthogonal, the singular values
/// are non-negative and descending, and `A = U Σ V^T`.
fn check_svd(a: &Mat<f64>) {
    let (m, n) = (a.nrows(), a.ncols());
    let label = format!("{m}x{n}");
    let svd = Svd::compute(a.as_ref()).expect("SVD");
    let (u, vt, sigma) = (svd.u(), svd.vt(), svd.singular_values());
    assert_eq!((u.nrows(), u.ncols()), (m, m));
    assert_eq!((vt.nrows(), vt.ncols()), (n, n));
    assert_eq!(sigma.len(), m.min(n));
    assert!(
        sigma.iter().all(|&s| s >= 0.0),
        "{label}: negative singular value"
    );
    assert!(
        sigma.windows(2).all(|w| w[0] >= w[1]),
        "{label}: singular values not sorted"
    );

    assert_orthogonal(u, m.min(n), 1e-12, &label);
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            v[(j, i)] = vt[(i, j)];
        }
    }
    let v_err = orthogonality_error(v.as_ref());
    assert!(v_err <= 1e-12, "{label}: max |V^T V - I| = {v_err:e}");

    let rec = svd.reconstruct();
    let mut worst = 0.0_f64;
    for j in 0..n {
        for i in 0..m {
            worst = worst.max((rec[(i, j)] - a[(i, j)]).abs());
        }
    }
    assert!(worst <= 1e-12, "{label}: max |U S V^T - A| = {worst:e}");
}

#[test]
fn test_tall_300x31() {
    check_svd(&pseudo_random(300, 31, 1));
}

#[test]
fn test_tall_1000x20() {
    check_svd(&pseudo_random(1000, 20, 2));
}

/// Rank-deficient tall matrix (repeated and zero columns): the columns of `U`
/// for the zero singular values come from the completion as well.
#[test]
fn test_tall_rank_deficient_300x31() {
    let mut a = pseudo_random(300, 31, 3);
    for i in 0..300 {
        a[(i, 5)] = a[(i, 2)];
        a[(i, 17)] = 0.0;
        a[(i, 30)] = a[(i, 9)] * 0.5;
    }
    let svd = Svd::compute(a.as_ref()).expect("SVD");
    assert_eq!(svd.rank(1e-10), 28);
    check_svd(&a);
}

/// Wide and square shapes need no completion beyond the zero singular values.
#[test]
fn test_wide_and_square() {
    check_svd(&pseudo_random(20, 120, 4));
    check_svd(&pseudo_random(60, 60, 5));
}

/// How the completed columns of `U` must compare with the previous release.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Completion {
    /// Bit for bit (inputs whose completion involves no rounding, such as
    /// coordinate columns).
    Exact,
    /// Within the rounding error of the previous release: `4 e + 64 eps`,
    /// where `e = max |U^T U - I|` of its (single-pass) `U`.
    Rounding,
}

/// Compares with the previous release (kept below as [`reference_compute`]):
/// the singular values, `V^T` and the columns of `U` that come from the
/// Jacobi iteration (singular values above `100 eps`) are bit-identical. The
/// completed columns of `U` are the same vectors, computed with a
/// reorthogonalized Gram-Schmidt process; `completion` says how close they
/// must be.
fn assert_matches_reference<T: Field + Real + bytemuck::Zeroable>(
    a: &Mat<T>,
    label: &str,
    completion: Completion,
) where
    T: core::fmt::Debug,
{
    let svd = Svd::compute(a.as_ref()).expect("SVD");
    let (u, sigma, vt) = reference_compute(a.as_ref()).expect("reference SVD");
    let (m, n) = (a.nrows(), a.ncols());
    // f32 -> f64 is exact (and keeps the sign of zero and NaN payloads), so
    // comparing the f64 bit patterns compares the values bit for bit.
    let to_f64 = |x: T| num_traits::ToPrimitive::to_f64(&x).unwrap_or(f64::NAN);
    let same = |x: T, y: T| to_f64(x).to_bits() == to_f64(y).to_bits();
    let completion_tol = match completion {
        Completion::Exact => 0.0,
        Completion::Rounding => {
            let u_f64: Vec<Vec<f64>> = (0..m)
                .map(|j| (0..m).map(|i| to_f64(u[(i, j)])).collect())
                .collect();
            let mut old_err = 0.0_f64;
            for i in 0..m {
                for j in 0..=i {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    let d = dot(&u_f64[i], &u_f64[j]) - expected;
                    if d.is_finite() {
                        old_err = old_err.max(d.abs());
                    }
                }
            }
            let eps = to_f64(<T as Scalar>::epsilon());
            4.0 * old_err + 64.0 * eps
        }
    };
    for (i, (&s, &r)) in svd.singular_values().iter().zip(&sigma).enumerate() {
        assert!(same(s, r), "{label}: sigma[{i}] = {s:?}, previously {r:?}");
    }
    for j in 0..n {
        for i in 0..n {
            assert!(
                same(svd.vt()[(i, j)], vt[(i, j)]),
                "{label}: Vt[{i},{j}] differs"
            );
        }
    }
    let tol = <T as Scalar>::epsilon() * T::from_f64(100.0).unwrap_or(T::one());
    for j in 0..m {
        let from_jacobi = j < sigma.len() && sigma[j] > tol;
        for i in 0..m {
            let (x, y) = (svd.u()[(i, j)], u[(i, j)]);
            let finite = to_f64(x).is_finite() && to_f64(y).is_finite();
            if from_jacobi || completion == Completion::Exact || !finite {
                assert!(same(x, y), "{label}: U[{i},{j}] = {x:?}, previously {y:?}");
            } else {
                let diff = (to_f64(x) - to_f64(y)).abs();
                assert!(
                    diff <= completion_tol,
                    "{label}: U[{i},{j}] = {x:?}, previously {y:?} (difference {diff:e})"
                );
            }
        }
    }
}

#[test]
fn test_completion_matches_previous_release() {
    for (m, n, seed) in [
        (30, 7, 11),
        (40, 3, 12),
        (12, 12, 13),
        (7, 19, 14),
        (1, 5, 15),
        (5, 1, 16),
        (2, 1, 17),
        (33, 32, 18),
    ] {
        assert_matches_reference(
            &pseudo_random(m, n, seed),
            &format!("random {m}x{n}"),
            Completion::Rounding,
        );
    }

    // Rank deficient: repeated, scaled and zero columns.
    let mut a = pseudo_random(25, 6, 19);
    for i in 0..25 {
        a[(i, 3)] = a[(i, 1)];
        a[(i, 4)] = 0.0;
        a[(i, 5)] = -2.0 * a[(i, 0)];
    }
    assert_matches_reference(&a, "rank deficient 25x6", Completion::Rounding);

    // Coordinate-aligned columns: e_b in the span of the leading columns of
    // U is rejected for every later column; no rounding is involved.
    let mut padded = Mat::zeros(20, 4);
    for (j, &(row, value)) in [(0, 3.0), (1, -1.0), (7, 2.0), (2, 0.5)].iter().enumerate() {
        padded[(row, j)] = value;
    }
    assert_matches_reference(&padded, "coordinate columns 20x4", Completion::Exact);
    let identity_padded = {
        let mut m = Mat::zeros(15, 5);
        for j in 0..5 {
            m[(j, j)] = 1.0;
        }
        m
    };
    assert_matches_reference(&identity_padded, "identity padded 15x5", Completion::Exact);

    assert_matches_reference(&Mat::<f64>::zeros(9, 3), "zero 9x3", Completion::Exact);

    // A NaN entry: every candidate is rejected for the columns it reaches,
    // which stay zero.
    let mut nan = pseudo_random(18, 5, 20);
    nan[(4, 2)] = f64::NAN;
    assert_matches_reference(&nan, "NaN 18x5", Completion::Rounding);

    // Single precision.
    let a32: Mat<f32> = {
        let a = pseudo_random(30, 7, 21);
        let mut m = Mat::zeros(30, 7);
        for j in 0..7 {
            for i in 0..30 {
                m[(i, j)] = a[(i, j)] as f32;
            }
        }
        m
    };
    assert_matches_reference(&a32, "f32 random 30x7", Completion::Rounding);
}

/// The completed columns of `U` are orthogonal to the leading ones to working
/// precision. The previous release orthogonalized each candidate once and
/// lost about three orders of magnitude here (`|U_k^T U_c|` about `1e-11` for
/// this 300 x 31 matrix, against `2e-14` for `|U_k^T U_k - I|`).
#[test]
fn test_completion_orthogonal_to_leading_columns() {
    let a = pseudo_random(300, 31, 1);
    let svd = Svd::compute(a.as_ref()).expect("SVD");
    let u = svd.u();
    let cols: Vec<&[f64]> = (0..300)
        .map(|j| u.col_as_slice(j).expect("contiguous column"))
        .collect();
    let mut worst = 0.0_f64;
    for j in 31..300 {
        for leading in cols.iter().take(31) {
            worst = worst.max(dot(cols[j], leading).abs());
        }
    }
    assert!(worst <= 1e-13, "max |U_k^T U_c| = {worst:e}");
}

/// The one-sided Jacobi SVD of the previous release (0.2.2), kept verbatim as
/// the oracle for [`test_completion_matches_previous_release`].
/// Its completion of `U` costs `O(m^4)`, so it is only run on small shapes.
fn reference_compute<T: Field + Real + bytemuck::Zeroable>(
    a: MatRef<'_, T>,
) -> Option<(Mat<T>, Vec<T>, Mat<T>)> {
    let m = a.nrows();
    let n = a.ncols();

    if m == 0 || n == 0 {
        return None;
    }

    // Handle 1x1 case
    if m == 1 && n == 1 {
        let val = a[(0, 0)];
        let sigma = vec![Scalar::abs(val)];
        let mut u = Mat::zeros(1, 1);
        let mut vt = Mat::zeros(1, 1);
        u[(0, 0)] = if val >= T::zero() {
            T::one()
        } else {
            -T::one()
        };
        vt[(0, 0)] = T::one();
        return Some((u, sigma, vt));
    }

    // Copy A into working matrix B (we'll work on B to get V, then compute U)
    let mut b = Mat::zeros(m, n);
    for i in 0..m {
        for j in 0..n {
            b[(i, j)] = a[(i, j)];
        }
    }

    // Initialize V as identity
    let mut v = Mat::zeros(n, n);
    for i in 0..n {
        v[(i, i)] = T::one();
    }

    let eps = <T as Scalar>::epsilon();
    let tol = eps * T::from_f64(100.0).unwrap_or(T::one());

    // One-sided Jacobi: Apply Jacobi rotations to columns of B to diagonalize B^T*B
    // After convergence, columns of B are orthogonal (and are U*Σ), V accumulates rotations
    let k = m.min(n);

    for _sweep in 0..30 {
        let mut converged = true;

        // Sweep through all column pairs (i, j) with i < j
        // Only sweep up to min(m,n) for the first index since we can have at most k singular values
        for i in 0..n {
            for j in (i + 1)..n {
                // Compute B[:, i]^T * B[:, j] and norms
                let mut dot_ij = T::zero();
                let mut norm_i_sq = T::zero();
                let mut norm_j_sq = T::zero();

                for row in 0..m {
                    let bi = b[(row, i)];
                    let bj = b[(row, j)];
                    dot_ij = dot_ij + bi * bj;
                    norm_i_sq = norm_i_sq + bi * bi;
                    norm_j_sq = norm_j_sq + bj * bj;
                }

                // Check if rotation is needed
                let off_diag = Scalar::abs(dot_ij);
                let threshold = tol * Real::sqrt(norm_i_sq) * Real::sqrt(norm_j_sq);

                if off_diag > threshold {
                    converged = false;

                    // Compute Jacobi rotation to zero dot_ij
                    // We want to find angle θ such that after rotation, columns are orthogonal
                    // The 2x2 Gram matrix is [[norm_i_sq, dot_ij], [dot_ij, norm_j_sq]]
                    // Jacobi rotation diagonalizes this

                    let tau =
                        (norm_j_sq - norm_i_sq) / (T::from_f64(2.0).unwrap_or(T::one()) * dot_ij);
                    let t = if tau >= T::zero() {
                        T::one() / (tau + Real::sqrt(T::one() + tau * tau))
                    } else {
                        T::one() / (tau - Real::sqrt(T::one() + tau * tau))
                    };
                    let c = T::one() / Real::sqrt(T::one() + t * t);
                    let s = c * t;

                    // Apply rotation to columns i and j of B
                    for row in 0..m {
                        let bi = b[(row, i)];
                        let bj = b[(row, j)];
                        b[(row, i)] = c * bi - s * bj;
                        b[(row, j)] = s * bi + c * bj;
                    }

                    // Accumulate rotation into V
                    for row in 0..n {
                        let vi = v[(row, i)];
                        let vj = v[(row, j)];
                        v[(row, i)] = c * vi - s * vj;
                        v[(row, j)] = s * vi + c * vj;
                    }
                }
            }
        }

        if converged {
            break;
        }
    }

    // Now columns of B are orthogonal. The singular values are the norms of columns.
    // For an m×n matrix, we have min(m,n) singular values.
    // We need to find the columns with the largest norms.

    // Compute norms of all columns
    let mut col_norms: Vec<(usize, T)> = Vec::with_capacity(n);
    for j in 0..n {
        let mut norm_sq = T::zero();
        for row in 0..m {
            norm_sq = norm_sq + b[(row, j)] * b[(row, j)];
        }
        col_norms.push((j, Real::sqrt(norm_sq)));
    }

    // Sort columns by norm (descending)
    col_norms.sort_by(|a, b| {
        if b.1 > a.1 {
            core::cmp::Ordering::Greater
        } else if b.1 < a.1 {
            core::cmp::Ordering::Less
        } else {
            core::cmp::Ordering::Equal
        }
    });

    // Extract singular values and construct U
    let mut sigma = vec![T::zero(); k];
    let mut u = Mat::zeros(m, m);

    for i in 0..k {
        let (col_idx, norm) = col_norms[i];
        sigma[i] = norm;

        if norm > tol {
            for row in 0..m {
                u[(row, i)] = b[(row, col_idx)] / norm;
            }
        } else {
            // Zero singular value - will be filled by Gram-Schmidt below
        }
    }

    // Complete U to full orthogonal matrix using Gram-Schmidt
    for col in 0..k {
        if sigma[col] <= tol {
            // Find a standard basis vector orthogonal to existing columns
            for basis in 0..m {
                u[(basis, col)] = T::one();

                // Orthogonalize against previous columns
                for j in 0..col {
                    let mut dot = T::zero();
                    for row in 0..m {
                        dot = dot + u[(row, col)] * u[(row, j)];
                    }
                    for row in 0..m {
                        u[(row, col)] = u[(row, col)] - dot * u[(row, j)];
                    }
                }

                // Normalize
                let mut norm_sq = T::zero();
                for row in 0..m {
                    norm_sq = norm_sq + u[(row, col)] * u[(row, col)];
                }
                let norm = Real::sqrt(norm_sq);
                if norm > tol {
                    for row in 0..m {
                        u[(row, col)] = u[(row, col)] / norm;
                    }
                    break;
                }
                // Reset and try next basis vector
                for row in 0..m {
                    u[(row, col)] = T::zero();
                }
            }
        }
    }

    // Fill remaining columns of U using Gram-Schmidt
    for i in k..m {
        for basis in 0..m {
            u[(basis, i)] = T::one();

            // Orthogonalize against all previous columns
            for j in 0..i {
                let mut dot = T::zero();
                for row in 0..m {
                    dot = dot + u[(row, i)] * u[(row, j)];
                }
                for row in 0..m {
                    u[(row, i)] = u[(row, i)] - dot * u[(row, j)];
                }
            }

            // Normalize
            let mut norm_sq = T::zero();
            for row in 0..m {
                norm_sq = norm_sq + u[(row, i)] * u[(row, i)];
            }
            let norm = Real::sqrt(norm_sq);
            if norm > tol {
                for row in 0..m {
                    u[(row, i)] = u[(row, i)] / norm;
                }
                break;
            }
            // Reset and try next basis vector
            for row in 0..m {
                u[(row, i)] = T::zero();
            }
        }
    }

    // Construct V^T with columns reordered according to col_norms
    // V^T[i, :] should correspond to sigma[i]
    let mut vt = Mat::zeros(n, n);

    // First k rows of V^T correspond to the k singular values
    for i in 0..k {
        let (col_idx, _) = col_norms[i];
        for j in 0..n {
            vt[(i, j)] = v[(j, col_idx)];
        }
    }

    // Fill remaining rows of V^T with orthogonal complement
    for i in k..n {
        let (col_idx, _) = col_norms[i];
        for j in 0..n {
            vt[(i, j)] = v[(j, col_idx)];
        }
    }

    // Singular values are already sorted descending from col_norms extraction
    // No need for additional sorting

    Some((u, sigma, vt))
}
