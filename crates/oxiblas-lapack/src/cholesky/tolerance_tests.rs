//! Tests for the per-column (scale-invariant) tolerances of the Cholesky
//! family: the positive-definiteness test of every LLᵀ variant, the
//! singularity test of LDLᵀ (dense and packed) and the zero threshold of
//! `Ldlt::inertia`.

use super::*;
use num_complex::Complex64;
use oxiblas_matrix::Mat;

/// 2^-64 and 2^64 with their square roots 2^-32 and 2^32: an LLᵀ factor of
/// `A · s` is the factor of `A` times `sqrt(s)`, bit for bit, because
/// correctly rounded square roots commute with even powers of two.
const TWO_POW_M64: f64 = 1.0 / 18_446_744_073_709_551_616.0;
const TWO_POW_64: f64 = 18_446_744_073_709_551_616.0;
const TWO_POW_M32: f64 = 1.0 / 4_294_967_296.0;
const TWO_POW_32: f64 = 4_294_967_296.0;
const POW2_SCALES: [(f64, f64); 2] = [(TWO_POW_M64, TWO_POW_M32), (TWO_POW_64, TWO_POW_32)];

/// Deterministic xorshift64 generator (no external dependency).
struct XorShift64(u64);

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform sample in `[-1, 1)`.
    fn next_signed(&mut self) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * unit - 1.0
    }
}

/// Symmetric positive definite `B Bᵀ + n I` with `B` uniform in `[-1, 1)`.
fn random_spd(n: usize, seed: u64) -> Mat<f64> {
    let mut rng = XorShift64::new(seed);
    let mut b: Mat<f64> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            b[(i, j)] = rng.next_signed();
        }
    }
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += b[(i, k)] * b[(j, k)];
            }
            a[(i, j)] = sum;
        }
        a[(j, j)] += n as f64;
    }
    a
}

/// Symmetric indefinite, strongly diagonally dominant: off-diagonal entries
/// in `[-1, 1)` and a diagonal of `±(n + 1)` with alternating signs, so
/// LDLᵀ without pivoting meets only pivots of magnitude about `n`.
fn random_dominant_indefinite(n: usize, seed: u64) -> Mat<f64> {
    let mut rng = XorShift64::new(seed);
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in (j + 1)..n {
            let v = rng.next_signed();
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        a[(j, j)] = sign * (n as f64 + 1.0);
    }
    a
}

/// Hermitian positive definite `B Bᴴ + n I` with complex `B`.
fn random_hpd(n: usize, seed: u64) -> Mat<Complex64> {
    let mut rng = XorShift64::new(seed);
    let mut b: Mat<Complex64> = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            b[(i, j)] = Complex64::new(rng.next_signed(), rng.next_signed());
        }
    }
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            let mut sum = Complex64::new(0.0, 0.0);
            for k in 0..n {
                sum += b[(i, k)] * b[(j, k)].conj();
            }
            a[(i, j)] = sum;
        }
        a[(j, j)] = Complex64::new(a[(j, j)].re + n as f64, 0.0);
    }
    a
}

fn scaled(a: &Mat<f64>, s: f64) -> Mat<f64> {
    let n = a.nrows();
    let mut b = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            b[(i, j)] = a[(i, j)] * s;
        }
    }
    b
}

/// `D · A · D` for `D = diag(d)`.
fn sym_scaled(a: &Mat<f64>, d: &[f64]) -> Mat<f64> {
    let n = a.nrows();
    let mut b = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            b[(i, j)] = d[i] * a[(i, j)] * d[j];
        }
    }
    b
}

/// One power of two in `2^-span ..= 2^span` per row and column.
fn symmetric_factors(n: usize, seed: u64, span: i32) -> Vec<f64> {
    let mut rng = XorShift64::new(seed);
    let width = 2 * span as u64 + 1;
    (0..n)
        .map(|_| 2.0f64.powi((rng.next_u64() % width) as i32 - span))
        .collect()
}

fn row_major(a: &Mat<f64>) -> Vec<f64> {
    let n = a.nrows();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            v[i * n + j] = a[(i, j)];
        }
    }
    v
}

fn diag(entries: &[f64]) -> Mat<f64> {
    let n = entries.len();
    let mut a = Mat::zeros(n, n);
    for (j, &d) in entries.iter().enumerate() {
        a[(j, j)] = d;
    }
    a
}

/// FEM stiffness matrix with a penalty term: the 1-D Laplacian
/// `tridiag(-1, 2, -1)` of order `n` plus `1e20 · e_i e_iᵀ`.
fn fem_penalty(n: usize, i: usize) -> Mat<f64> {
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        a[(j, j)] = 2.0;
        if j + 1 < n {
            a[(j + 1, j)] = -1.0;
            a[(j, j + 1)] = -1.0;
        }
    }
    a[(i, i)] += 1.0e20;
    a
}

/// Symmetric positive definite matrices that factor exactly but whose
/// diagonal entries differ wildly in scale. The previous global references
/// (`max a_ii` for LLᵀ, `max |a_ij|` for LDLᵀ) rejected all of them; the
/// per-column references (like LAPACK) accept them.
fn badly_scaled_matrices() -> Vec<(&'static str, Mat<f64>)> {
    vec![
        ("diag(1e6, 1e-10)", diag(&[1.0e6, 1.0e-10])),
        ("diag(1e20, 1)", diag(&[1.0e20, 1.0])),
        ("FEM penalty", fem_penalty(6, 2)),
    ]
}

/// Matrices that are not positive definite, with the index at which every
/// LLᵀ variant must report it, independently of the scale.
fn non_pd_matrices() -> Vec<(&'static str, Mat<f64>, usize)> {
    let mut zero_row_col = random_spd(4, 9);
    for k in 0..4 {
        zero_row_col[(1, k)] = 0.0;
        zero_row_col[(k, 1)] = 0.0;
    }
    vec![
        ("indefinite", Mat::from_rows(&[&[1.0, 2.0], &[2.0, 1.0]]), 1),
        (
            "singular semidefinite",
            Mat::from_rows(&[&[1.0, 1.0], &[1.0, 1.0]]),
            1,
        ),
        ("zero", Mat::zeros(3, 3), 0),
        ("zero row and column", zero_row_col, 1),
        (
            "negative diagonal",
            Mat::from_rows(&[&[-1.0, 0.0], &[0.0, 1.0]]),
            0,
        ),
    ]
}

type CholVariant = (
    &'static str,
    fn(&Mat<f64>) -> Result<Cholesky<f64>, CholeskyError>,
);

fn llt_variants() -> Vec<CholVariant> {
    let base: [CholVariant; 7] = [
        ("compute", |a| Cholesky::compute(a.as_ref())),
        ("compute_with_block_size(1)", |a| {
            Cholesky::compute_with_block_size(a.as_ref(), 1)
        }),
        ("compute_with_block_size(8)", |a| {
            Cholesky::compute_with_block_size(a.as_ref(), 8)
        }),
        ("compute_with_block_size(32)", |a| {
            Cholesky::compute_with_block_size(a.as_ref(), 32)
        }),
        ("compute_blocked", |a| Cholesky::compute_blocked(a.as_ref())),
        ("compute_recursive", |a| {
            Cholesky::compute_recursive(a.as_ref())
        }),
        ("compute_auto", |a| Cholesky::compute_auto(a.as_ref())),
    ];
    #[cfg(feature = "parallel")]
    let par: [CholVariant; 2] = [
        ("compute_blocked_par", |a| {
            Cholesky::compute_blocked_par(a.as_ref())
        }),
        ("compute_blocked_par_with_block_size(8)", |a| {
            Cholesky::compute_blocked_par_with_block_size(a.as_ref(), 8)
        }),
    ];
    #[cfg(not(feature = "parallel"))]
    let par: [CholVariant; 0] = [];
    base.into_iter().chain(par).collect()
}

fn spd_matrices() -> Vec<(String, Mat<f64>)> {
    [1usize, 2, 5, 33, 70, 150]
        .iter()
        .enumerate()
        .map(|(idx, &n)| (format!("spd n={n}"), random_spd(n, 100 + idx as u64)))
        .collect()
}

/// Every LLᵀ test matrix: SPD, not positive definite and badly scaled.
fn llt_test_matrices() -> Vec<(String, Mat<f64>)> {
    let mut mats = spd_matrices();
    for (name, a, _) in non_pd_matrices() {
        mats.push((name.to_string(), a));
    }
    for (name, a) in badly_scaled_matrices() {
        mats.push((name.to_string(), a));
    }
    mats
}

/// Lower triangle of `other` equals that of `base` with row `i` multiplied
/// by `rows[i]`, bit for bit.
fn assert_lower_scaled(label: &str, base: &Mat<f64>, other: &Mat<f64>, rows: &[f64]) {
    let n = base.nrows();
    assert_eq!(other.nrows(), n, "{label}: sizes differ");
    for j in 0..n {
        for i in j..n {
            assert_eq!(
                other[(i, j)].to_bits(),
                (base[(i, j)] * rows[i]).to_bits(),
                "{label}: L({i}, {j}) is {} but expected {}",
                other[(i, j)],
                base[(i, j)] * rows[i]
            );
        }
    }
}

/// `other[idx] == base[idx] * factors[idx]`, bit for bit.
fn assert_slice_scaled(label: &str, base: &[f64], other: &[f64], factors: &[f64]) {
    assert_eq!(base.len(), other.len(), "{label}: lengths differ");
    assert_eq!(base.len(), factors.len(), "{label}: factor count differs");
    for (idx, ((&x, &y), &f)) in base.iter().zip(other).zip(factors).enumerate() {
        assert_eq!(
            y.to_bits(),
            (x * f).to_bits(),
            "{label}: entry {idx} is {y} but expected {}",
            x * f
        );
    }
}

/// Compares two outcomes: equal errors, or `check` on both factorizations.
fn compare_outcomes<F, E: core::fmt::Debug + PartialEq>(
    label: &str,
    base: &Result<F, E>,
    other: &Result<F, E>,
    check: impl FnOnce(&F, &F),
) {
    match (base, other) {
        (Ok(b), Ok(o)) => check(b, o),
        (Err(e1), Err(e2)) => assert_eq!(e1, e2, "{label}: errors differ"),
        (b, o) => panic!(
            "{label}: outcome depends on scale: A -> {:?}, scaled A -> {:?}",
            b.as_ref().map(|_| ()),
            o.as_ref().map(|_| ())
        ),
    }
}

/// A scaling of the input together with the factor of every row of the LLᵀ
/// factor: `(name, D, rows)` describes `A -> D A D` (`rows = D`); a global
/// scaling by `s` is `D = sqrt(s) I`.
fn llt_scalings(n: usize) -> Vec<(String, Vec<f64>)> {
    let mut scalings: Vec<(String, Vec<f64>)> = POW2_SCALES
        .iter()
        .map(|&(s, r)| (format!("scaled by {s:e}"), vec![r; n]))
        .collect();
    for seed in [1u64, 2] {
        scalings.push((
            format!("symmetric scaling #{seed}"),
            symmetric_factors(n, seed, 30),
        ));
    }
    scalings
}

#[test]
fn test_cholesky_tolerance_llt_scaling() {
    // Under A -> D A D the radicand of column j and a_jj both scale by d_j²,
    // and L -> D L: outcome, index and factor are reproduced exactly.
    for (variant, factor) in llt_variants() {
        for (name, a) in llt_test_matrices() {
            let base = factor(&a);
            for (how, d) in llt_scalings(a.nrows()) {
                let label = format!("Cholesky::{variant} on {name} {how}");
                compare_outcomes(&label, &base, &factor(&sym_scaled(&a, &d)), |b, o| {
                    assert_lower_scaled(&label, &b.l_factor(), &o.l_factor(), &d);
                });
            }
        }
    }
}

#[test]
fn test_cholesky_tolerance_hermitian_scaling() {
    let mut mats: Vec<(String, Mat<Complex64>)> = [1usize, 4, 20]
        .iter()
        .enumerate()
        .map(|(idx, &n)| (format!("hpd n={n}"), random_hpd(n, 200 + idx as u64)))
        .collect();
    let i2 = Complex64::new(0.0, 2.0);
    let one = Complex64::new(1.0, 0.0);
    let mut indefinite: Mat<Complex64> = Mat::zeros(2, 2);
    indefinite[(0, 0)] = one;
    indefinite[(1, 0)] = -i2;
    indefinite[(0, 1)] = i2;
    indefinite[(1, 1)] = one;
    mats.push(("indefinite".to_string(), indefinite));
    for (name, a) in badly_scaled_matrices() {
        let n = a.nrows();
        let mut c: Mat<Complex64> = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                c[(i, j)] = Complex64::new(a[(i, j)], 0.0);
            }
        }
        mats.push((name.to_string(), c));
    }

    for (name, a) in &mats {
        let n = a.nrows();
        let base = HermitianCholesky::compute(a.as_ref());
        for (how, d) in llt_scalings(n) {
            let mut b: Mat<Complex64> = Mat::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    b[(i, j)] = a[(i, j)] * (d[i] * d[j]);
                }
            }
            let label = format!("HermitianCholesky on {name} {how}");
            compare_outcomes(
                &label,
                &base,
                &HermitianCholesky::compute(b.as_ref()),
                |x, y| {
                    let (lx, ly) = (x.l_factor(), y.l_factor());
                    for j in 0..n {
                        for i in j..n {
                            let (re, im) = (lx[(i, j)].re * d[i], lx[(i, j)].im * d[i]);
                            assert_eq!(ly[(i, j)].re.to_bits(), re.to_bits(), "{label}");
                            assert_eq!(ly[(i, j)].im.to_bits(), im.to_bits(), "{label}");
                        }
                    }
                },
            );
        }
    }
}

#[test]
fn test_cholesky_tolerance_band_and_packed_scaling() {
    for (name, a) in llt_test_matrices() {
        let n = a.nrows();
        let kd = 2.min(n.saturating_sub(1));
        let ldab = kd + 1;
        let ab = dense_to_band_lower(&row_major(&a), n, kd);
        let ap_lower = dense_to_packed_lower(a.as_ref());
        let ap_upper = dense_to_packed_upper(a.as_ref());
        let band = BandCholesky::compute(n, kd, &ab);
        let lower = PackedCholesky::compute(&ap_lower, n, Uplo::Lower);
        let upper = PackedCholesky::compute(&ap_upper, n, Uplo::Upper);
        for (how, d) in llt_scalings(n) {
            let label = format!("{name} {how}");
            // Band storage: AB[r + j * ldab] = A[j + r, j]; L -> D L.
            let band_input: Vec<f64> = ab
                .iter()
                .enumerate()
                .map(|(idx, &x)| {
                    let (r, j) = (idx % ldab, idx / ldab);
                    if j + r < n { x * d[j + r] * d[j] } else { x }
                })
                .collect();
            let band_rows: Vec<f64> = (0..ab.len())
                .map(|idx| {
                    let (r, j) = (idx % ldab, idx / ldab);
                    if j + r < n { d[j + r] } else { 1.0 }
                })
                .collect();
            compare_outcomes(
                &format!("BandCholesky {label}"),
                &band,
                &BandCholesky::compute(n, kd, &band_input),
                |x, y| {
                    assert_slice_scaled(
                        &format!("BandCholesky {label}"),
                        x.ab(),
                        y.ab(),
                        &band_rows,
                    )
                },
            );
            // Packed lower: L -> D L, entry (i, j) scales by d_i. Packed
            // upper: A = Uᵀ U, U -> U D, entry (i, j) scales by d_j.
            let (mut lower_in, mut lower_f) = (Vec::new(), Vec::new());
            for j in 0..n {
                for i in j..n {
                    lower_in.push(ap_lower[packed_lower_index(n, i, j)] * d[i] * d[j]);
                    lower_f.push(d[i]);
                }
            }
            let (mut upper_in, mut upper_f) = (Vec::new(), Vec::new());
            for j in 0..n {
                for i in 0..=j {
                    upper_in.push(ap_upper[packed_upper_index(n, i, j)] * d[i] * d[j]);
                    upper_f.push(d[j]);
                }
            }
            compare_outcomes(
                &format!("PackedCholesky lower {label}"),
                &lower,
                &PackedCholesky::compute(&lower_in, n, Uplo::Lower),
                |x, y| assert_slice_scaled(&label, x.factor(), y.factor(), &lower_f),
            );
            compare_outcomes(
                &format!("PackedCholesky upper {label}"),
                &upper,
                &PackedCholesky::compute(&upper_in, n, Uplo::Upper),
                |x, y| assert_slice_scaled(&label, x.factor(), y.factor(), &upper_f),
            );
        }
    }
}

/// LDLᵀ fixtures: indefinite and SPD ones whose column maxima sit on the
/// diagonal, an exactly singular one, a zero row and column, and the badly
/// scaled SPD matrices.
fn ldlt_test_matrices() -> Vec<(String, Mat<f64>)> {
    let mut mats: Vec<(String, Mat<f64>)> = [1usize, 2, 5, 33, 70]
        .iter()
        .enumerate()
        .map(|(idx, &n)| {
            (
                format!("indefinite n={n}"),
                random_dominant_indefinite(n, 300 + idx as u64),
            )
        })
        .collect();
    mats.extend(spd_matrices().into_iter().take(4));
    mats.push((
        "singular".to_string(),
        Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]),
    ));
    let mut zero_row_col = random_dominant_indefinite(5, 310);
    for k in 0..5 {
        zero_row_col[(3, k)] = 0.0;
        zero_row_col[(k, 3)] = 0.0;
    }
    mats.push(("zero row and column".to_string(), zero_row_col));
    for (name, a) in badly_scaled_matrices() {
        mats.push((name.to_string(), a));
    }
    mats
}

/// Scalings for LDLᵀ: global by 2^±64, and `D A D` with `D` within
/// `2^-8 ..= 2^8`. The LDLᵀ reference of column j counts off-diagonal
/// entries, which scale by `d_i d_j` rather than `d_j²`; the outcome is
/// unchanged here because each fixture's pivots stay far from the threshold
/// even after a factor-2^16 spread (see the dedicated test below for a
/// matrix where `D A D` changes it).
fn ldlt_scalings(n: usize) -> Vec<(String, Vec<f64>)> {
    let mut scalings: Vec<(String, Vec<f64>)> = POW2_SCALES
        .iter()
        .map(|&(_, r)| (format!("scaled by {:e}", r * r), vec![r; n]))
        .collect();
    for seed in [5u64, 6] {
        scalings.push((
            format!("symmetric scaling #{seed}"),
            symmetric_factors(n, seed, 8),
        ));
    }
    scalings
}

#[test]
fn test_cholesky_tolerance_ldlt_scaling() {
    // Under A -> D A D: L -> D L D⁻¹ (entry (i, j) scales by d_i / d_j) and
    // D -> D² D (pivot j scales by d_j²), bit for bit.
    for (name, a) in ldlt_test_matrices() {
        let n = a.nrows();
        let base = Ldlt::compute(a.as_ref());
        let ap_lower = dense_to_packed_lower(a.as_ref());
        let ap_upper = dense_to_packed_upper(a.as_ref());
        let packed_lower = PackedLdlt::compute(&ap_lower, n, Uplo::Lower);
        let packed_upper = PackedLdlt::compute(&ap_upper, n, Uplo::Upper);
        for (how, d) in ldlt_scalings(n) {
            let label = format!("{name} {how}");
            let pivots: Vec<f64> = d.iter().map(|&x| x * x).collect();
            compare_outcomes(
                &format!("Ldlt {label}"),
                &base,
                &Ldlt::compute(sym_scaled(&a, &d).as_ref()),
                |x, y| {
                    let (lx, ly) = (x.l_factor(), y.l_factor());
                    for j in 0..n {
                        for i in 0..n {
                            let expected = lx[(i, j)] * d[i] / d[j];
                            assert_eq!(
                                ly[(i, j)].to_bits(),
                                expected.to_bits(),
                                "Ldlt L({i}, {j}) {label}"
                            );
                        }
                    }
                    assert_slice_scaled(&label, &x.d_diagonal(), &y.d_diagonal(), &pivots);
                    assert_eq!(x.inertia(), y.inertia(), "Ldlt inertia {label}");
                },
            );
            // Packed lower: L entry (i, j) scales by d_i / d_j. Packed upper:
            // A = Uᵀ D U, U -> D⁻¹ U D, entry (i, j) scales by d_j / d_i.
            let (mut lower_in, mut lower_f) = (Vec::new(), Vec::new());
            for j in 0..n {
                for i in j..n {
                    lower_in.push(ap_lower[packed_lower_index(n, i, j)] * d[i] * d[j]);
                    lower_f.push(d[i] / d[j]);
                }
            }
            let (mut upper_in, mut upper_f) = (Vec::new(), Vec::new());
            for j in 0..n {
                for i in 0..=j {
                    upper_in.push(ap_upper[packed_upper_index(n, i, j)] * d[i] * d[j]);
                    upper_f.push(d[j] / d[i]);
                }
            }
            for (uplo, input, factors, base_packed) in [
                (Uplo::Lower, &lower_in, &lower_f, &packed_lower),
                (Uplo::Upper, &upper_in, &upper_f, &packed_upper),
            ] {
                compare_outcomes(
                    &format!("PackedLdlt {uplo:?} {label}"),
                    base_packed,
                    &PackedLdlt::compute(input, n, uplo),
                    |x, y| {
                        assert_slice_scaled(&label, x.factor(), y.factor(), factors);
                        assert_slice_scaled(&label, x.diagonal(), y.diagonal(), &pivots);
                    },
                );
            }
        }
    }
}

#[test]
fn test_cholesky_tolerance_tiny_scale_accepted() {
    // Matrices scaled to 1e-30 factor perfectly well; the old absolute
    // threshold eps * n rejected their very first pivot.
    for &n in &[2usize, 5, 40, 100] {
        let a = scaled(&random_spd(n, 400 + n as u64), 1.0e-30);
        for (variant, factor) in llt_variants() {
            let chol =
                factor(&a).unwrap_or_else(|e| panic!("Cholesky::{variant} n={n} at 1e-30: {e}"));
            // det(A) underflows for large n; check L L^T = A instead.
            let l = chol.l_factor();
            for j in 0..n {
                for i in j..n {
                    let mut sum = 0.0;
                    for k in 0..=j {
                        sum += l[(i, k)] * l[(j, k)];
                    }
                    let err = (sum - a[(i, j)]).abs() / a[(j, j)];
                    assert!(
                        err < 1e-12,
                        "Cholesky::{variant} n={n}: ({i},{j}) err {err}"
                    );
                }
            }
        }
        let kd = 2.min(n - 1);
        let ab = dense_to_band_lower(&row_major(&a), n, kd);
        assert!(BandCholesky::compute(n, kd, &ab).is_ok(), "band n={n}");
        let ap = dense_to_packed_lower(a.as_ref());
        assert!(
            PackedCholesky::compute(&ap, n, Uplo::Lower).is_ok(),
            "packed n={n}"
        );
        let ap = dense_to_packed_upper(a.as_ref());
        assert!(
            PackedCholesky::compute(&ap, n, Uplo::Upper).is_ok(),
            "packed n={n}"
        );

        let h = random_hpd(n.min(20), 500 + n as u64);
        let mut h_tiny: Mat<Complex64> = Mat::zeros(h.nrows(), h.nrows());
        for j in 0..h.nrows() {
            for i in 0..h.nrows() {
                h_tiny[(i, j)] = h[(i, j)] * 1.0e-30;
            }
        }
        assert!(
            HermitianCholesky::compute(h_tiny.as_ref()).is_ok(),
            "hermitian n={n}"
        );

        let d = scaled(&random_dominant_indefinite(n, 600 + n as u64), 1.0e-30);
        let ldlt = Ldlt::compute(d.as_ref()).unwrap_or_else(|e| panic!("Ldlt n={n} at 1e-30: {e}"));
        let (pos, neg, zero) = ldlt.inertia();
        assert_eq!(
            (pos, neg, zero),
            (n.div_ceil(2), n / 2, 0),
            "Ldlt inertia n={n}"
        );
        let ap = dense_to_packed_lower(d.as_ref());
        assert!(
            PackedLdlt::compute(&ap, n, Uplo::Lower).is_ok(),
            "packed ldlt n={n}"
        );
    }
}

#[test]
fn test_cholesky_tolerance_non_pd_still_fails() {
    // Besides 2^-64 and 2^64, two scales that are not powers of two:
    // 9 * 2^-100 (about 7e-30) and 9 * 2^100 (about 1.1e31). They are exact
    // squares (of 3 * 2^-50 and 3 * 2^50), so the semidefinite fixture keeps an
    // exactly zero second radicand instead of a rounding residue.
    let nine_pow_m100 = 9.0 * TWO_POW_M64 * (1.0 / 68_719_476_736.0);
    let nine_pow_100 = 9.0 * TWO_POW_64 * 68_719_476_736.0;
    for s in [1.0, TWO_POW_M64, TWO_POW_64, nine_pow_m100, nine_pow_100] {
        for (name, a, index) in non_pd_matrices() {
            let b = scaled(&a, s);
            let n = b.nrows();
            for (variant, factor) in llt_variants() {
                assert_eq!(
                    factor(&b).map(|_| ()),
                    Err(CholeskyError::NotPositiveDefinite { index }),
                    "Cholesky::{variant} on {name} scaled by {s:e}"
                );
            }
            let kd = n - 1;
            assert_eq!(
                BandCholesky::compute(n, kd, &dense_to_band_lower(&row_major(&b), n, kd))
                    .map(|_| ()),
                Err(BandCholeskyError::NotPositiveDefinite { index }),
                "BandCholesky on {name} scaled by {s:e}"
            );
            for (uplo, ap) in [
                (Uplo::Lower, dense_to_packed_lower(b.as_ref())),
                (Uplo::Upper, dense_to_packed_upper(b.as_ref())),
            ] {
                assert_eq!(
                    PackedCholesky::compute(&ap, n, uplo).map(|_| ()),
                    Err(PackedCholeskyError::NotPositiveDefinite { index }),
                    "PackedCholesky {uplo:?} on {name} scaled by {s:e}"
                );
            }
            let mut h: Mat<Complex64> = Mat::zeros(n, n);
            for j in 0..n {
                for i in 0..n {
                    h[(i, j)] = Complex64::new(b[(i, j)], 0.0);
                }
            }
            assert_eq!(
                HermitianCholesky::compute(h.as_ref()).map(|_| ()),
                Err(HermitianCholeskyError::NotPositiveDefinite { index }),
                "HermitianCholesky on {name} scaled by {s:e}"
            );
        }

        let singular = scaled(&Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]), s);
        assert_eq!(
            Ldlt::compute(singular.as_ref()).map(|_| ()),
            Err(LdltError::Singular { index: 1 }),
            "Ldlt singular scaled by {s:e}"
        );
        assert_eq!(
            PackedLdlt::compute(&dense_to_packed_lower(singular.as_ref()), 2, Uplo::Lower)
                .map(|_| ()),
            Err(PackedLdltError::Singular { index: 1 }),
            "PackedLdlt singular scaled by {s:e}"
        );
        let zero: Mat<f64> = Mat::zeros(3, 3);
        assert_eq!(
            Ldlt::compute(zero.as_ref()).map(|_| ()),
            Err(LdltError::Singular { index: 0 })
        );
        // A zero row and column: its pivot is exactly zero.
        let mut zero_row_col = scaled(&random_dominant_indefinite(5, 310), s);
        for k in 0..5 {
            zero_row_col[(3, k)] = 0.0;
            zero_row_col[(k, 3)] = 0.0;
        }
        assert_eq!(
            Ldlt::compute(zero_row_col.as_ref()).map(|_| ()),
            Err(LdltError::Singular { index: 3 }),
            "Ldlt zero row and column scaled by {s:e}"
        );
        for (uplo, ap) in [
            (Uplo::Lower, dense_to_packed_lower(zero_row_col.as_ref())),
            (Uplo::Upper, dense_to_packed_upper(zero_row_col.as_ref())),
        ] {
            assert_eq!(
                PackedLdlt::compute(&ap, 5, uplo).map(|_| ()),
                Err(PackedLdltError::Singular { index: 3 }),
                "PackedLdlt {uplo:?} zero row and column scaled by {s:e}"
            );
        }
    }
}

#[test]
fn test_cholesky_tolerance_badly_scaled_accepted() {
    // diag(1e6, 1e-10), diag(1e20, 1) and a penalised FEM matrix are SPD and
    // factor exactly: every radicand is large next to its own a_jj. The
    // global reference rejected them (for diag(1e20, 1): 1 <= eps * 2 * 1e20),
    // as 0.2.2 and LAPACK do not.
    for (name, a) in badly_scaled_matrices() {
        let n = a.nrows();
        for (variant, factor) in llt_variants() {
            let l = factor(&a)
                .unwrap_or_else(|e| panic!("Cholesky::{variant} on {name}: {e}"))
                .l_factor();
            for j in 0..n {
                for i in j..n {
                    let mut sum = 0.0;
                    for k in 0..=j {
                        sum += l[(i, k)] * l[(j, k)];
                    }
                    let err = (sum - a[(i, j)]).abs() / (a[(i, i)] * a[(j, j)]).sqrt();
                    assert!(
                        err < 1e-15,
                        "Cholesky::{variant} on {name}: ({i}, {j}) err {err}"
                    );
                }
            }
        }
        let mut h: Mat<Complex64> = Mat::zeros(n, n);
        for j in 0..n {
            for i in 0..n {
                h[(i, j)] = Complex64::new(a[(i, j)], 0.0);
            }
        }
        assert!(
            HermitianCholesky::compute(h.as_ref()).is_ok(),
            "HermitianCholesky on {name}"
        );
        let kd = 1.min(n - 1);
        assert!(
            BandCholesky::compute(n, kd, &dense_to_band_lower(&row_major(&a), n, kd)).is_ok(),
            "BandCholesky on {name}"
        );
        for (uplo, ap) in [
            (Uplo::Lower, dense_to_packed_lower(a.as_ref())),
            (Uplo::Upper, dense_to_packed_upper(a.as_ref())),
        ] {
            assert!(
                PackedCholesky::compute(&ap, n, uplo).is_ok(),
                "PackedCholesky {uplo:?} on {name}"
            );
            assert!(
                PackedLdlt::compute(&ap, n, uplo).is_ok(),
                "PackedLdlt {uplo:?} on {name}"
            );
        }
        let ldlt = Ldlt::compute(a.as_ref()).unwrap_or_else(|e| panic!("Ldlt on {name}: {e}"));
        assert_eq!(ldlt.inertia(), (n, 0, 0), "Ldlt inertia on {name}");
        assert!(ldlt.is_positive_definite(), "Ldlt on {name}");
    }

    // A radicand that is tiny relative to its own a_jj is still rejected: in
    // [[1, 1], [1, 1 + eps]] the second radicand is the cancellation residue
    // eps, below eps * 2 * (1 + eps). LDLᵀ meets the same pivot.
    let near: Mat<f64> = Mat::from_rows(&[&[1.0, 1.0], &[1.0, 1.0 + f64::EPSILON]]);
    for (variant, factor) in llt_variants() {
        assert_eq!(
            factor(&near).map(|_| ()),
            Err(CholeskyError::NotPositiveDefinite { index: 1 }),
            "Cholesky::{variant} on [[1, 1], [1, 1 + eps]]"
        );
    }
    assert_eq!(
        Ldlt::compute(near.as_ref()).map(|_| ()),
        Err(LdltError::Singular { index: 1 })
    );
}

#[test]
fn test_cholesky_tolerance_ldlt_explicit_tolerance_and_inertia() {
    // An explicit tolerance is absolute and used exactly as given.
    let tiny = scaled(&random_spd(4, 7), 1.0e-30);
    assert_eq!(
        Ldlt::compute_with_tol(tiny.as_ref(), Some(1.0e-20)).map(|_| ()),
        Err(LdltError::Singular { index: 0 })
    );
    assert!(Ldlt::compute_with_tol(tiny.as_ref(), Some(0.0)).is_ok());
    let c = diag(&[1.0e6, 1.0e-10]);
    assert_eq!(
        Ldlt::compute_with_tol(c.as_ref(), Some(1.0e-9)).map(|_| ()),
        Err(LdltError::Singular { index: 1 })
    );

    // The inertia's zero threshold is per column too: with every pivot
    // accepted, d = (1e6, 1e-12) has two positive pivots, since 1e-12 is the
    // whole of its column (the global reference counted it as zero next to
    // 1e6), while the cancellation residue eps of [[1, 1], [1, 1 + eps]] is
    // zero next to its column's 1 + eps. Both hold at every scale.
    let wide = diag(&[1.0e6, 1.0e-12]);
    let near: Mat<f64> = Mat::from_rows(&[&[1.0, 1.0], &[1.0, 1.0 + f64::EPSILON]]);
    for s in [1.0, TWO_POW_M64, TWO_POW_64] {
        let ldlt = Ldlt::compute_with_tol(scaled(&wide, s).as_ref(), Some(0.0))
            .expect("every nonzero pivot is accepted");
        assert_eq!(ldlt.inertia(), (2, 0, 0), "diag inertia scaled by {s:e}");
        assert!(ldlt.is_positive_definite());
        let ldlt = Ldlt::compute_with_tol(scaled(&near, s).as_ref(), Some(0.0))
            .expect("every nonzero pivot is accepted");
        assert_eq!(ldlt.inertia(), (1, 0, 1), "near inertia scaled by {s:e}");
        assert!(!ldlt.is_positive_definite());
    }
}

#[test]
fn test_cholesky_tolerance_non_finite_entries_propagate() {
    // NaN and infinite diagonal entries do not enter the references, so they
    // keep propagating through the factor instead of rejecting a pivot. LAPACK
    // `dpotrf` reports a NaN radicand as `INFO = j` instead; this crate keeps
    // the IEEE-754 propagation of its LU factorizations.
    let nan: Mat<f64> = Mat::from_rows(&[&[4.0, f64::NAN], &[f64::NAN, 5.0]]);
    let inf: Mat<f64> = Mat::from_rows(&[&[4.0, 0.0], &[0.0, f64::INFINITY]]);
    for (variant, factor) in llt_variants() {
        let l = factor(&nan)
            .unwrap_or_else(|e| panic!("Cholesky::{variant} NaN: {e}"))
            .l_factor();
        assert_eq!(l[(0, 0)], 2.0, "Cholesky::{variant} NaN l00");
        assert!(l[(1, 1)].is_nan(), "Cholesky::{variant} NaN l11");
        let l = factor(&inf)
            .unwrap_or_else(|e| panic!("Cholesky::{variant} inf: {e}"))
            .l_factor();
        assert_eq!(l[(0, 0)], 2.0, "Cholesky::{variant} inf l00");
        assert!(l[(1, 1)].is_infinite(), "Cholesky::{variant} inf l11");
    }
    assert!(Ldlt::compute(inf.as_ref()).is_ok());
    assert!(Ldlt::compute(nan.as_ref()).is_ok());
}

#[test]
fn test_cholesky_tolerance_ldlt_reference_counts_off_diagonal_entries() {
    // LDLᵀ without pivoting on [[d, 1], [1, d]] with a tiny d needs the
    // multiplier 1/d. The reference of column 0 is max |a_i0| = 1, not
    // |a_00| = d, so the first pivot d = 1e-17 is rejected instead of being
    // accepted against eps * 2 * 1e-17 = 4.4e-33, at every global scale.
    let a: Mat<f64> = Mat::from_rows(&[&[1.0e-17, 1.0], &[1.0, 1.0e-17]]);
    for s in [1.0, TWO_POW_M64, TWO_POW_64] {
        let b = scaled(&a, s);
        assert_eq!(
            Ldlt::compute(b.as_ref()).map(|_| ()),
            Err(LdltError::Singular { index: 0 }),
            "Ldlt scaled by {s:e}"
        );
        for (uplo, ap) in [
            (Uplo::Lower, dense_to_packed_lower(b.as_ref())),
            (Uplo::Upper, dense_to_packed_upper(b.as_ref())),
        ] {
            assert_eq!(
                PackedLdlt::compute(&ap, 2, uplo).map(|_| ()),
                Err(PackedLdltError::Singular { index: 0 }),
                "PackedLdlt {uplo:?} scaled by {s:e}"
            );
        }
    }
    // Because off-diagonal entries count, LDLᵀ is not invariant under every
    // symmetric scaling D A D (unlike LLᵀ): D = diag(2^60, 1) turns the same
    // matrix into [[1.3e19, 1.2e18], [1.2e18, 1e-17]], whose first pivot is
    // the maximum of its column; the multiplier drops from 1e17 to 0.087 and
    // the factorization succeeds with pivots 1.3e19 and -1e17.
    let d = [2.0f64.powi(60), 1.0];
    let b = sym_scaled(&a, &d);
    let ldlt = Ldlt::compute(b.as_ref()).expect("D A D has no growth");
    assert_eq!(ldlt.inertia(), (1, 1, 0));
}
