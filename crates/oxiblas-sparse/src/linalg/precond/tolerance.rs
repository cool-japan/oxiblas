//! Scale-invariant pivot and zero tests, overflow-safe norms and exact
//! power-of-two scalings, shared by the preconditioners, the sparse
//! factorizations (`SparseLU`, `SparseCholesky`, and the incomplete LU and
//! Cholesky factorizations) and the iterative solvers.
//!
//! # Pivot and radicand thresholds
//!
//! Every test a preconditioner or factorization applies while it is built
//! either rejects only an exact zero, or compares a quantity with a reference
//! of the same units taken from the input. Both forms are invariant under
//! scaling the matrix by a power of two `s` (as long as nothing overflows or
//! underflows): building on `s·A` succeeds exactly when building on `A` does,
//! and gives the exactly scaled preconditioner or factors. An absolute
//! threshold such as `|a_ii| < 1e-14` has neither property: it rejects every
//! diagonal of a matrix scaled by `2^-64`, and misses numerically singular
//! pivots of a matrix scaled by `2^64`.
//!
//! The relative thresholds follow the crate's dense factorizations
//! (`oxiblas-lapack`), with the order `n` replaced by the number of terms that
//! actually form the tested quantity. A pivot or radicand is negligible when
//! it is at or below `eps · m · s`:
//!
//! - `s` is the reference from the original matrix: the largest finite
//!   magnitude of the original column (LU), or the original diagonal entry
//!   `a_jj` (the LLᵀ radicand). Only finite values contribute to a reference,
//!   so an infinite entry cannot raise a threshold to ∞, and a NaN never
//!   counts;
//! - `m` is the number of terms summed to form the pivot or radicand: the
//!   original entry plus every product actually subtracted from it during the
//!   elimination (`l_ik·u_kj` for LU, `l_jk²` for LLᵀ; products with an
//!   exactly zero factor, and multipliers dropped by a threshold rule, are not
//!   subtracted and do not count). It is the length of the sum whose rounding
//!   the threshold accounts for. For a dense matrix `m <= n` (and `m = j + 1`
//!   for column `j`), which recovers the dense rule; a global `n` would make
//!   the test far too strict for a large matrix with short rows: an identity
//!   of order `10^6` with one `2 x 2` block `[[1, 1], [1, 1 + 1e-12]]` would be
//!   rejected (`eps · 10^6 ≈ 2.2e-10 > 1e-12`), although the block's own
//!   radicand `1e-12` carries no rounding error to speak of.
//!
//! `m` depends only on which entries are (exactly) zero or dropped, and both
//! are scale-invariant decisions, so the thresholds stay exactly scale
//! invariant: `eps · m` is formed first and multiplied by `s` last.
//!
//! # Overflow-safe norms
//!
//! [`sum_of_squares`] (and the norm derived from it) first forms the plain
//! left-to-right sum `Σ v_i²` and keeps it when it is finite and at least
//! `min_positive / eps²` (`2^-918` for `f64`): then no square can have
//! overflowed, and a square that underflowed is far below half an ulp of the
//! sum. Otherwise it recomputes the sum with the values scaled by a power of
//! two `2^-e` close to their largest magnitude, and scales the result back.
//! Multiplying by a power of two is exact and commutes with every rounding, so
//! the rescaled sum gives the same bits as the plain one wherever the plain
//! one is representable, and stays accurate where the plain sum overflows
//! (values near `2^520`) or underflows (values near `2^-600`). The plain sum
//! of `s·v` is exactly `s²` times that of `v`, and `2^e` scales exactly with
//! the data, so the norm of `s·v` is exactly `|s|` times the norm of `v` for
//! every power of two `s` that keeps the values finite and normal. In the
//! common case the cost is that of the plain sum.
//!
//! # Power-of-four rescaling
//!
//! [`pow4_normalizer`] and [`gain_normalizer`] return powers of four that
//! bring a magnitude (or the ratio of two magnitudes) close to one. Powers of
//! four keep square roots exact (`sqrt(4^k · x) = 2^k · sqrt(x)`), which the
//! iterative solvers rely on when they work on a rescaled copy of a problem.

use oxiblas_core::scalar::{Field, Real, Scalar};

/// Returns `true` when `value` is finite (neither ±∞ nor NaN).
///
/// `value − value` is exactly zero for every finite IEEE-754 value and NaN for
/// ±∞ and NaN, so the test needs only field arithmetic.
#[inline]
pub(crate) fn is_finite<T: Scalar + Clone + Field>(value: &T) -> bool {
    value.clone() - value.clone() == T::zero()
}

/// Raises `slot` to `value` when `value` is finite and larger. NaN and ±∞
/// never count.
#[inline]
pub(crate) fn raise_finite<T: Scalar<Real = T> + Clone + Field + PartialOrd>(
    slot: &mut T,
    value: T,
) {
    if value > *slot && is_finite(&value) {
        *slot = value;
    }
}

/// Returns the largest finite `|v|` over `values`, or zero if there is none.
pub(crate) fn finite_max_abs<T: Scalar<Real = T> + Clone + Field + PartialOrd>(
    values: impl IntoIterator<Item = T>,
) -> T {
    let mut max = T::zero();
    for value in values {
        raise_finite(&mut max, Scalar::abs(value));
    }
    max
}

/// [`finite_max_abs`] of a slice, computed with a plain `max` reduction
/// (which skips NaN) and recomputed with the finite-only scan only when that
/// reduction is infinite.
pub(crate) fn finite_max_abs_slice<T: Scalar<Real = T> + Clone + Field + Real>(values: &[T]) -> T {
    let mut max = T::zero();
    for value in values {
        max = num_traits::Float::max(max, Scalar::abs(*value));
    }
    if is_finite(&max) {
        max
    } else {
        finite_max_abs(values.iter().copied())
    }
}

/// Returns the pivot threshold `eps · terms · scale`, `terms` being the
/// number of terms summed to form the pivot (see the module documentation).
///
/// `eps · terms` is formed first: it is far below one for any sum that fits
/// in memory, so the product cannot overflow even for a `scale` close to the
/// largest finite value. For a tiny `scale` the threshold may round to zero,
/// which leaves the exact-zero test as the floor.
#[inline]
pub(crate) fn pivot_tolerance<T: Scalar<Real = T> + Clone + Field + PartialOrd>(
    terms: usize,
    scale: T,
) -> T {
    let terms_real = T::from_usize(terms).unwrap_or(T::one());
    (<T as Scalar>::epsilon() * terms_real) * scale
}

/// Returns the per-column LU pivot references `s_j = max_i |a_ij|` of an
/// `n x n` matrix whose stored entries are given as `(column, value)` pairs
/// (in any order). Only finite values raise a column's reference, so a column
/// without finite nonzero entries gets the reference zero, which leaves the
/// exact-zero test as the floor. The threshold of a pivot of column `j` formed
/// from `m` terms is `pivot_tolerance(m, s_j)`.
///
/// # Panics
///
/// Panics if a column index is `>= n` (not possible for the entries of a
/// valid square CSR or CSC matrix of order `n`).
pub(crate) fn lu_column_scales<T: Scalar<Real = T> + Clone + Field + PartialOrd>(
    n: usize,
    entries: impl IntoIterator<Item = (usize, T)>,
) -> Vec<T> {
    let mut scales = vec![T::zero(); n];
    for (col, value) in entries {
        raise_finite(&mut scales[col], Scalar::abs(value));
    }
    scales
}

/// Returns the LLᵀ positive-definiteness threshold `eps · terms · max(0, a_jj)`
/// of a radicand `a_jj − Σ l_jk²` formed from `terms` terms (the diagonal
/// entry plus the squares actually subtracted), `a_jj` being the original
/// diagonal entry; a NaN, ±∞ or negative `a_jj` counts as zero.
///
/// The radicand must exceed this threshold. Radicand and `a_jj` scale alike
/// under `A → s·A` (and under the symmetric scaling `A → D·A·D`), and `terms`
/// does not change, so the outcome does not change.
#[inline]
pub(crate) fn llt_radicand_tolerance<T: Scalar<Real = T> + Clone + Field + PartialOrd>(
    terms: usize,
    a_jj: T,
) -> T {
    let mut scale = T::zero();
    raise_finite(&mut scale, a_jj);
    pivot_tolerance(terms, scale)
}

// =============================================================================
// Exact powers of two
// =============================================================================

/// Number of entries of [`PowerTable`]: enough for every IEEE-754 binary
/// format up to binary256.
const POWER_TABLE_LEN: usize = 20;

/// The exact powers `2^(2^k)`, `k = 0, 1, ...`, as long as they are finite.
struct PowerTable<T> {
    powers: [T; POWER_TABLE_LEN],
    len: usize,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> PowerTable<T> {
    fn new() -> Self {
        let mut powers = [T::one(); POWER_TABLE_LEN];
        let mut len = 0;
        let mut power = T::one() + T::one();
        while len < POWER_TABLE_LEN && is_finite(&power) {
            powers[len] = power;
            len += 1;
            power = power * power;
        }
        Self { powers, len }
    }

    /// Exponent `2^k` of entry `k`.
    fn exponent(k: usize) -> i32 {
        1i32 << k
    }
}

/// Returns `floor(log2(x))` for a finite `x > 0` (subnormal values included),
/// computed exactly by dividing or multiplying by powers of two; returns zero
/// for zero, negative and non-finite `x`.
pub(crate) fn binary_exponent<T: Scalar<Real = T> + Clone + Field + Real>(x: T) -> i32 {
    let one = T::one();
    let two = one + one;
    if !(x > T::zero()) || !is_finite(&x) {
        return 0;
    }
    let table = PowerTable::<T>::new();
    if table.len == 0 {
        return 0;
    }
    let top = table.len - 1;
    let mut y = x;
    if y >= two {
        let mut e = 0i32;
        // Every finite value is below the square of the largest finite power,
        // so the leading loop runs at most once; the descending scan then
        // leaves y in [1, 2).
        while y >= table.powers[top] {
            y = y / table.powers[top];
            e += PowerTable::<T>::exponent(top);
        }
        for k in (0..top).rev() {
            if y >= table.powers[k] {
                y = y / table.powers[k];
                e += PowerTable::<T>::exponent(k);
            }
        }
        e
    } else if y < one {
        // Largest m with x · 2^m < 1; then floor(log2(x)) = −(m + 1).
        let mut m = 0i32;
        while y * table.powers[top] < one {
            y = y * table.powers[top];
            m += PowerTable::<T>::exponent(top);
        }
        for k in (0..top).rev() {
            let candidate = y * table.powers[k];
            if candidate < one {
                y = candidate;
                m += PowerTable::<T>::exponent(k);
            }
        }
        -(m + 1)
    } else {
        0
    }
}

/// Range of exponents `e` for which both `2^e` and `2^-e` are normal
/// numbers: `[−emax, emax]` with `emax = min(e_max, −e_min)` (`1022` for
/// `f64`, `126` for `f32`).
fn normal_exponent_limit<T: Scalar<Real = T> + Clone + Field + Real>() -> i32 {
    let e_max = binary_exponent(<T as Scalar>::max_value());
    let e_min = binary_exponent(<T as Scalar>::min_positive());
    e_max.min(-e_min)
}

/// Returns `2^e` exactly, `e` clamped to the range where both `2^e` and its
/// reciprocal are normal numbers.
pub(crate) fn exact_pow2<T: Scalar<Real = T> + Clone + Field + Real>(e: i32) -> T {
    let limit = normal_exponent_limit::<T>();
    let e = e.clamp(-limit, limit);
    let table = PowerTable::<T>::new();
    let mut magnitude = e.unsigned_abs();
    let mut power = T::one();
    for k in 0..table.len {
        if magnitude == 0 {
            break;
        }
        if magnitude & 1 == 1 {
            power = power * table.powers[k];
        }
        magnitude >>= 1;
    }
    if e < 0 { T::one() / power } else { power }
}

/// Exponents within this distance of zero are left alone by the
/// power-of-four normalizers: `emax / 16` (63 for `f64`, 7 for `f32`), so a
/// value inside the band stays far enough from the overflow and underflow
/// thresholds that products of a few such values (`pᵀAp`, `tᵀt`, ...) are
/// safe.
fn dead_band<T: Scalar<Real = T> + Clone + Field + Real>() -> i32 {
    binary_exponent(<T as Scalar>::max_value()) / 16
}

/// Returns the power of four `f` with `f · m ∈ [1, 4)` when `m` is a finite
/// positive magnitude outside the band `2^±emax/16` around one (see
/// [`dead_band`]); returns one for a magnitude inside the band, zero, or not
/// finite. `f` is clamped so that `f` and `1/f` are normal numbers.
///
/// The factor scales exactly with `m`: for `s = 4^k`, `pow4_normalizer(s·m)`
/// is `pow4_normalizer(m) / s` whenever both magnitudes lie outside the band.
pub(crate) fn pow4_normalizer<T: Scalar<Real = T> + Clone + Field + Real>(m: T) -> T {
    if !(m > T::zero()) || !is_finite(&m) {
        return T::one();
    }
    let e = binary_exponent(m);
    if e.abs() <= dead_band::<T>() {
        return T::one();
    }
    exact_pow2(-2 * e.div_euclid(2))
}

/// Returns the power of four `g` that brings the ratio `output / input` of
/// two finite positive magnitudes close to one (`g · output / input ∈
/// (1/2, 4)`), or one when the ratio is within `2^±emax/16` of one (see
/// [`dead_band`]) or either magnitude is zero or not finite. `g` is clamped so
/// that `g` and `1/g` are normal numbers.
pub(crate) fn gain_normalizer<T: Scalar<Real = T> + Clone + Field + Real>(
    input: T,
    output: T,
) -> T {
    let usable = |v: &T| *v > T::zero() && is_finite(v);
    if !usable(&input) || !usable(&output) {
        return T::one();
    }
    let d = binary_exponent(output) - binary_exponent(input);
    if d.abs() <= dead_band::<T>() {
        return T::one();
    }
    exact_pow2(-2 * d.div_euclid(2))
}

// =============================================================================
// Overflow-safe sums of squares
// =============================================================================

/// A sum of squares `Σ v_i²` held as `scale² · sum`, `scale` a power of two
/// close to `max |v_i|` (see the module documentation).
#[derive(Debug, Clone, Copy)]
pub(crate) struct SumOfSquares<T> {
    scale: T,
    sum: T,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> SumOfSquares<T> {
    /// The 2-norm `sqrt(Σ v_i²)`: `+∞` if a value is infinite, NaN if a value
    /// is NaN.
    #[inline]
    pub(crate) fn norm(&self) -> T {
        self.scale * Real::sqrt(self.sum)
    }

    /// The sum of squares `Σ v_i²` itself (which overflows to `+∞` when it is
    /// not representable).
    #[inline]
    pub(crate) fn value(&self) -> T {
        self.scale * (self.scale * self.sum)
    }

    /// `true` when every value is zero.
    #[inline]
    pub(crate) fn is_zero(&self) -> bool {
        self.sum == T::zero()
    }

    /// The sum of squares of both sequences together, for two parts computed
    /// with a common scale by [`sum_of_squares_split`] (or with at least one
    /// part zero): the scaled sums are added, `self` first.
    pub(crate) fn plus(&self, other: &Self) -> Self {
        if self.sum == T::zero() {
            return *other;
        }
        if other.sum == T::zero() {
            return *self;
        }
        Self {
            scale: self.scale,
            sum: self.sum + other.sum,
        }
    }
}

/// Magnitude statistics of a sequence: the largest finite magnitude, and
/// whether a NaN or an infinite value was seen.
#[derive(Clone, Copy)]
struct MagnitudeScan<T> {
    max: T,
    nan: bool,
    infinite: bool,
}

impl<T: Scalar<Real = T> + Clone + Field + Real> MagnitudeScan<T> {
    fn new() -> Self {
        Self {
            max: T::zero(),
            nan: false,
            infinite: false,
        }
    }

    #[inline]
    fn add(&mut self, value: T) {
        let magnitude = Scalar::abs(value);
        if magnitude > self.max {
            if is_finite(&magnitude) {
                self.max = magnitude;
            } else {
                self.infinite = true;
            }
        } else if !(magnitude <= self.max) {
            // Neither above nor at or below the maximum: NaN.
            self.nan = true;
        }
    }

    /// The power of two `(2^e, 2^-e)` used to scale the values, or `None`
    /// when the result is fixed by the non-finite or zero values alone.
    fn scaling(&self) -> Result<(T, T), SumOfSquares<T>> {
        if self.nan {
            return Err(SumOfSquares {
                scale: T::one(),
                sum: T::zero() / T::zero(),
            });
        }
        if self.infinite {
            return Err(SumOfSquares {
                scale: T::one(),
                sum: T::one() / T::zero(),
            });
        }
        if self.max == T::zero() {
            return Err(SumOfSquares {
                scale: T::one(),
                sum: T::zero(),
            });
        }
        // exact_pow2 clamps e so that 2^e and 2^-e are normal: values below
        // the smallest normal number are scaled up exactly, and the largest
        // magnitude becomes at most 4 after scaling.
        let e = binary_exponent(self.max);
        Ok((exact_pow2(e), exact_pow2(-e)))
    }
}

/// Smallest plain sum of squares accepted without rescaling:
/// `min_positive / eps²` (`2^-918` for `f64`). Above it, a square that
/// underflowed is far below half an ulp of the sum, so it cannot have changed
/// the result; below it (and for a sum that is not finite) the sum is
/// recomputed with power-of-two scaling.
fn plain_sum_floor<T: Scalar<Real = T> + Clone + Field + Real>() -> T {
    let eps = <T as Scalar>::epsilon();
    <T as Scalar>::min_positive() / (eps * eps)
}

/// The plain sum of squares when it can be trusted (finite and at least
/// [`plain_sum_floor`]), held with scale one.
#[inline]
fn trusted_plain_sum<T: Scalar<Real = T> + Clone + Field + Real>(
    sum: T,
) -> Option<SumOfSquares<T>> {
    if is_finite(&sum) && sum >= plain_sum_floor::<T>() {
        Some(SumOfSquares {
            scale: T::one(),
            sum,
        })
    } else {
        None
    }
}

/// Two-pass sum of squares with power-of-two scaling (see the module
/// documentation).
fn scaled_sum_of_squares<T, I>(values: I) -> SumOfSquares<T>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    I: Iterator<Item = T> + Clone,
{
    let mut scan = MagnitudeScan::new();
    for value in values.clone() {
        scan.add(value);
    }
    let (scale, inverse) = match scan.scaling() {
        Ok(factors) => factors,
        Err(fixed) => return fixed,
    };
    let mut sum = T::zero();
    for value in values {
        let scaled = value * inverse;
        sum = sum + scaled * scaled;
    }
    SumOfSquares { scale, sum }
}

/// Overflow-safe sum of squares of `values`.
///
/// One pass computes the plain left-to-right sum of squares, which is used as
/// is when it is finite and not tiny (the common case: no overhead, and the
/// result is bit for bit the plain computation). Otherwise the sum is
/// recomputed in two passes with power-of-two scaling, which is accurate near
/// the overflow and underflow thresholds and gives the same bits as the plain
/// sum (exactly scaled) whenever the latter is representable; see the module
/// documentation.
pub(crate) fn sum_of_squares<T, I>(values: I) -> SumOfSquares<T>
where
    T: Scalar<Real = T> + Clone + Field + Real,
    I: Iterator<Item = T> + Clone,
{
    let mut sum = T::zero();
    for value in values.clone() {
        sum = sum + value * value;
    }
    trusted_plain_sum(sum).unwrap_or_else(|| scaled_sum_of_squares(values))
}

/// Overflow-safe sums of squares of two sequences with a common scale (taken
/// from the largest finite magnitude of both), so that
/// `first.plus(&second)` is the sum of squares of the concatenation, bit for
/// bit `Σ first² + Σ second²` whenever the plain sums are representable.
pub(crate) fn sum_of_squares_split<T, I, J>(
    first: I,
    second: J,
) -> (SumOfSquares<T>, SumOfSquares<T>)
where
    T: Scalar<Real = T> + Clone + Field + Real,
    I: Iterator<Item = T> + Clone,
    J: Iterator<Item = T> + Clone,
{
    let mut scan = MagnitudeScan::new();
    for value in first.clone() {
        scan.add(value);
    }
    for value in second.clone() {
        scan.add(value);
    }
    match scan.scaling() {
        // Zero or non-finite values: each part on its own (both scales are
        // then one, or a part is zero).
        Err(_) => (sum_of_squares(first), sum_of_squares(second)),
        Ok((scale, inverse)) => {
            let scaled_sum = |values: &mut dyn Iterator<Item = T>| {
                let mut sum = T::zero();
                for value in values {
                    let scaled = value * inverse;
                    sum = sum + scaled * scaled;
                }
                sum
            };
            let mut first = first;
            let mut second = second;
            (
                SumOfSquares {
                    scale,
                    sum: scaled_sum(&mut first),
                },
                SumOfSquares {
                    scale,
                    sum: scaled_sum(&mut second),
                },
            )
        }
    }
}

/// Overflow-safe 2-norm of `values` (see [`sum_of_squares`]).
#[inline]
pub(crate) fn norm2<T, I>(values: I) -> T
where
    T: Scalar<Real = T> + Clone + Field + Real,
    I: Iterator<Item = T> + Clone,
{
    sum_of_squares(values).norm()
}

/// Returns `(u·v, Σ u_i², Σ v_i²)` from one fused pass: the inner product
/// accumulated in index order exactly like a plain dot product, and the sums
/// of squares as in [`sum_of_squares`] (a sum that cannot be trusted is
/// recomputed with scaling, in two more passes over its vector).
///
/// # Panics
///
/// Panics if `u` and `v` have different lengths.
pub(crate) fn dot_and_sums_of_squares<T>(u: &[T], v: &[T]) -> (T, SumOfSquares<T>, SumOfSquares<T>)
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    assert_eq!(u.len(), v.len());
    let mut uv = T::zero();
    let mut uu = T::zero();
    let mut vv = T::zero();
    for (ui, vi) in u.iter().zip(v) {
        uv = uv + *ui * *vi;
        uu = uu + *ui * *ui;
        vv = vv + *vi * *vi;
    }
    let uu = trusted_plain_sum(uu).unwrap_or_else(|| scaled_sum_of_squares(u.iter().copied()));
    let vv = trusted_plain_sum(vv).unwrap_or_else(|| scaled_sum_of_squares(v.iter().copied()));
    (uv, uu, vv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_finite() {
        assert!(is_finite(&0.0_f64));
        assert!(is_finite(&-0.0_f64));
        assert!(is_finite(&f64::MAX));
        assert!(is_finite(&f64::MIN_POSITIVE));
        assert!(is_finite(&5e-324_f64));
        assert!(!is_finite(&f64::INFINITY));
        assert!(!is_finite(&f64::NEG_INFINITY));
        assert!(!is_finite(&f64::NAN));
        assert!(is_finite(&1.0_f32));
        assert!(!is_finite(&f32::NAN));
    }

    #[test]
    fn test_finite_max_abs_skips_non_finite() {
        assert_eq!(finite_max_abs([1.0, -3.0, 2.0]), 3.0);
        assert_eq!(finite_max_abs([1.0, f64::INFINITY, -2.0, f64::NAN]), 2.0);
        assert_eq!(finite_max_abs::<f64>([]), 0.0);
        assert_eq!(finite_max_abs([f64::NAN]), 0.0);
    }

    #[test]
    fn test_llt_radicand_tolerance() {
        let eps = f64::EPSILON;
        assert_eq!(llt_radicand_tolerance(4, 2.0), eps * 4.0 * 2.0);
        assert_eq!(llt_radicand_tolerance(4, -2.0), 0.0);
        assert_eq!(llt_radicand_tolerance(4, f64::NAN), 0.0);
        assert_eq!(llt_radicand_tolerance(4, f64::INFINITY), 0.0);
    }

    #[test]
    fn test_lu_column_scales() {
        // Columns: 0 -> {3, -5}, 1 -> {inf, 2}, 2 -> {NaN}, 3 -> {}.
        let entries = [
            (0, 3.0),
            (1, f64::INFINITY),
            (0, -5.0),
            (2, f64::NAN),
            (1, 2.0),
        ];
        let scales = lu_column_scales(4, entries);
        assert_eq!(scales, vec![5.0, 2.0, 0.0, 0.0]);
        let eps = f64::EPSILON;
        assert_eq!(pivot_tolerance(3, scales[0]), eps * 3.0 * 5.0);
    }

    #[test]
    fn test_pivot_tolerance_scales_exactly() {
        let s = 2.0_f64.powi(64);
        for scale in [1.0, 3.5, 1e-3, 7e12] {
            for terms in [1, 2, 10, 1_000_000] {
                let base = pivot_tolerance(terms, scale);
                assert_eq!(
                    pivot_tolerance(terms, scale * s).to_bits(),
                    (base * s).to_bits()
                );
                assert_eq!(
                    pivot_tolerance(terms, scale / s).to_bits(),
                    (base / s).to_bits()
                );
            }
        }
    }

    #[test]
    fn test_binary_exponent_exact() {
        for e in -1074..=1023 {
            let x = 2.0_f64.powi(e.max(-1022))
                * if e < -1022 {
                    2.0_f64.powi(e + 1022)
                } else {
                    1.0
                };
            assert_eq!(binary_exponent(x), e, "2^{e}");
            if e > -1074 {
                // Just below the power of two.
                let below = f64::from_bits(x.to_bits() - 1);
                assert_eq!(binary_exponent(below), e - 1, "below 2^{e}");
            }
        }
        assert_eq!(binary_exponent(f64::MAX), 1023);
        assert_eq!(binary_exponent(3.0_f64), 1);
        assert_eq!(binary_exponent(0.75_f64), -1);
        assert_eq!(binary_exponent(0.0_f64), 0);
        assert_eq!(binary_exponent(f64::NAN), 0);
        assert_eq!(binary_exponent(f32::MAX), 127);
        assert_eq!(binary_exponent(f32::MIN_POSITIVE), -126);
    }

    #[test]
    fn test_exact_pow2_clamped_to_normal_range() {
        assert_eq!(exact_pow2::<f64>(0), 1.0);
        assert_eq!(exact_pow2::<f64>(520), 2.0_f64.powi(520));
        assert_eq!(exact_pow2::<f64>(-600), 2.0_f64.powi(-600));
        assert_eq!(exact_pow2::<f64>(1022), 2.0_f64.powi(1022));
        assert_eq!(exact_pow2::<f64>(5000), 2.0_f64.powi(1022));
        assert_eq!(exact_pow2::<f64>(-5000), 2.0_f64.powi(-1022));
        assert_eq!(exact_pow2::<f32>(200), 2.0_f32.powi(126));
    }

    #[test]
    fn test_pow4_normalizer() {
        // Inside the dead band: no rescaling.
        assert_eq!(pow4_normalizer(5.5_f64), 1.0);
        assert_eq!(pow4_normalizer(2.0_f64.powi(63)), 1.0);
        assert_eq!(pow4_normalizer(2.0_f64.powi(-63)), 1.0);
        assert_eq!(pow4_normalizer(0.0_f64), 1.0);
        assert_eq!(pow4_normalizer(f64::INFINITY), 1.0);
        // Outside: f·m in [1, 4), f a power of four that scales with m.
        for m in [
            5.5 * 2.0_f64.powi(520),
            1.75 * 2.0_f64.powi(-520),
            3.0e200,
            7.0e-250,
        ] {
            let f = pow4_normalizer(m);
            let e = binary_exponent(f);
            assert_eq!(e % 2, 0, "{m:e}: {f:e} is not a power of four");
            assert_eq!(f, 2.0_f64.powi(e));
            assert!((1.0..4.0).contains(&(f * m)), "{m:e}: f·m = {}", f * m);
            let s = 2.0_f64.powi(64);
            assert_eq!(pow4_normalizer(m * s), f / s);
        }
    }

    #[test]
    fn test_gain_normalizer() {
        assert_eq!(gain_normalizer(1.0_f64, 0.25), 1.0);
        assert_eq!(gain_normalizer(1.0_f64, 0.0), 1.0);
        let g = gain_normalizer(1.0_f64, 2.0_f64.powi(-520) * 0.3);
        assert!((0.5..4.0).contains(&(g * 2.0_f64.powi(-520) * 0.3)));
        assert_eq!(binary_exponent(g) % 2, 0);
    }

    fn plain_norm(v: &[f64]) -> f64 {
        v.iter().fold(0.0, |sum, x| sum + x * x).sqrt()
    }

    #[test]
    fn test_norm2_bitwise_equal_to_plain_norm_in_range() {
        let vectors: [&[f64]; 5] = [
            &[1.0, 2.0, 3.0],
            &[0.1, -0.7, 1e-3, 12.5, -3.25],
            &[1e100, 3e99, -2e100],
            &[1e-100, 3e-101],
            &[0.0, 0.0, 4.0],
        ];
        for v in vectors {
            let n = norm2(v.iter().copied());
            assert_eq!(n.to_bits(), plain_norm(v).to_bits(), "{v:?}");
            let squares = sum_of_squares(v.iter().copied()).value();
            let plain_squares = v.iter().fold(0.0, |sum, x| sum + x * x);
            assert_eq!(squares.to_bits(), plain_squares.to_bits(), "{v:?}");
        }
    }

    #[test]
    fn test_norm2_survives_overflow_and_underflow() {
        let base = [1.0, -2.0, 2.0];
        for s in [
            2.0_f64.powi(520),
            2.0_f64.powi(-600),
            2.0_f64.powi(1000),
            2.0_f64.powi(-1060),
        ] {
            let v: Vec<f64> = base.iter().map(|x| x * s).collect();
            let n = norm2(v.iter().copied());
            assert_eq!(n, 3.0 * s, "scale {s:e}");
        }
        assert_eq!(
            norm2([f64::MAX, f64::MAX].iter().copied()),
            f64::MAX * 2.0_f64.sqrt()
        );
        assert!(norm2([1.0, f64::INFINITY].iter().copied()).is_infinite());
        assert!(norm2([1.0, f64::NAN, f64::INFINITY].iter().copied()).is_nan());
        assert_eq!(norm2(std::iter::empty::<f64>()), 0.0);
        assert!(sum_of_squares([0.0, 0.0].iter().copied()).is_zero());
    }

    #[test]
    fn test_norm2_scales_exactly() {
        let v = [0.3, -1.7, 2.9e3, 1e-5, 42.0];
        let base = norm2(v.iter().copied());
        for s in [
            2.0_f64.powi(64),
            2.0_f64.powi(-64),
            2.0_f64.powi(520),
            2.0_f64.powi(-520),
        ] {
            let scaled: Vec<f64> = v.iter().map(|x| x * s).collect();
            assert_eq!(
                norm2(scaled.iter().copied()).to_bits(),
                (base * s).to_bits()
            );
        }
    }

    #[test]
    fn test_sum_of_squares_split_matches_plain_sums() {
        let above = [3.0_f64, -1.5];
        let below = [0.25_f64, 2.0, -7.0];
        let (a, b) = sum_of_squares_split(above.iter().copied(), below.iter().copied());
        let plain_above = above.iter().fold(0.0_f64, |s, x| s + x * x);
        let plain_below = below.iter().fold(0.0_f64, |s, x| s + x * x);
        assert_eq!(b.norm().to_bits(), plain_below.sqrt().to_bits());
        assert_eq!(
            a.plus(&b).norm().to_bits(),
            (plain_above + plain_below).sqrt().to_bits()
        );
        // A zero part, and parts far beyond the overflow threshold.
        let (z, b) = sum_of_squares_split([0.0_f64].iter().copied(), below.iter().copied());
        assert_eq!(z.plus(&b).norm().to_bits(), plain_below.sqrt().to_bits());
        let s = 2.0_f64.powi(700);
        let big_above: Vec<f64> = above.iter().map(|x| x * s).collect();
        let big_below: Vec<f64> = below.iter().map(|x| x * s).collect();
        let (a, b) = sum_of_squares_split(big_above.iter().copied(), big_below.iter().copied());
        assert_eq!(a.plus(&b).norm(), (plain_above + plain_below).sqrt() * s);
    }

    #[test]
    fn test_dot_and_sums_of_squares() {
        let u = [1.0, -2.0, 0.5, 3.0];
        let v = [0.25, 4.0, -8.0, 1.0];
        let (uv, uu, vv) = dot_and_sums_of_squares(&u, &v);
        let plain_dot = u.iter().zip(&v).fold(0.0_f64, |s, (a, b)| s + a * b);
        assert_eq!(uv.to_bits(), plain_dot.to_bits());
        assert_eq!(uu.norm().to_bits(), plain_norm(&u).to_bits());
        assert_eq!(vv.norm().to_bits(), plain_norm(&v).to_bits());
        let s = 2.0_f64.powi(600);
        let us: Vec<f64> = u.iter().map(|x| x * s).collect();
        let (_, uu_s, _) = dot_and_sums_of_squares(&us, &v);
        assert_eq!(uu_s.norm(), uu.norm() * s);
    }
}
