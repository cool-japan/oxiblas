//! Helper functions shared across iterative solvers.

use oxiblas_core::scalar::{Field, Real, Scalar};

pub(super) fn givens_rotation_gmres<T: Scalar<Real = T> + Clone + Field + Real>(
    a: T,
    b: T,
) -> (T, T, T) {
    if Scalar::abs(b.clone()) <= <T as Scalar>::epsilon() {
        return (T::one(), T::zero(), a);
    }

    if Scalar::abs(a.clone()) <= <T as Scalar>::epsilon() {
        let sign = if b >= T::zero() {
            T::one()
        } else {
            T::zero() - T::one()
        };
        return (T::zero(), sign, Scalar::abs(b));
    }

    let r = Real::sqrt(a.clone() * a.clone() + b.clone() * b.clone());
    let c = a / r.clone();
    let s = b / r.clone();

    (c, s, r)
}

pub(super) fn solve_upper_triangular<T: Scalar<Real = T> + Clone + Field + Real>(
    h: &[Vec<T>],
    g: &[T],
    m: usize,
) -> Vec<T> {
    let mut y = vec![T::zero(); m];

    for i in (0..m).rev() {
        let mut sum = g[i].clone();
        for j in (i + 1)..m {
            sum = sum - h[j][i].clone() * y[j].clone();
        }
        if Scalar::abs(h[i][i].clone()) > <T as Scalar>::epsilon() {
            y[i] = sum / h[i][i].clone();
        }
    }

    y
}

pub(super) fn dot<T: Scalar + Clone + Field>(a: &[T], b: &[T]) -> T {
    assert_eq!(a.len(), b.len());
    let mut sum = T::zero();
    for i in 0..a.len() {
        sum = sum + a[i].clone() * b[i].clone();
    }
    sum
}

pub(super) fn norm<T: Scalar<Real = T> + Clone + Field + Real>(v: &[T]) -> T {
    Real::sqrt(dot(v, v))
}

pub(super) fn solve_lower_triangular_s<T: Scalar<Real = T> + Clone + Field + Real>(
    m: &[Vec<T>],
    b: &[T],
) -> Vec<T> {
    let s = b.len();
    let mut x = vec![T::zero(); s];

    for i in 0..s {
        let mut sum = b[i].clone();
        for j in 0..i {
            sum = sum - m[i][j].clone() * x[j].clone();
        }
        if Scalar::abs(m[i][i].clone()) > <T as Scalar>::epsilon() {
            x[i] = sum / m[i][i].clone();
        } else {
            x[i] = sum;
        }
    }

    x
}

pub(super) fn block_qr_gmres<T>(vecs: &[Vec<T>], n: usize, p: usize) -> (Vec<Vec<T>>, Vec<Vec<T>>)
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let mut q: Vec<Vec<T>> = vecs.to_vec();
    let mut r: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); p]).collect();

    for j in 0..p {
        for i in 0..j {
            let dot_val = dot(&q[j], &q[i]);
            r[i][j] = dot_val.clone();
            for k in 0..n {
                q[j][k] = q[j][k].clone() - dot_val.clone() * q[i][k].clone();
            }
        }

        let nrm = norm(&q[j]);
        r[j][j] = nrm.clone();

        if nrm > <T as Scalar>::epsilon() {
            for k in 0..n {
                q[j][k] = q[j][k].clone() / nrm.clone();
            }
        }
    }

    (q, r)
}

pub(super) fn block_inner_prod_gmres<T>(v: &[Vec<T>], w: &[Vec<T>], p: usize) -> Vec<Vec<T>>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    let mut result: Vec<Vec<T>> = (0..p).map(|_| vec![T::zero(); p]).collect();
    for i in 0..p {
        for j in 0..p {
            result[i][j] = dot(&v[i], &w[j]);
        }
    }
    result
}

pub(super) fn solve_block_ls_gmres<T>(
    h: &[Vec<T>],
    rhs: &[Vec<T>],
    m: usize,
    n_cols: usize,
    p: usize,
) -> Vec<Vec<T>>
where
    T: Scalar<Real = T> + Clone + Field + Real,
{
    if n_cols == 0 {
        return vec![];
    }

    let mut a: Vec<Vec<T>> = h.to_vec();
    let mut b_mat: Vec<Vec<T>> = rhs.to_vec();

    // Householder QR
    for j in 0..n_cols.min(m) {
        let mut col_norm_sq = T::zero();
        for i in j..m {
            col_norm_sq = col_norm_sq + a[i][j].clone() * a[i][j].clone();
        }
        let col_norm = Real::sqrt(col_norm_sq);

        if col_norm < <T as Scalar>::epsilon() {
            continue;
        }

        let sign = if a[j][j] >= T::zero() {
            T::one()
        } else {
            T::zero() - T::one()
        };
        let v0 = a[j][j].clone() + sign * col_norm.clone();

        let mut v_h: Vec<T> = vec![T::zero(); m - j];
        v_h[0] = T::one();
        for i in 1..(m - j) {
            if Scalar::abs(v0.clone()) > <T as Scalar>::epsilon() {
                v_h[i] = a[j + i][j].clone() / v0.clone();
            }
        }

        let mut vtv = T::zero();
        for vi in &v_h {
            vtv = vtv + vi.clone() * vi.clone();
        }
        let tau = if Scalar::abs(vtv.clone()) > <T as Scalar>::epsilon() {
            (T::one() + T::one()) / vtv
        } else {
            T::zero()
        };

        for k in j..n_cols {
            let mut dot_v = T::zero();
            for i in 0..(m - j) {
                dot_v = dot_v + v_h[i].clone() * a[j + i][k].clone();
            }
            for i in 0..(m - j) {
                a[j + i][k] = a[j + i][k].clone() - tau.clone() * v_h[i].clone() * dot_v.clone();
            }
        }

        for k in 0..p {
            let mut dot_v = T::zero();
            for i in 0..(m - j) {
                dot_v = dot_v + v_h[i].clone() * b_mat[j + i][k].clone();
            }
            for i in 0..(m - j) {
                b_mat[j + i][k] =
                    b_mat[j + i][k].clone() - tau.clone() * v_h[i].clone() * dot_v.clone();
            }
        }
    }

    // Back substitution
    let mut y: Vec<Vec<T>> = vec![vec![T::zero(); p]; n_cols];

    for j in (0..n_cols).rev() {
        for k in 0..p {
            let mut sum = b_mat[j][k].clone();
            for i in (j + 1)..n_cols {
                sum = sum - a[j][i].clone() * y[i][k].clone();
            }
            if Scalar::abs(a[j][j].clone()) > <T as Scalar>::epsilon() {
                y[j][k] = sum / a[j][j].clone();
            } else {
                y[j][k] = T::zero();
            }
        }
    }

    y
}
