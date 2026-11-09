//! A block size of zero made the blocked factorizations loop forever for
//! every `n >= 1` (the block start never advanced); it is now treated as 1.
//!
//! Every call runs on a helper thread under a time limit, so a regression
//! fails the test instead of hanging the run (a detached spinning thread does
//! not keep the test process alive). A block size of zero must also give
//! exactly the result of block size 1.

use std::sync::mpsc;
use std::thread;
use std::time::Duration;

use oxiblas_lapack::cholesky::Cholesky;
use oxiblas_lapack::evd::Hessenberg;
use oxiblas_lapack::lu::Lu;
use oxiblas_lapack::qr::Qr;
use oxiblas_lapack::svd::BidiagFactors;
use oxiblas_matrix::{Mat, MatRef};

/// Far beyond what these tiny factorizations need, even unoptimized.
const LIMIT: Duration = Duration::from_secs(30);

/// Runs `f` on a helper thread and panics if it does not return in [`LIMIT`]
/// (or panics itself).
fn terminates<R, F>(label: &str, f: F) -> R
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        // The receiver only goes away after a timeout, when the result no
        // longer matters.
        let _ = tx.send(f());
    });
    match rx.recv_timeout(LIMIT) {
        Ok(out) => out,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{label}: did not return within {LIMIT:?}")
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{label}: panicked"),
    }
}

/// Diagonally dominant general matrix (nonsingular for the square shapes).
fn general(m: usize, n: usize) -> Mat<f64> {
    let mut a = Mat::zeros(m, n);
    for j in 0..n {
        for i in 0..m {
            let diag = if i == j { 8.0 } else { 0.0 };
            a[(i, j)] = ((3 * i + 5 * j) % 7) as f64 - 2.5 + diag;
        }
    }
    a
}

/// Symmetric positive definite (diagonally dominant) matrix.
fn spd(n: usize) -> Mat<f64> {
    let mut a = Mat::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            a[(i, j)] = if i == j {
                n as f64 + 1.0
            } else {
                1.0 / (1.0 + (i + j) as f64)
            };
        }
    }
    a
}

fn assert_same(label: &str, got: MatRef<'_, f64>, want: MatRef<'_, f64>) {
    assert_eq!(
        (got.nrows(), got.ncols()),
        (want.nrows(), want.ncols()),
        "{label}: shape"
    );
    for j in 0..want.ncols() {
        for i in 0..want.nrows() {
            assert_eq!(
                got[(i, j)].to_bits(),
                want[(i, j)].to_bits(),
                "{label}: ({i}, {j})"
            );
        }
    }
}

fn assert_same_slice(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (idx, (x, y)) in got.iter().zip(want).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{label}[{idx}]");
    }
}

#[test]
fn lu_with_block_size_zero_terminates() {
    for n in [1usize, 2, 5, 9] {
        let label = format!("Lu::compute_with_block_size n={n}");
        let a = general(n, n);
        let b = a.clone();
        let zero = terminates(&label, move || Lu::compute_with_block_size(b.as_ref(), 0))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one = Lu::compute_with_block_size(a.as_ref(), 1)
            .unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&label, zero.lu_matrix(), one.lu_matrix());
        assert_eq!(zero.pivot(), one.pivot(), "{label}: pivots");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn lu_blocked_par_with_block_size_zero_terminates() {
    for n in [1usize, 2, 5, 9] {
        let label = format!("Lu::compute_blocked_par_with_block_size n={n}");
        let a = general(n, n);
        let b = a.clone();
        let zero = terminates(&label, move || {
            Lu::compute_blocked_par_with_block_size(b.as_ref(), 0)
        })
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one = Lu::compute_with_block_size(a.as_ref(), 1)
            .unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&label, zero.lu_matrix(), one.lu_matrix());
        assert_eq!(zero.pivot(), one.pivot(), "{label}: pivots");
    }
}

#[test]
fn cholesky_with_block_size_zero_terminates() {
    for n in [1usize, 2, 5, 9] {
        let label = format!("Cholesky::compute_with_block_size n={n}");
        let a = spd(n);
        let b = a.clone();
        let zero = terminates(&label, move || {
            Cholesky::compute_with_block_size(b.as_ref(), 0)
        })
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one = Cholesky::compute_with_block_size(a.as_ref(), 1)
            .unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&label, zero.l_factor().as_ref(), one.l_factor().as_ref());
    }
}

#[cfg(feature = "parallel")]
#[test]
fn cholesky_blocked_par_with_block_size_zero_terminates() {
    for n in [1usize, 2, 5, 9] {
        let label = format!("Cholesky::compute_blocked_par_with_block_size n={n}");
        let a = spd(n);
        let b = a.clone();
        let zero = terminates(&label, move || {
            Cholesky::compute_blocked_par_with_block_size(b.as_ref(), 0)
        })
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one = Cholesky::compute_with_block_size(a.as_ref(), 1)
            .unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&label, zero.l_factor().as_ref(), one.l_factor().as_ref());
    }
}

#[test]
fn qr_blocked_block_size_zero_terminates() {
    for (m, n) in [(1usize, 1usize), (5, 3), (4, 6), (7, 7)] {
        let label = format!("Qr::compute_blocked {m}x{n}");
        let a = general(m, n);
        let b = a.clone();
        let zero = terminates(&label, move || Qr::compute_blocked(b.as_ref(), 0))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one =
            Qr::compute_blocked(a.as_ref(), 1).unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&label, zero.qr_factors(), one.qr_factors());
        assert_same_slice(&format!("{label} tau"), zero.tau(), one.tau());
    }
}

#[test]
fn bidiag_with_block_size_zero_terminates() {
    // A smaller dimension above 2 takes the blocked path.
    for (m, n) in [(2usize, 2usize), (3, 3), (6, 4), (4, 6), (8, 8)] {
        let label = format!("BidiagFactors::compute_blocked_with_block_size {m}x{n}");
        let a = general(m, n);
        let b = a.clone();
        let zero = terminates(&label, move || {
            BidiagFactors::compute_blocked_with_block_size(b.as_ref(), 0)
        })
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one = BidiagFactors::compute_blocked_with_block_size(a.as_ref(), 1)
            .unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&label, zero.work.as_ref(), one.work.as_ref());
        assert_same_slice(&format!("{label} d"), &zero.d, &one.d);
        assert_same_slice(&format!("{label} e"), &zero.e, &one.e);
        assert_same_slice(&format!("{label} tauq"), &zero.tauq, &one.tauq);
        assert_same_slice(&format!("{label} taup"), &zero.taup, &one.taup);
    }
}

#[test]
fn hessenberg_with_block_size_zero_terminates() {
    // An order above 3 takes the blocked path.
    for n in [1usize, 3, 4, 7] {
        let label = format!("Hessenberg::compute_blocked_with_block_size n={n}");
        let a = general(n, n);
        let b = a.clone();
        let zero = terminates(&label, move || {
            Hessenberg::compute_blocked_with_block_size(b.as_ref(), 0)
        })
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let one = Hessenberg::compute_blocked_with_block_size(a.as_ref(), 1)
            .unwrap_or_else(|e| panic!("{label} (nb = 1): {e}"));
        assert_same(&format!("{label} H"), zero.h(), one.h());
        assert_same(&format!("{label} Q"), zero.q(), one.q());
    }
}
