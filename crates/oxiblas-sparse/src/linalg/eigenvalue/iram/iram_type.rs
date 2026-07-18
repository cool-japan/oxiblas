//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::IRAMConfig;


/// Implicitly Restarted Arnoldi Method (IRAM).
///
/// IRAM is a memory-efficient eigenvalue algorithm that computes a few eigenvalues
/// of large sparse matrices. It maintains a Krylov subspace of bounded size through
/// implicit restarts using shifted QR iterations.
///
/// # Algorithm Overview
///
/// 1. Build initial Arnoldi factorization: A*V_m = V_m*H_m + f_m*e_m^T
/// 2. Compute Ritz values (eigenvalues of H_m)
/// 3. Select p = m - k unwanted Ritz values as shifts
/// 4. Apply p implicit QR shifts to compress factorization to dimension k
/// 5. Continue Arnoldi from dimension k back to m
/// 6. Repeat until convergence
///
/// For symmetric matrices, IRAM reduces to Implicitly Restarted Lanczos (IRL),
/// where H is tridiagonal and the algorithm is more efficient.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::csr::CsrMatrix;
/// use oxiblas_sparse::linalg::eigenvalue::{IRAM, IRAMConfig, WhichEigenvalues};
///
/// // Create a large sparse matrix
/// let a = CsrMatrix::<f64>::eye(1000);
///
/// let config = IRAMConfig {
///     num_eigenvalues: 10,
///     which: WhichEigenvalues::LargestMagnitude,
///     krylov_dimension: 30,  // ncv = 30 > nev = 10
///     symmetric: true,  // More efficient for symmetric matrices
///     ..Default::default()
/// };
///
/// let iram = IRAM::new(config);
/// let result = iram.compute(&a, None).unwrap();
/// println!("Converged: {}, eigenvalues: {:?}", result.converged, result.eigenvalues_real);
/// ```
pub struct IRAM<T> {
    pub(super) config: IRAMConfig<T>,
}
