//! OxiBLAS Sparse - Sparse matrix support.
//!
//! This crate provides sparse matrix formats and operations:
//!
//! - **CSR**: Compressed Sparse Row format
//! - **CSC**: Compressed Sparse Column format
//! - **COO**: Coordinate format (for construction)
//! - **DIA**: Diagonal format (for banded matrices)
//! - **ELL**: ELLPACK format (for GPU computation)
//! - **BSR**: Block Sparse Row format (for block-structured matrices)
//! - **BSC**: Block Sparse Column format (for column-oriented block structure)
//! - **HYB**: Hybrid ELL+COO format (for irregular sparsity patterns)
//! - **SELL**: Sliced ELLPACK format (for GPU-optimized row-variable matrices)
//!
//! # Sparse Matrix Formats
//!
//! ## CSR (Compressed Sparse Row)
//!
//! Efficient for row-wise operations and matrix-vector products.
//! Stores values row by row.
//!
//! ## CSC (Compressed Sparse Column)
//!
//! Efficient for column-wise operations and direct solvers.
//! Stores values column by column.
//!
//! ## DIA (Diagonal)
//!
//! Efficient for banded matrices (tridiagonal, pentadiagonal, etc.).
//! Stores diagonals explicitly.
//!
//! ## ELL (ELLPACK)
//!
//! Efficient for GPU computation with uniform row lengths.
//! Fixed number of entries per row.
//!
//! ## BSR (Block Sparse Row)
//!
//! Efficient for block-structured matrices (FEM, etc.).
//! Stores dense blocks in CSR-like structure.
//!
//! # Example
//!
//! ```
//! use oxiblas_sparse::{CsrMatrix, CscMatrix, DiaMatrix, EllMatrix, BsrMatrix};
//!
//! // Create a sparse matrix in CSR format
//! // [1 0 2]
//! // [0 3 0]
//! // [4 0 5]
//! let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//! let col_indices = vec![0, 2, 1, 0, 2];
//! let row_ptrs = vec![0, 2, 3, 5];
//!
//! let csr = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
//! assert_eq!(csr.nnz(), 5);
//!
//! // Convert to CSC
//! let csc = csr.to_csc();
//! assert_eq!(csc.nnz(), 5);
//!
//! // Create a tridiagonal matrix in DIA format
//! let sub = vec![1.0, 1.0];
//! let main = vec![2.0, 2.0, 2.0];
//! let super_diag = vec![1.0, 1.0];
//! let dia = DiaMatrix::tridiagonal(sub, main, super_diag).unwrap();
//! assert_eq!(dia.ndiag(), 3);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::module_name_repetitions)]
// Sparse library uses Clone::clone() for generic T where Copy isn't always available
#![allow(clippy::clone_on_copy)]
// Manual assign patterns are often clearer in numerical code
#![allow(clippy::assign_op_pattern)]
// Numerical code often has complex generic types for performance
#![allow(clippy::type_complexity)]
// Loop index variables are common in matrix operations
#![allow(clippy::needless_range_loop)]
// Sparse functions have many parameters by design
#![allow(clippy::too_many_arguments)]
// Partial ordering comparisons are intentional
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Bounds in two places for clarity in generic numerical code
#![allow(clippy::multiple_bound_locations)]
// Manual slice copying for explicit control
#![allow(clippy::manual_memcpy)]
// Vec vs slice in internal APIs is acceptable
#![allow(clippy::ptr_arg)]
// Iterative solver implementations may have unusual control flow
#![allow(clippy::never_loop)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::iter_cloned_collect)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::manual_strip)]
#![allow(clippy::doc_overindented_list_items)]
#![allow(clippy::unnecessary_unwrap)]

pub mod bsc;
pub mod bsr;
pub mod convert;
pub mod coo;
pub mod csc;
pub mod csr;
pub mod dia;
pub mod ell;
pub mod graph;
pub mod hyb;
pub mod linalg;
pub mod mtx;
pub mod ops;
pub mod sell;

pub use bsc::{BscError, BscMatrix};
pub use bsr::{BsrError, BsrMatrix, DenseBlock};
pub use convert::{
    RecommendedFormat,
    SparsityAnalysis,
    // Analysis utilities
    analyze_sparsity_pattern,
    // New format conversions
    bsc_to_bsr,
    bsc_to_csr,
    bsr_to_bsc,
    // Core conversions
    bsr_to_csr,
    bsr_to_dia,
    bsr_to_ell,
    coo_to_csc,
    coo_to_csr,
    csc_to_coo,
    csc_to_csr,
    csr_to_bsc,
    csr_to_bsr,
    csr_to_coo,
    csr_to_csc,
    csr_to_dia,
    csr_to_ell,
    csr_to_hyb,
    csr_to_sell,
    dia_to_bsr,
    dia_to_csr,
    dia_to_ell,
    ell_to_bsr,
    ell_to_csr,
    ell_to_dia,
    ell_to_hyb,
    hyb_to_csr,
    hyb_to_ell,
    sell_to_csr,
};
pub use coo::{CooMatrix, CooMatrixBuilder};
pub use csc::CscMatrix;
pub use csr::CsrMatrix;
pub use dia::{DiaError, DiaMatrix};
pub use ell::{EllError, EllMatrix};
pub use graph::{
    BandwidthProfileResult, BipartiteMatchingResult, BipartiteResult, ConnectedComponentsResult,
    LevelSetResult, PartitionResult, WeightedMatchingResult, bandwidth_profile,
    bandwidth_profile_csc, bipartite_matching, connected_components, connected_components_csc,
    degree_sequence, is_bipartite, is_structurally_symmetric, level_sets, partition_graph_bisect,
    partition_graph_kway, pseudo_peripheral_vertex, weighted_bipartite_matching,
};
pub use hyb::{HybError, HybMatrix, HybStats, HybWidthStrategy};
pub use mtx::{
    MtxError, MtxField, MtxFormat, MtxHeader, MtxObject, MtxSymmetry, read_matrix_market,
    read_matrix_market_coo, read_matrix_market_str, write_matrix_market, write_matrix_market_str,
    write_matrix_market_symmetric,
};
pub use sell::{SellError, SellMatrix, SellStats, SliceSize};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::bsc::BscMatrix;
    pub use crate::bsr::{BsrMatrix, DenseBlock};
    pub use crate::coo::{CooMatrix, CooMatrixBuilder};
    pub use crate::csc::CscMatrix;
    pub use crate::csr::CsrMatrix;
    pub use crate::dia::DiaMatrix;
    pub use crate::ell::EllMatrix;
    pub use crate::graph::{
        BandwidthProfileResult, BipartiteMatchingResult, BipartiteResult,
        ConnectedComponentsResult, LevelSetResult, PartitionResult, WeightedMatchingResult,
        bandwidth_profile, bipartite_matching, connected_components, degree_sequence, is_bipartite,
        is_structurally_symmetric, level_sets, partition_graph_bisect, partition_graph_kway,
        pseudo_peripheral_vertex, weighted_bipartite_matching,
    };
    pub use crate::hyb::{HybMatrix, HybWidthStrategy};
    pub use crate::linalg::prelude::*;
    pub use crate::ops::{spmm, spmm_sparse, spmv};
    pub use crate::sell::{SellMatrix, SliceSize};
}
