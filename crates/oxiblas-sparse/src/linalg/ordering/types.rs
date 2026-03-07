//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use crate::csc::CscMatrix;
use oxiblas_core::scalar::Scalar;
/// Available ordering algorithms for comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderingAlgorithm {
    /// Approximate Minimum Degree.
    AMD,
    /// Multiple Minimum Degree.
    MMD,
    /// Reverse Cuthill-McKee.
    RCM,
    /// Nested Dissection (level-set based).
    NestedDissection,
    /// Multilevel Nested Dissection (METIS-equivalent, pure Rust).
    MultilevelND,
    /// Natural (identity) ordering.
    Natural,
}
impl OrderingAlgorithm {
    /// Returns the name of the algorithm.
    pub fn name(&self) -> &'static str {
        match self {
            OrderingAlgorithm::AMD => "AMD",
            OrderingAlgorithm::MMD => "MMD",
            OrderingAlgorithm::RCM => "RCM",
            OrderingAlgorithm::NestedDissection => "NestedDissection",
            OrderingAlgorithm::MultilevelND => "MultilevelND",
            OrderingAlgorithm::Natural => "Natural",
        }
    }
}
/// Configuration for Nested Dissection ordering.
#[derive(Debug, Clone)]
pub struct NestedDissectionConfig {
    /// Minimum subgraph size to continue recursion (default: 20).
    /// Subgraphs smaller than this use AMD ordering.
    pub min_size: usize,
    /// Maximum recursion depth (default: 50).
    pub max_depth: usize,
    /// Separator balance tolerance (default: 0.2).
    /// Allows separators to create partitions that differ by up to this fraction.
    pub balance_tolerance: f64,
}
/// Result of ordering quality comparison.
#[derive(Debug, Clone)]
pub struct OrderingComparison {
    /// Name of the ordering algorithm.
    pub name: String,
    /// Estimated non-zeros in L.
    pub nnz_l: usize,
    /// Fill-in ratio.
    pub fill_ratio: f64,
    /// Computation time in microseconds (if measured).
    pub time_us: Option<u64>,
}
/// Elimination tree for a symmetric matrix.
///
/// parent\[j\] = min { i : i > j and L\[i,j\] != 0 }
#[derive(Debug, Clone)]
pub struct EliminationTree {
    /// Parent pointers (None for roots).
    parent: Vec<Option<usize>>,
    /// Children lists.
    children: Vec<Vec<usize>>,
}
impl EliminationTree {
    /// Computes the elimination tree from a symmetric matrix.
    pub fn new<T: Scalar>(a: &CscMatrix<T>) -> Self {
        let parent = elimination_tree(a);
        let n = parent.len();
        let mut children = vec![Vec::new(); n];
        for (j, &p) in parent.iter().enumerate() {
            if let Some(i) = p {
                children[i].push(j);
            }
        }
        Self { parent, children }
    }
    /// Returns the parent of node j, or None if j is a root.
    pub fn parent(&self, j: usize) -> Option<usize> {
        self.parent[j]
    }
    /// Returns the children of node j.
    pub fn children(&self, j: usize) -> &[usize] {
        &self.children[j]
    }
    /// Returns the number of nodes.
    pub fn len(&self) -> usize {
        self.parent.len()
    }
    /// Returns true if the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.parent.is_empty()
    }
    /// Returns the roots of the forest.
    pub fn roots(&self) -> Vec<usize> {
        self.parent
            .iter()
            .enumerate()
            .filter(|(_, p)| p.is_none())
            .map(|(i, _)| i)
            .collect()
    }
    /// Returns the depth of node j.
    pub fn depth(&self, j: usize) -> usize {
        let mut d = 0;
        let mut current = j;
        while let Some(p) = self.parent[current] {
            d += 1;
            current = p;
        }
        d
    }
}
/// Symbolic Cholesky factorization.
///
/// Computes the sparsity pattern of L without performing numeric factorization.
#[derive(Debug, Clone)]
pub struct SymbolicCholesky {
    /// Number of rows/columns.
    n: usize,
    /// Column pointers for L.
    l_col_ptrs: Vec<usize>,
    /// Row indices for L (pattern only).
    l_row_indices: Vec<usize>,
    /// Elimination tree parent pointers.
    parent: Vec<Option<usize>>,
    /// Post-ordering of elimination tree.
    post_order: Vec<usize>,
}
impl SymbolicCholesky {
    /// Computes the symbolic factorization of a symmetric matrix.
    ///
    /// Only the lower triangle of the input is used.
    pub fn new<T: Scalar>(a: &CscMatrix<T>) -> Self {
        let n = a.nrows();
        let parent = elimination_tree(a);
        let post_order = postorder_tree(&parent);
        let col_counts = column_counts(a, &parent, &post_order);
        let mut l_col_ptrs = Vec::with_capacity(n + 1);
        l_col_ptrs.push(0);
        for &count in &col_counts {
            l_col_ptrs.push(l_col_ptrs.last().expect("collection should be non-empty") + count);
        }
        let l_row_indices = compute_l_pattern(a, &parent, &l_col_ptrs);
        Self {
            n,
            l_col_ptrs,
            l_row_indices,
            parent,
            post_order,
        }
    }
    /// Returns the number of rows/columns.
    pub fn n(&self) -> usize {
        self.n
    }
    /// Returns the number of non-zeros in L.
    pub fn nnz(&self) -> usize {
        self.l_row_indices.len()
    }
    /// Returns the column pointers for L.
    pub fn l_col_ptrs(&self) -> &[usize] {
        &self.l_col_ptrs
    }
    /// Returns the row indices for L.
    pub fn l_row_indices(&self) -> &[usize] {
        &self.l_row_indices
    }
    /// Returns the elimination tree.
    pub fn parent(&self) -> &[Option<usize>] {
        &self.parent
    }
    /// Returns the post-ordering.
    pub fn post_order(&self) -> &[usize] {
        &self.post_order
    }
}
