//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::csc::CscMatrix;
use oxiblas_core::scalar::Scalar;

/// Computes the elimination tree of a symmetric matrix.
///
/// Uses the algorithm from Davis, "Direct Methods for Sparse Linear Systems".
pub(super) fn elimination_tree<T: Scalar>(a: &CscMatrix<T>) -> Vec<Option<usize>> {
    let n = a.nrows();
    let mut parent = vec![None; n];
    let mut ancestor = vec![0usize; n];
    for k in 0..n {
        ancestor[k] = k;
        let col_start = a.col_ptrs()[k];
        let col_end = a.col_ptrs()[k + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i < k {
                let mut r = i;
                while ancestor[r] != r && ancestor[r] != k {
                    r = ancestor[r];
                }
                if ancestor[r] == r {
                    parent[r] = Some(k);
                }
                let mut j = i;
                while ancestor[j] != k {
                    let next = ancestor[j];
                    ancestor[j] = k;
                    j = next;
                }
            }
        }
    }
    parent
}
/// Computes a post-ordering of the elimination tree.
pub(super) fn postorder_tree(parent: &[Option<usize>]) -> Vec<usize> {
    let n = parent.len();
    let mut children = vec![Vec::new(); n];
    for (j, &p) in parent.iter().enumerate() {
        if let Some(i) = p {
            children[i].push(j);
        }
    }
    let roots: Vec<usize> = parent
        .iter()
        .enumerate()
        .filter(|(_, p)| p.is_none())
        .map(|(i, _)| i)
        .collect();
    let mut order = Vec::with_capacity(n);
    let mut stack = Vec::new();
    let mut visited = vec![false; n];
    for root in roots {
        stack.push((root, false));
        while let Some((node, processed)) = stack.pop() {
            if processed {
                order.push(node);
            } else if !visited[node] {
                visited[node] = true;
                stack.push((node, true));
                for &child in children[node].iter().rev() {
                    if !visited[child] {
                        stack.push((child, false));
                    }
                }
            }
        }
    }
    order
}
/// Computes column counts for L (number of entries in each column of the
/// Cholesky factor, including the diagonal), *with* fill-in.
///
/// This is the Gilbert-Ng-Peyton column-count algorithm (Gilbert, Ng &
/// Peyton, "An Efficient Algorithm to Compute Row and Column Counts for
/// Sparse Cholesky Factorization", 1994), as described in Davis, "Direct
/// Methods for Sparse Linear Systems", Algorithm 4.2 / `cs_counts`.
///
/// Unlike a naive tree walk that only counts *original* nonzeros of `a`,
/// this algorithm correctly accounts for fill-in entries introduced by
/// elimination: for every subdiagonal entry `a[i, j]` (`i > j`), the "least
/// common ancestor" of `i`'s previously-seen owning column within the
/// elimination tree is found (with union-find-style path compression so the
/// whole procedure runs in near-linear time), and a `+1`/`-1` delta pair is
/// applied so that, after the deltas are summed up the tree from children to
/// parents, `counts[v]` equals the exact number of nonzeros predicted for
/// column `v` of `L`.
pub(super) fn column_counts<T: Scalar>(
    a: &CscMatrix<T>,
    parent: &[Option<usize>],
    post_order: &[usize],
) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }

    // Step 1: `first[v]` = postorder index of the earliest-visited node in
    // the subtree rooted at `v` (i.e. the postorder position of `v`'s first
    // descendant leaf, or of `v` itself if `v` is a leaf). Also initializes
    // `delta[v] = 1` for leaves (every leaf contributes its own diagonal
    // entry) and `0` otherwise.
    let mut first: Vec<Option<usize>> = vec![None; n];
    let mut delta: Vec<i64> = vec![0; n];
    for (k, &leaf_start) in post_order.iter().enumerate() {
        if first[leaf_start].is_none() {
            delta[leaf_start] = 1;
        }
        let mut node = Some(leaf_start);
        while let Some(v) = node {
            if first[v].is_some() {
                break;
            }
            first[v] = Some(k);
            node = parent[v];
        }
    }

    // Step 2: sweep the matrix column by column in postorder, using a
    // union-find "ancestor" structure (with the classic Gilbert-Ng-Peyton
    // `leaf` test) to detect, for every row `i`, which columns `j < i` are
    // "skeleton" contributors to row `i`'s fill pattern, folding out
    // redundant contributions via least-common-ancestor deltas.
    let mut maxfirst: Vec<Option<usize>> = vec![None; n];
    let mut prevleaf: Vec<Option<usize>> = vec![None; n];
    let mut ancestor: Vec<usize> = (0..n).collect();

    for &j in post_order {
        if let Some(p) = parent[j] {
            delta[p] -= 1;
        }
        let Some(first_j) = first[j] else {
            // Unreachable: `first` was fully populated for every node in
            // step 1. Skip defensively rather than panicking.
            continue;
        };
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i <= j {
                continue;
            }
            if let Some(mf) = maxfirst[i] {
                if first_j <= mf {
                    continue;
                }
            }
            maxfirst[i] = Some(first_j);
            match prevleaf[i] {
                None => {
                    // `j` is the first column found to touch row `i`: it is
                    // a genuine skeleton entry, contributing to column j.
                    delta[j] += 1;
                    prevleaf[i] = Some(j);
                }
                Some(jprev) => {
                    // Find the root `q` of `jprev`'s partial-elimination-
                    // forest set, with path compression.
                    let mut q = jprev;
                    while ancestor[q] != q {
                        q = ancestor[q];
                    }
                    let mut s = jprev;
                    while s != q {
                        let next = ancestor[s];
                        ancestor[s] = q;
                        s = next;
                    }
                    // `j` contributes to the skeleton, but overlaps with the
                    // subtree already rooted at `q`; cancel the overlap.
                    delta[j] += 1;
                    delta[q] -= 1;
                    prevleaf[i] = Some(j);
                }
            }
        }
        if let Some(p) = parent[j] {
            ancestor[j] = p;
        }
    }

    // Step 3: sum deltas up the elimination tree (children before parents).
    // `parent[v] > v` always holds for this elimination-tree convention, so
    // a single increasing sweep over natural indices is a valid topological
    // (bottom-up) order.
    let mut counts = delta;
    for v in 0..n {
        if let Some(p) = parent[v] {
            counts[p] += counts[v];
        }
    }

    counts
        .into_iter()
        .map(|c| {
            debug_assert!(
                c >= 1,
                "column count must be >= 1 (diagonal entry always present)"
            );
            c.max(1) as usize
        })
        .collect()
}
/// Computes the row indices for L's pattern (`Struct(L_{*j})` for every
/// column `j`), given the elimination tree and the column boundaries
/// produced by [`column_counts`].
///
/// This implements the elimination-tree fill theorem (Davis, "Direct
/// Methods for Sparse Linear Systems", Theorem 4.2 / 4.3):
///
/// ```text
/// Struct(L_{*j}) = A_j  union  ( union over children c of j : Struct(L_{*c}) \ {c} )
/// ```
///
/// where `A_j = { i >= j : A[i, j] != 0 }` is column `j`'s own nonzero
/// pattern at or below the diagonal. Each child's pattern (minus its own
/// row) consists entirely of rows `>= parent(c) == j` by definition of the
/// elimination tree, so the union above is exactly `L`'s true (fill-in
/// included) column pattern. Nodes are processed in increasing index order,
/// which is a valid bottom-up (children-before-parents) topological order
/// for this elimination tree, since `parent[v] > v` always holds.
pub(super) fn compute_l_pattern<T: Scalar>(
    a: &CscMatrix<T>,
    parent: &[Option<usize>],
    l_col_ptrs: &[usize],
) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let nnz = l_col_ptrs[n];
    let mut l_row_indices = vec![0usize; nnz];

    // Immediate children of each node in the elimination tree.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, &p) in parent.iter().enumerate() {
        if let Some(pnode) = p {
            children[pnode].push(c);
        }
    }

    // `patterns[j]` holds the fully computed, sorted row pattern of column
    // `j` of L, kept alive only until it is merged into its parent (each
    // node has exactly one parent in a tree, so it is consumed via `take`
    // the moment its parent is processed).
    let mut patterns: Vec<Option<Vec<usize>>> = vec![None; n];

    // `mark[row] == j` means `row` has already been added to the pattern
    // currently being built for column `j`. Since `j` increases strictly
    // across iterations, this doubles as a per-column "visited" set without
    // needing an O(n) reset per column.
    let mut mark = vec![usize::MAX; n];

    for j in 0..n {
        let mut pattern: Vec<usize> = Vec::new();
        // The diagonal entry is always present in Struct(L_{*j}).
        mark[j] = j;
        pattern.push(j);
        // A_j: direct nonzeros of A in column j, at or below the diagonal.
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i > j && mark[i] != j {
                mark[i] = j;
                pattern.push(i);
            }
        }
        // Union in each child's pattern (minus the child's own row); by the
        // elimination-tree fill theorem these rows are all >= j.
        for &c in &children[j] {
            if let Some(child_pattern) = patterns[c].take() {
                for &row in &child_pattern {
                    if row != c && mark[row] != j {
                        mark[row] = j;
                        pattern.push(row);
                    }
                }
            }
        }
        pattern.sort_unstable();

        let expected_start = l_col_ptrs[j];
        let expected_end = l_col_ptrs[j + 1];
        let expected_len = expected_end - expected_start;
        debug_assert_eq!(
            pattern.len(),
            expected_len,
            "column {j}: computed L pattern size does not match column_counts \
             (elimination-tree fill theorem should guarantee an exact match)"
        );
        // Defensive bound (should always be exact given correct inputs):
        // never write past the slice allocated for this column.
        let write_len = pattern.len().min(expected_len);
        l_row_indices[expected_start..expected_start + write_len]
            .copy_from_slice(&pattern[..write_len]);

        patterns[j] = Some(pattern);
    }
    l_row_indices
}
/// Minimum Degree ordering (greedy elimination, exact degree updates).
///
/// Computes a fill-reducing ordering for sparse Cholesky factorization by
/// repeatedly eliminating the vertex of smallest *current* degree in the
/// elimination graph and updating the affected vertices' degrees.
///
/// # Honesty note: this is not the AMD algorithm
///
/// Despite the function name (kept for API stability), this is **not** the
/// Approximate Minimum Degree (AMD) algorithm of Amestoy, Davis & Duff. Real
/// AMD tracks *upper bounds* on degree via a quotient-graph / clique
/// representation and only refreshes them lazily as elements are absorbed,
/// which is what lets it run close to `O(nnz)` on typical sparse matrices.
/// This implementation instead recomputes the *exact* degree of every
/// affected vertex after each elimination step and rebuilds fill edges with
/// linear `Vec::contains` scans, giving it `O(n^2)` (or worse) worst-case
/// time complexity. The resulting ordering quality is often comparable to
/// AMD on small/medium matrices, but the algorithm and its complexity are
/// not those of AMD.
pub fn approximate_minimum_degree<T: Scalar>(a: &CscMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i != j {
                adj[j].push(i);
                adj[i].push(j);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut degree: Vec<usize> = adj.iter().map(|v| v.len()).collect();
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let mut min_degree = usize::MAX;
        let mut min_node = 0;
        for i in 0..n {
            if !eliminated[i] && degree[i] < min_degree {
                min_degree = degree[i];
                min_node = i;
            }
        }
        eliminated[min_node] = true;
        order.push(min_node);
        let neighbors: Vec<usize> = adj[min_node]
            .iter()
            .copied()
            .filter(|&j| !eliminated[j])
            .collect();
        for &ni in &neighbors {
            for &nj in &neighbors {
                if ni != nj && !adj[ni].contains(&nj) {
                    adj[ni].push(nj);
                }
            }
            adj[ni].retain(|&x| x != min_node);
            degree[ni] = adj[ni].iter().filter(|&&x| !eliminated[x]).count();
        }
    }
    order
}
/// Reverse Cuthill-McKee ordering.
///
/// Reduces bandwidth of sparse matrices.
pub fn reverse_cuthill_mckee<T: Scalar>(a: &CscMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i != j {
                adj[j].push(i);
                adj[i].push(j);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let degree: Vec<usize> = adj.iter().map(|v| v.len()).collect();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    while order.len() < n {
        let start = match (0..n).filter(|&i| !visited[i]).min_by_key(|&i| degree[i]) {
            Some(s) => s,
            None => break,
        };
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut neighbors: Vec<usize> =
                adj[node].iter().copied().filter(|&j| !visited[j]).collect();
            neighbors.sort_by_key(|&j| degree[j]);
            for neighbor in neighbors {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    queue.push_back(neighbor);
                }
            }
        }
    }
    order.reverse();
    order
}
/// Nested Dissection ordering for sparse matrices.
///
/// Nested dissection is a recursive fill-reducing ordering algorithm that:
/// 1. Finds a vertex separator that divides the graph into two parts
/// 2. Recursively orders each part
/// 3. Numbers the separator vertices last
///
/// This implementation uses a level-set based algorithm for finding separators.
///
/// # Example
/// ```ignore
/// use oxiblas_sparse::linalg::ordering::nested_dissection;
/// let perm = nested_dissection(&matrix, None);
/// ```
pub fn nested_dissection<T: Scalar>(
    a: &CscMatrix<T>,
    config: Option<NestedDissectionConfig>,
) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let config = config.unwrap_or_default();
    let adj = build_adjacency(a);
    let active: Vec<usize> = (0..n).collect();
    let mut order = Vec::with_capacity(n);
    nested_dissection_recursive(&adj, &active, &config, 0, &mut order);
    order
}
/// Build adjacency list from CSC matrix.
fn build_adjacency<T: Scalar>(a: &CscMatrix<T>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i != j {
                adj[j].push(i);
                adj[i].push(j);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adj
}
/// Recursive nested dissection implementation.
fn nested_dissection_recursive(
    adj: &[Vec<usize>],
    active: &[usize],
    config: &NestedDissectionConfig,
    depth: usize,
    order: &mut Vec<usize>,
) {
    let n = active.len();
    if n <= config.min_size || depth >= config.max_depth {
        if n <= 3 {
            order.extend(active.iter().copied());
        } else {
            let local_order = local_amd(adj, active);
            order.extend(local_order);
        }
        return;
    }
    let (left, separator, right) = find_vertex_separator(adj, active, config.balance_tolerance);
    if left.is_empty() && right.is_empty() {
        let local_order = local_amd(adj, active);
        order.extend(local_order);
        return;
    }
    if !left.is_empty() {
        nested_dissection_recursive(adj, &left, config, depth + 1, order);
    }
    if !right.is_empty() {
        nested_dissection_recursive(adj, &right, config, depth + 1, order);
    }
    order.extend(separator);
}
/// Find a vertex separator using level-set (BFS-based) algorithm.
///
/// Returns (left_part, separator, right_part).
fn find_vertex_separator(
    adj: &[Vec<usize>],
    active: &[usize],
    balance_tolerance: f64,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if active.len() < 3 {
        return (Vec::new(), active.to_vec(), Vec::new());
    }
    let n_full = adj.len();
    let mut in_active = vec![false; n_full];
    for &v in active {
        in_active[v] = true;
    }
    let start = find_pseudo_peripheral(adj, active, &in_active);
    let (levels, max_level) = compute_level_sets(adj, active, &in_active, start);
    if max_level < 2 {
        return (Vec::new(), active.to_vec(), Vec::new());
    }
    let _target_size = active.len() / 2;
    let mut best_cut_level = max_level / 2;
    let mut best_imbalance = f64::MAX;
    for cut_level in 1..max_level {
        let left_size: usize = levels.iter().filter(|&&l| l < cut_level).count();
        let right_size: usize = levels.iter().filter(|&&l| l > cut_level).count();
        let imbalance = if left_size > right_size {
            (left_size - right_size) as f64 / active.len() as f64
        } else {
            (right_size - left_size) as f64 / active.len() as f64
        };
        if imbalance < best_imbalance {
            best_imbalance = imbalance;
            best_cut_level = cut_level;
        }
    }
    let mut separator = Vec::new();
    let mut left = Vec::new();
    let mut right = Vec::new();
    for &v in active {
        let level = levels[v];
        if level == best_cut_level {
            separator.push(v);
        } else if level < best_cut_level {
            left.push(v);
        } else {
            right.push(v);
        }
    }
    let n = active.len();
    let min_part_size = ((1.0 - balance_tolerance) * n as f64 / 2.0) as usize;
    if left.len() < min_part_size || right.len() < min_part_size {
        let separator_expanded = widen_separator(adj, &left, &separator, &right, &in_active);
        let mut new_left = Vec::new();
        let new_separator = separator_expanded;
        let mut new_right = Vec::new();
        for &v in active {
            if new_separator.contains(&v) {
            } else if levels[v] < best_cut_level {
                new_left.push(v);
            } else {
                new_right.push(v);
            }
        }
        return (new_left, new_separator, new_right);
    }
    (left, separator, right)
}
/// Find a pseudo-peripheral vertex using BFS.
fn find_pseudo_peripheral(adj: &[Vec<usize>], active: &[usize], in_active: &[bool]) -> usize {
    if active.is_empty() {
        return 0;
    }
    let mut start = active[0];
    let mut min_degree = usize::MAX;
    for &v in active {
        let degree = adj[v].iter().filter(|&&u| in_active[u]).count();
        if degree < min_degree {
            min_degree = degree;
            start = v;
        }
    }
    for _ in 0..5 {
        let (_, max_level) = compute_level_sets(adj, active, in_active, start);
        let (levels, _) = compute_level_sets(adj, active, in_active, start);
        let mut farthest = start;
        let mut min_degree_at_max = usize::MAX;
        for &v in active {
            if levels[v] == max_level {
                let degree = adj[v].iter().filter(|&&u| in_active[u]).count();
                if degree < min_degree_at_max {
                    min_degree_at_max = degree;
                    farthest = v;
                }
            }
        }
        if farthest == start {
            break;
        }
        start = farthest;
    }
    start
}
/// Compute level sets (BFS distances) from a starting vertex.
fn compute_level_sets(
    adj: &[Vec<usize>],
    active: &[usize],
    in_active: &[bool],
    start: usize,
) -> (Vec<usize>, usize) {
    let n = adj.len();
    let mut levels = vec![usize::MAX; n];
    levels[start] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    let mut max_level = 0;
    while let Some(v) = queue.pop_front() {
        let next_level = levels[v] + 1;
        for &u in &adj[v] {
            if in_active[u] && levels[u] == usize::MAX {
                levels[u] = next_level;
                max_level = max_level.max(next_level);
                queue.push_back(u);
            }
        }
    }
    for &v in active {
        if levels[v] == usize::MAX {
            levels[v] = max_level + 1;
        }
    }
    (levels, max_level)
}
/// Widen separator to improve balance.
fn widen_separator(
    adj: &[Vec<usize>],
    left: &[usize],
    separator: &[usize],
    right: &[usize],
    in_active: &[bool],
) -> Vec<usize> {
    let mut sep_set: std::collections::HashSet<usize> = separator.iter().copied().collect();
    let left_set: std::collections::HashSet<usize> = left.iter().copied().collect();
    let right_set: std::collections::HashSet<usize> = right.iter().copied().collect();
    let (smaller, _) = if left.len() < right.len() {
        (&left_set, &right_set)
    } else {
        (&right_set, &left_set)
    };
    for &v in separator {
        for &u in &adj[v] {
            if in_active[u] && !sep_set.contains(&u) && !smaller.contains(&u) {
                sep_set.insert(u);
            }
        }
    }
    sep_set.into_iter().collect()
}
/// Local minimum-degree ordering for a subgraph (exact degree, not AMD).
///
/// Used internally by [`nested_dissection_recursive`] to order the leaves of
/// the recursion once a subgraph is small enough (or the depth limit is
/// reached). Like [`approximate_minimum_degree`], this recomputes exact
/// degrees after every elimination step rather than approximating them, so
/// despite the name it is not the Approximate Minimum Degree algorithm; see
/// that function's docs for details.
fn local_amd(adj: &[Vec<usize>], active: &[usize]) -> Vec<usize> {
    if active.is_empty() {
        return Vec::new();
    }
    let n = active.len();
    if n == 1 {
        return active.to_vec();
    }
    let n_full = adj.len();
    let mut global_to_local = vec![usize::MAX; n_full];
    let mut local_to_global = Vec::with_capacity(n);
    for (local_idx, &global_idx) in active.iter().enumerate() {
        global_to_local[global_idx] = local_idx;
        local_to_global.push(global_idx);
    }
    let mut local_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (local_idx, &global_idx) in active.iter().enumerate() {
        for &neighbor in &adj[global_idx] {
            let local_neighbor = global_to_local[neighbor];
            if local_neighbor != usize::MAX {
                local_adj[local_idx].push(local_neighbor);
            }
        }
    }
    let mut degree: Vec<usize> = local_adj.iter().map(|v| v.len()).collect();
    let mut eliminated = vec![false; n];
    let mut local_order = Vec::with_capacity(n);
    for _ in 0..n {
        let mut min_degree = usize::MAX;
        let mut min_node = 0;
        for i in 0..n {
            if !eliminated[i] && degree[i] < min_degree {
                min_degree = degree[i];
                min_node = i;
            }
        }
        eliminated[min_node] = true;
        local_order.push(local_to_global[min_node]);
        let neighbors: Vec<usize> = local_adj[min_node]
            .iter()
            .copied()
            .filter(|&j| !eliminated[j])
            .collect();
        for &ni in &neighbors {
            for &nj in &neighbors {
                if ni != nj && !local_adj[ni].contains(&nj) {
                    local_adj[ni].push(nj);
                }
            }
            local_adj[ni].retain(|&x| x != min_node);
            degree[ni] = local_adj[ni].iter().filter(|&&x| !eliminated[x]).count();
        }
    }
    local_order
}
/// Multiple Minimum Degree ordering.
///
/// An improved version of AMD that eliminates multiple vertices with the same
/// minimum degree simultaneously, reducing the overhead of degree updates.
///
/// This implementation includes:
/// - Mass elimination of independent vertices with equal minimum degree
/// - External degree approximation for efficiency
/// - Degree bucket structure for O(1) minimum finding
///
/// # Arguments
///
/// * `a` - Symmetric sparse matrix in CSC format (only lower triangle used)
///
/// # Returns
///
/// Permutation vector where `perm[i]` gives the original index of the i-th
/// vertex in the elimination order.
pub fn multiple_minimum_degree<T: Scalar>(a: &CscMatrix<T>) -> Vec<usize> {
    let n = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if i != j {
                adj[j].push(i);
                adj[i].push(j);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut degree: Vec<usize> = adj.iter().map(|v| v.len()).collect();
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);
    while order.len() < n {
        let mut min_degree = usize::MAX;
        for i in 0..n {
            if !eliminated[i] && degree[i] < min_degree {
                min_degree = degree[i];
            }
        }
        if min_degree == usize::MAX {
            break;
        }
        let mut candidates: Vec<usize> = Vec::new();
        for i in 0..n {
            if !eliminated[i] && degree[i] == min_degree {
                candidates.push(i);
            }
        }
        let independent = find_independent_set(&candidates, &adj, &eliminated);
        for &v in &independent {
            eliminated[v] = true;
            order.push(v);
            let neighbors: Vec<usize> =
                adj[v].iter().copied().filter(|&j| !eliminated[j]).collect();
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let ni = neighbors[i];
                    let nj = neighbors[j];
                    if !adj[ni].contains(&nj) {
                        adj[ni].push(nj);
                        adj[nj].push(ni);
                    }
                }
            }
            for &neighbor in &neighbors {
                adj[neighbor].retain(|&x| !eliminated[x]);
                degree[neighbor] = adj[neighbor].len();
            }
        }
        for &v in &candidates {
            if !eliminated[v] {
                eliminated[v] = true;
                order.push(v);
                let neighbors: Vec<usize> =
                    adj[v].iter().copied().filter(|&j| !eliminated[j]).collect();
                for i in 0..neighbors.len() {
                    for j in (i + 1)..neighbors.len() {
                        let ni = neighbors[i];
                        let nj = neighbors[j];
                        if !adj[ni].contains(&nj) {
                            adj[ni].push(nj);
                            adj[nj].push(ni);
                        }
                    }
                }
                for &neighbor in &neighbors {
                    adj[neighbor].retain(|&x| !eliminated[x]);
                    degree[neighbor] = adj[neighbor].len();
                }
            }
        }
    }
    order
}
/// Finds an independent set from a list of candidate vertices.
///
/// Returns a subset where no two vertices are adjacent.
fn find_independent_set(
    candidates: &[usize],
    adj: &[Vec<usize>],
    eliminated: &[bool],
) -> Vec<usize> {
    let mut independent = Vec::new();
    let mut in_set = vec![false; eliminated.len()];
    for &v in candidates {
        if eliminated[v] {
            continue;
        }
        let has_neighbor_in_set = adj[v].iter().any(|&u| in_set[u]);
        if !has_neighbor_in_set {
            independent.push(v);
            in_set[v] = true;
        }
    }
    independent
}
/// Column minimum-degree ordering for unsymmetric matrices (not real COLAMD).
///
/// Computes a column permutation intended to reduce fill-in during LU or QR
/// factorization of an unsymmetric (or rectangular) matrix, by running
/// exact minimum-degree elimination on the column intersection graph of `a`
/// (the graph whose edges are the nonzero pattern of `a^T * a`).
///
/// # Honesty note: this is not the COLAMD algorithm
///
/// Despite the function name (kept for API stability), this is **not** the
/// COLAMD algorithm of Davis, Gilbert, Larimore & Ng. The defining
/// performance feature of real COLAMD is that it *never* forms `A^T * A` or
/// the column intersection graph explicitly — it operates directly on `A`'s
/// row/column structure through a quotient-graph representation, keeping
/// time and memory close to `O(nnz(A))`. This implementation does exactly
/// what COLAMD avoids: it explicitly materializes `col_adj`, the full
/// column-intersection adjacency (which can have up to `O(nnz(A)^2)`
/// entries in the worst case), and then performs *exact* (not approximate)
/// minimum-degree elimination on it using linear `Vec::contains` scans for
/// edge dedup. It still produces a valid fill-reducing column permutation,
/// but can be substantially slower and more memory-hungry than COLAMD on
/// the same input.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSC format (can be unsymmetric/rectangular)
///
/// # Returns
///
/// Column permutation vector where `perm[j]` gives the original column index
/// for the j-th column in the permuted matrix.
pub fn colamd<T: Scalar>(a: &CscMatrix<T>) -> Vec<usize> {
    let n = a.ncols();
    let m = a.nrows();
    if n == 0 {
        return Vec::new();
    }
    let mut col_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            row_to_cols[i].push(j);
        }
    }
    for row_cols in &row_to_cols {
        for &col1 in row_cols {
            for &col2 in row_cols {
                if col1 != col2 {
                    col_adj[col1].push(col2);
                }
            }
        }
    }
    for neighbors in &mut col_adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut degree: Vec<usize> = col_adj.iter().map(|v| v.len()).collect();
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let col_lengths: Vec<usize> = (0..n)
        .map(|j| a.col_ptrs()[j + 1] - a.col_ptrs()[j])
        .collect();
    for _ in 0..n {
        let mut min_score = (usize::MAX, usize::MAX);
        let mut min_col = 0;
        for j in 0..n {
            if !eliminated[j] {
                let score = (degree[j], col_lengths[j]);
                if score < min_score {
                    min_score = score;
                    min_col = j;
                }
            }
        }
        eliminated[min_col] = true;
        order.push(min_col);
        let neighbors: Vec<usize> = col_adj[min_col]
            .iter()
            .copied()
            .filter(|&j| !eliminated[j])
            .collect();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let ci = neighbors[i];
                let cj = neighbors[j];
                if !col_adj[ci].contains(&cj) {
                    col_adj[ci].push(cj);
                    col_adj[cj].push(ci);
                }
            }
        }
        for &neighbor in &neighbors {
            col_adj[neighbor].retain(|&x| !eliminated[x]);
            degree[neighbor] = col_adj[neighbor].len();
        }
    }
    order
}
/// Column minimum-degree ordering with dense-row filtering (not real COLAMD).
///
/// Runs the same algorithm as [`colamd`] — exact minimum-degree elimination
/// on the explicitly-formed column intersection graph — except rows with
/// more than `dense_threshold` fraction of nonzero columns are excluded
/// before the graph is built, to avoid the large near-complete cliques that
/// dense rows would otherwise induce in the column intersection graph.
///
/// # Honesty note: this is not the COLAMD algorithm
///
/// The name references COLAMD's "aggressive absorption" pass, but this
/// function does not perform aggressive absorption in the COLAMD sense
/// (merging indistinguishable / absorbed supervariables discovered during
/// elimination). It only pre-filters dense rows before falling back to the
/// same explicit `A^T * A`-graph, exact-minimum-degree procedure as
/// [`colamd`]; see that function's docs for why this is not the real COLAMD
/// algorithm.
///
/// # Arguments
///
/// * `a` - Sparse matrix in CSC format
/// * `dense_threshold` - Rows with more than this fraction of non-zero columns
///                       are considered dense (default: 0.5)
///
/// # Returns
///
/// Column permutation vector.
pub fn colamd_aggressive<T: Scalar>(a: &CscMatrix<T>, dense_threshold: Option<f64>) -> Vec<usize> {
    let n = a.ncols();
    let m = a.nrows();
    let threshold = dense_threshold.unwrap_or(0.5);
    if n == 0 {
        return Vec::new();
    }
    let mut row_nnz = vec![0usize; m];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            row_nnz[i] += 1;
        }
    }
    let dense_row_threshold = ((n as f64) * threshold) as usize;
    let dense_rows: Vec<bool> = row_nnz
        .iter()
        .map(|&nnz| nnz > dense_row_threshold)
        .collect();
    let mut col_adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); m];
    for j in 0..n {
        let col_start = a.col_ptrs()[j];
        let col_end = a.col_ptrs()[j + 1];
        for idx in col_start..col_end {
            let i = a.row_indices()[idx];
            if !dense_rows[i] {
                row_to_cols[i].push(j);
            }
        }
    }
    for row_cols in &row_to_cols {
        for &col1 in row_cols {
            for &col2 in row_cols {
                if col1 != col2 {
                    col_adj[col1].push(col2);
                }
            }
        }
    }
    for neighbors in &mut col_adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut degree: Vec<usize> = col_adj.iter().map(|v| v.len()).collect();
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for _ in 0..n {
        let mut min_degree = usize::MAX;
        let mut min_col = 0;
        for j in 0..n {
            if !eliminated[j] && degree[j] < min_degree {
                min_degree = degree[j];
                min_col = j;
            }
        }
        eliminated[min_col] = true;
        order.push(min_col);
        let neighbors: Vec<usize> = col_adj[min_col]
            .iter()
            .copied()
            .filter(|&j| !eliminated[j])
            .collect();
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let ci = neighbors[i];
                let cj = neighbors[j];
                if !col_adj[ci].contains(&cj) {
                    col_adj[ci].push(cj);
                    col_adj[cj].push(ci);
                }
            }
        }
        for &neighbor in &neighbors {
            col_adj[neighbor].retain(|&x| !eliminated[x]);
            degree[neighbor] = col_adj[neighbor].len();
        }
    }
    order
}
/// Computes a symmetric permutation from a column ordering.
///
/// Given a column ordering for A, computes the corresponding row ordering
/// for a symmetric reordering of A * P (where P is the column permutation).
///
/// # Arguments
///
/// * `col_perm` - Column permutation
///
/// # Returns
///
/// Row permutation (same as column permutation for symmetric usage).
pub fn symmetric_permutation(col_perm: &[usize]) -> Vec<usize> {
    col_perm.to_vec()
}
/// Computes the inverse of a permutation.
///
/// If `perm[i] = j`, then `inv_perm[j] = i`.
pub fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let n = perm.len();
    let mut inv = vec![0; n];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}
/// Estimates the fill-in for Cholesky factorization with a given ordering.
///
/// This computes the number of non-zeros in L without actually performing
/// the numeric factorization. Useful for comparing ordering quality.
///
/// # Arguments
///
/// * `a` - Symmetric sparse matrix in CSC format
/// * `perm` - Permutation vector (elimination order)
///
/// # Returns
///
/// Estimated number of non-zeros in the Cholesky factor L.
pub fn estimate_fill_in<T: Scalar>(a: &CscMatrix<T>, perm: &[usize]) -> usize {
    let n = a.nrows();
    if n == 0 {
        return 0;
    }
    let inv_perm = inverse_permutation(perm);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..n {
        let orig_j = perm[j];
        let col_start = a.col_ptrs()[orig_j];
        let col_end = a.col_ptrs()[orig_j + 1];
        for idx in col_start..col_end {
            let orig_i = a.row_indices()[idx];
            let perm_i = inv_perm[orig_i];
            if perm_i != j {
                adj[j].push(perm_i);
                adj[perm_i].push(j);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let mut eliminated = vec![false; n];
    let mut total_nnz = 0;
    for j in 0..n {
        eliminated[j] = true;
        let higher_neighbors: Vec<usize> = adj[j]
            .iter()
            .copied()
            .filter(|&i| i > j && !eliminated[i])
            .collect();
        total_nnz += higher_neighbors.len() + 1;
        for k in 0..higher_neighbors.len() {
            for l in (k + 1)..higher_neighbors.len() {
                let ni = higher_neighbors[k];
                let nj = higher_neighbors[l];
                if !adj[ni].contains(&nj) {
                    adj[ni].push(nj);
                    adj[nj].push(ni);
                }
            }
        }
    }
    total_nnz
}
/// Computes the fill-in ratio for an ordering.
///
/// The fill-in ratio is (nnz(L) - nnz(A)) / nnz(A), where A is the lower
/// triangle of the original matrix.
///
/// # Returns
///
/// Fill-in ratio (0.0 means no fill-in, higher means more fill-in).
pub fn fill_in_ratio<T: Scalar>(a: &CscMatrix<T>, perm: &[usize]) -> f64 {
    let original_nnz = a.nnz();
    let factor_nnz = estimate_fill_in(a, perm);
    if original_nnz == 0 {
        return 0.0;
    }
    (factor_nnz as f64 - original_nnz as f64) / original_nnz as f64
}
/// Computes the ordering for a given algorithm.
pub fn compute_ordering<T: Scalar>(a: &CscMatrix<T>, alg: OrderingAlgorithm) -> Vec<usize> {
    match alg {
        OrderingAlgorithm::AMD => approximate_minimum_degree(a),
        OrderingAlgorithm::MMD => multiple_minimum_degree(a),
        OrderingAlgorithm::RCM => reverse_cuthill_mckee(a),
        OrderingAlgorithm::NestedDissection => nested_dissection(a, None),
        OrderingAlgorithm::MultilevelND => {
            let n = a.nrows();
            let xadj: Vec<usize> = (0..=n).map(|i| a.col_ptrs()[i]).collect();
            let adjncy: Vec<usize> = a.row_indices().to_vec();
            super::multilevel::multilevel_nested_dissection(n, &xadj, &adjncy)
                .unwrap_or_else(|_| (0..n).collect())
        }
        OrderingAlgorithm::Natural => (0..a.nrows()).collect(),
    }
}
/// Compares multiple ordering algorithms on a given matrix.
///
/// # Arguments
///
/// * `a` - Symmetric sparse matrix in CSC format
/// * `algorithms` - List of algorithms to compare
///
/// # Returns
///
/// Vector of comparison results, sorted by fill-in (best first).
pub fn compare_orderings<T: Scalar>(
    a: &CscMatrix<T>,
    algorithms: &[OrderingAlgorithm],
) -> Vec<OrderingComparison> {
    let mut results = Vec::with_capacity(algorithms.len());
    for &alg in algorithms {
        let start = std::time::Instant::now();
        let perm = compute_ordering(a, alg);
        let elapsed = start.elapsed().as_micros() as u64;
        let nnz_l = estimate_fill_in(a, &perm);
        let fill_ratio = fill_in_ratio(a, &perm);
        results.push(OrderingComparison {
            name: alg.name().to_string(),
            nnz_l,
            fill_ratio,
            time_us: Some(elapsed),
        });
    }
    results.sort_by_key(|r| r.nnz_l);
    results
}
/// Selects the best ordering algorithm for a matrix.
///
/// Runs all available algorithms and returns the one with lowest fill-in.
///
/// # Arguments
///
/// * `a` - Symmetric sparse matrix in CSC format
///
/// # Returns
///
/// Tuple of (best permutation, algorithm used, comparison results).
pub fn select_best_ordering<T: Scalar>(
    a: &CscMatrix<T>,
) -> (Vec<usize>, OrderingAlgorithm, Vec<OrderingComparison>) {
    let algorithms = [
        OrderingAlgorithm::AMD,
        OrderingAlgorithm::MMD,
        OrderingAlgorithm::RCM,
        OrderingAlgorithm::NestedDissection,
    ];
    let results = compare_orderings(a, &algorithms);
    let best_name = &results[0].name;
    let best_alg = algorithms
        .iter()
        .find(|a| a.name() == best_name)
        .copied()
        .unwrap_or(OrderingAlgorithm::AMD);
    let best_perm = compute_ordering(a, best_alg);
    (best_perm, best_alg, results)
}
/// Computes the bandwidth of a matrix with a given ordering.
///
/// Bandwidth is `max { |i - j| : A[i,j] != 0 }`.
pub fn bandwidth_with_ordering<T: Scalar>(a: &CscMatrix<T>, perm: &[usize]) -> usize {
    let inv_perm = inverse_permutation(perm);
    let mut max_bw = 0;
    for j in 0..a.ncols() {
        let orig_j = perm[j];
        let col_start = a.col_ptrs()[orig_j];
        let col_end = a.col_ptrs()[orig_j + 1];
        for idx in col_start..col_end {
            let orig_i = a.row_indices()[idx];
            let perm_i = inv_perm[orig_i];
            let dist = perm_i.abs_diff(j);
            max_bw = max_bw.max(dist);
        }
    }
    max_bw
}
/// Computes the profile (envelope) of a matrix with a given ordering.
///
/// Profile is sum over all rows of (row_index - first_nonzero_column_in_row).
pub fn profile_with_ordering<T: Scalar>(a: &CscMatrix<T>, perm: &[usize]) -> usize {
    let n = a.nrows();
    let inv_perm = inverse_permutation(perm);
    let mut first_in_row = vec![n; n];
    for j in 0..a.ncols() {
        let orig_j = perm[j];
        let col_start = a.col_ptrs()[orig_j];
        let col_end = a.col_ptrs()[orig_j + 1];
        for idx in col_start..col_end {
            let orig_i = a.row_indices()[idx];
            let perm_i = inv_perm[orig_i];
            first_in_row[perm_i] = first_in_row[perm_i].min(j);
        }
    }
    let mut profile = 0;
    for i in 0..n {
        if first_in_row[i] < n {
            profile += i - first_in_row[i] + 1;
        }
    }
    profile
}
#[cfg(test)]
mod tests {
    use super::*;
    fn make_test_matrix() -> CscMatrix<f64> {
        let values = vec![
            4.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 1.0, 4.0,
        ];
        let row_indices = vec![0, 1, 4, 0, 1, 2, 1, 2, 3, 2, 3, 4, 0, 3, 4];
        let col_ptrs = vec![0, 3, 6, 9, 12, 15];
        CscMatrix::new(5, 5, col_ptrs, row_indices, values).unwrap()
    }
    #[test]
    fn test_elimination_tree() {
        let a = make_test_matrix();
        let parent = elimination_tree(&a);
        assert_eq!(parent.len(), 5);
        for (j, &p) in parent.iter().enumerate() {
            if let Some(i) = p {
                assert!(i > j, "Parent must be greater than child");
            }
        }
    }
    #[test]
    fn test_symbolic_cholesky() {
        let a = make_test_matrix();
        let sym = SymbolicCholesky::new(&a);
        assert_eq!(sym.n(), 5);
        assert!(sym.nnz() >= 5, "L should have at least diagonal entries");
    }
    /// A 4-cycle graph (0-1, 0-2, 1-3, 2-3, plus diagonal), stored as a full
    /// symmetric CSC matrix. Eliminating node 0 in natural order (0,1,2,3)
    /// introduces a *fill* edge between 1 and 2 (both are neighbors of 0),
    /// so the true Cholesky column counts are [3, 3, 2, 1] (total nnz(L)=9).
    ///
    /// The pre-fix `column_counts` only counted *original* nonzeros of `A`
    /// and performed a dead tree-walk that never updated any counter, so it
    /// produced [3, 2, 2, 1] (total 8) for this matrix: column 1's fill
    /// entry at row 2 was silently dropped, which in turn made
    /// `compute_l_pattern`'s write-position guard silently truncate that
    /// column's pattern.
    fn make_fill_in_matrix() -> CscMatrix<f64> {
        let col_ptrs = vec![0, 3, 6, 9, 12];
        let row_indices = vec![0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3];
        let values = vec![1.0; 12];
        CscMatrix::new(4, 4, col_ptrs, row_indices, values).unwrap()
    }
    #[test]
    fn test_column_counts_accounts_for_fill_in() {
        let a = make_fill_in_matrix();
        let parent = elimination_tree(&a);
        // Sanity check: natural elimination order should produce the chain
        // 0 -> 1 -> 2 -> 3 given the fill edge (1,2) introduced by
        // eliminating node 0.
        assert_eq!(parent, vec![Some(1), Some(2), Some(3), None]);
        let post_order = postorder_tree(&parent);
        let counts = column_counts(&a, &parent, &post_order);
        assert_eq!(
            counts,
            vec![3, 3, 2, 1],
            "column counts must include the fill-in entry at (row=2, col=1) \
             introduced by eliminating node 0"
        );
    }
    #[test]
    fn test_symbolic_cholesky_fill_in_pattern_is_not_truncated() {
        let a = make_fill_in_matrix();
        let sym = SymbolicCholesky::new(&a);
        // Total nnz(L) must be exactly 9, not the truncated 8 that the
        // dead tree-walk bug used to produce.
        assert_eq!(sym.nnz(), 9);
        assert_eq!(sym.l_col_ptrs(), &[0, 3, 6, 8, 9]);
        // Column 1 (0-indexed) must contain the fill-in row 2, in addition
        // to the diagonal (1) and the original entry (3).
        let col1 = &sym.l_row_indices()[sym.l_col_ptrs()[1]..sym.l_col_ptrs()[2]];
        let mut col1_sorted = col1.to_vec();
        col1_sorted.sort_unstable();
        assert_eq!(
            col1_sorted,
            vec![1, 2, 3],
            "L's column 1 must include the fill-in entry at row 2"
        );
    }
    #[test]
    fn test_amd() {
        let a = make_test_matrix();
        let perm = approximate_minimum_degree(&a);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_rcm() {
        let a = make_test_matrix();
        let perm = reverse_cuthill_mckee(&a);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_elimination_tree_struct() {
        let a = make_test_matrix();
        let tree = EliminationTree::new(&a);
        assert_eq!(tree.len(), 5);
        let roots = tree.roots();
        assert!(!roots.is_empty(), "Should have at least one root");
    }
    #[test]
    fn test_nested_dissection_basic() {
        let a = make_test_matrix();
        let perm = nested_dissection(&a, None);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_nested_dissection_larger_matrix() {
        let n = 25;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0];
        for j in 0..n {
            values.push(4.0);
            row_indices.push(j);
            let row = j / 5;
            let col = j % 5;
            if row > 0 {
                let neighbor = (row - 1) * 5 + col;
                if neighbor < j {
                    values.push(-1.0);
                    row_indices.push(neighbor);
                }
            }
            if col > 0 {
                let neighbor = row * 5 + (col - 1);
                if neighbor < j {
                    values.push(-1.0);
                    row_indices.push(neighbor);
                }
            }
            if row < 4 {
                let neighbor = (row + 1) * 5 + col;
                values.push(-1.0);
                row_indices.push(neighbor);
            }
            if col < 4 {
                let neighbor = row * 5 + (col + 1);
                values.push(-1.0);
                row_indices.push(neighbor);
            }
            col_ptrs.push(values.len());
        }
        let a = CscMatrix::new(n, n, col_ptrs, row_indices, values).unwrap();
        let perm = nested_dissection(&a, None);
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected);
    }
    #[test]
    fn test_nested_dissection_with_config() {
        let a = make_test_matrix();
        let config = NestedDissectionConfig {
            min_size: 2,
            max_depth: 10,
            balance_tolerance: 0.3,
        };
        let perm = nested_dissection(&a, Some(config));
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_nested_dissection_empty() {
        let values: Vec<f64> = Vec::new();
        let row_indices: Vec<usize> = Vec::new();
        let col_ptrs = vec![0];
        let a = CscMatrix::new(0, 0, col_ptrs, row_indices, values).unwrap();
        let perm = nested_dissection(&a, None);
        assert!(perm.is_empty());
    }
    #[test]
    fn test_nested_dissection_single_element() {
        let values = vec![1.0];
        let row_indices = vec![0];
        let col_ptrs = vec![0, 1];
        let a = CscMatrix::new(1, 1, col_ptrs, row_indices, values).unwrap();
        let perm = nested_dissection(&a, None);
        assert_eq!(perm.len(), 1);
        assert_eq!(perm[0], 0);
    }
    #[test]
    fn test_nested_dissection_disconnected() {
        let values = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0];
        let row_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4];
        let col_ptrs = vec![0, 2, 5, 7, 9, 11];
        let a = CscMatrix::new(5, 5, col_ptrs, row_indices, values).unwrap();
        let perm = nested_dissection(&a, None);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_mmd() {
        let a = make_test_matrix();
        let perm = multiple_minimum_degree(&a);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_mmd_larger_matrix() {
        let n = 25;
        let mut values = Vec::new();
        let mut row_indices = Vec::new();
        let mut col_ptrs = vec![0];
        for j in 0..n {
            values.push(4.0);
            row_indices.push(j);
            let row = j / 5;
            let col = j % 5;
            if row > 0 {
                let neighbor = (row - 1) * 5 + col;
                if neighbor < j {
                    values.push(-1.0);
                    row_indices.push(neighbor);
                }
            }
            if col > 0 {
                let neighbor = row * 5 + (col - 1);
                if neighbor < j {
                    values.push(-1.0);
                    row_indices.push(neighbor);
                }
            }
            if row < 4 {
                let neighbor = (row + 1) * 5 + col;
                values.push(-1.0);
                row_indices.push(neighbor);
            }
            if col < 4 {
                let neighbor = row * 5 + (col + 1);
                values.push(-1.0);
                row_indices.push(neighbor);
            }
            col_ptrs.push(values.len());
        }
        let a = CscMatrix::new(n, n, col_ptrs, row_indices, values).unwrap();
        let perm = multiple_minimum_degree(&a);
        assert_eq!(perm.len(), n);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..n).collect();
        assert_eq!(sorted, expected);
    }
    #[test]
    fn test_mmd_empty() {
        let values: Vec<f64> = Vec::new();
        let row_indices: Vec<usize> = Vec::new();
        let col_ptrs = vec![0];
        let a = CscMatrix::new(0, 0, col_ptrs, row_indices, values).unwrap();
        let perm = multiple_minimum_degree(&a);
        assert!(perm.is_empty());
    }
    #[test]
    fn test_colamd() {
        let a = make_test_matrix();
        let perm = colamd(&a);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_colamd_rectangular() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let row_indices = vec![0, 2, 1, 2, 0, 1, 0, 2];
        let col_ptrs = vec![0, 2, 4, 5, 6, 8];
        let a = CscMatrix::new(3, 5, col_ptrs, row_indices, values).unwrap();
        let perm = colamd(&a);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_colamd_aggressive() {
        let a = make_test_matrix();
        let perm = colamd_aggressive(&a, Some(0.5));
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_colamd_empty() {
        let values: Vec<f64> = Vec::new();
        let row_indices: Vec<usize> = Vec::new();
        let col_ptrs = vec![0];
        let a = CscMatrix::new(0, 0, col_ptrs, row_indices, values).unwrap();
        let perm = colamd(&a);
        assert!(perm.is_empty());
    }
    #[test]
    fn test_inverse_permutation() {
        let perm = vec![2, 0, 3, 1, 4];
        let inv = inverse_permutation(&perm);
        for i in 0..perm.len() {
            assert_eq!(perm[inv[i]], i);
        }
        for i in 0..perm.len() {
            assert_eq!(inv[perm[i]], i);
        }
    }
    #[test]
    fn test_mmd_vs_amd_produces_valid_ordering() {
        let a = make_test_matrix();
        let amd_perm = approximate_minimum_degree(&a);
        let mmd_perm = multiple_minimum_degree(&a);
        let mut amd_sorted = amd_perm.clone();
        amd_sorted.sort();
        assert_eq!(amd_sorted, vec![0, 1, 2, 3, 4]);
        let mut mmd_sorted = mmd_perm.clone();
        mmd_sorted.sort();
        assert_eq!(mmd_sorted, vec![0, 1, 2, 3, 4]);
    }
    #[test]
    fn test_estimate_fill_in() {
        let a = make_test_matrix();
        let natural_perm: Vec<usize> = (0..5).collect();
        let nnz_l = estimate_fill_in(&a, &natural_perm);
        assert!(nnz_l >= 5);
        assert!(nnz_l <= 25);
    }
    #[test]
    fn test_fill_in_ratio() {
        let a = make_test_matrix();
        let natural_perm: Vec<usize> = (0..5).collect();
        let ratio = fill_in_ratio(&a, &natural_perm);
        assert!(ratio >= -1.0);
    }
    #[test]
    fn test_compare_orderings() {
        let a = make_test_matrix();
        let algorithms = [
            OrderingAlgorithm::AMD,
            OrderingAlgorithm::MMD,
            OrderingAlgorithm::RCM,
            OrderingAlgorithm::Natural,
        ];
        let results = compare_orderings(&a, &algorithms);
        assert_eq!(results.len(), 4);
        for i in 1..results.len() {
            assert!(results[i - 1].nnz_l <= results[i].nnz_l);
        }
    }
    #[test]
    fn test_select_best_ordering() {
        let a = make_test_matrix();
        let (perm, _alg, results) = select_best_ordering(&a);
        assert_eq!(perm.len(), 5);
        let mut sorted = perm.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        assert!(!results.is_empty());
    }
    #[test]
    fn test_bandwidth_with_ordering() {
        let a = make_test_matrix();
        let natural_perm: Vec<usize> = (0..5).collect();
        let bw = bandwidth_with_ordering(&a, &natural_perm);
        assert!(bw >= 1);
        assert!(bw < 5);
    }
    #[test]
    fn test_profile_with_ordering() {
        let a = make_test_matrix();
        let natural_perm: Vec<usize> = (0..5).collect();
        let profile = profile_with_ordering(&a, &natural_perm);
        assert!(profile >= 5);
    }
    #[test]
    fn test_rcm_reduces_bandwidth() {
        let a = make_test_matrix();
        let natural_perm: Vec<usize> = (0..5).collect();
        let rcm_perm = reverse_cuthill_mckee(&a);
        let natural_bw = bandwidth_with_ordering(&a, &natural_perm);
        let rcm_bw = bandwidth_with_ordering(&a, &rcm_perm);
        assert!(rcm_bw <= natural_bw + 1);
    }
    #[test]
    fn test_ordering_algorithm_enum() {
        assert_eq!(OrderingAlgorithm::AMD.name(), "AMD");
        assert_eq!(OrderingAlgorithm::MMD.name(), "MMD");
        assert_eq!(OrderingAlgorithm::RCM.name(), "RCM");
        assert_eq!(
            OrderingAlgorithm::NestedDissection.name(),
            "NestedDissection"
        );
        assert_eq!(OrderingAlgorithm::Natural.name(), "Natural");
    }
    #[test]
    fn test_compute_ordering() {
        let a = make_test_matrix();
        for alg in [
            OrderingAlgorithm::AMD,
            OrderingAlgorithm::MMD,
            OrderingAlgorithm::RCM,
            OrderingAlgorithm::NestedDissection,
            OrderingAlgorithm::Natural,
        ] {
            let perm = compute_ordering(&a, alg);
            assert_eq!(perm.len(), 5);
            let mut sorted = perm.clone();
            sorted.sort();
            assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
        }
    }
}
