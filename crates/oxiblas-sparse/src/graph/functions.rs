//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::csc::CscMatrix;
use crate::csr::CsrMatrix;
use oxiblas_core::scalar::Scalar;
/// Find connected components of a sparse matrix viewed as a graph.
///
/// The matrix is treated as the adjacency matrix of an undirected graph.
/// For non-symmetric matrices, the graph is symmetrized (A + A^T).
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format)
///
/// # Returns
///
/// Connected components result with labels, count, and sizes.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::graph::connected_components;
///
/// let result = connected_components(&matrix);
/// println!("Number of components: {}", result.num_components);
/// ```
pub fn connected_components<T: Scalar>(a: &CsrMatrix<T>) -> ConnectedComponentsResult {
    let n = a.nrows();
    if n == 0 {
        return ConnectedComponentsResult {
            labels: Vec::new(),
            num_components: 0,
            component_sizes: Vec::new(),
        };
    }
    let adj = build_symmetric_adjacency(a);
    let mut labels = vec![usize::MAX; n];
    let mut component_sizes = Vec::new();
    let mut current_component = 0;
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        labels[start] = current_component;
        let mut size = 0;
        while let Some(v) = queue.pop_front() {
            size += 1;
            for &u in &adj[v] {
                if labels[u] == usize::MAX {
                    labels[u] = current_component;
                    queue.push_back(u);
                }
            }
        }
        component_sizes.push(size);
        current_component += 1;
    }
    ConnectedComponentsResult {
        labels,
        num_components: current_component,
        component_sizes,
    }
}
/// Find connected components from CSC matrix.
pub fn connected_components_csc<T: Scalar>(a: &CscMatrix<T>) -> ConnectedComponentsResult {
    let n = a.nrows();
    if n == 0 {
        return ConnectedComponentsResult {
            labels: Vec::new(),
            num_components: 0,
            component_sizes: Vec::new(),
        };
    }
    let adj = build_symmetric_adjacency_csc(a);
    let mut labels = vec![usize::MAX; n];
    let mut component_sizes = Vec::new();
    let mut current_component = 0;
    for start in 0..n {
        if labels[start] != usize::MAX {
            continue;
        }
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        labels[start] = current_component;
        let mut size = 0;
        while let Some(v) = queue.pop_front() {
            size += 1;
            for &u in &adj[v] {
                if labels[u] == usize::MAX {
                    labels[u] = current_component;
                    queue.push_back(u);
                }
            }
        }
        component_sizes.push(size);
        current_component += 1;
    }
    ConnectedComponentsResult {
        labels,
        num_components: current_component,
        component_sizes,
    }
}
/// Build symmetric adjacency list from CSR matrix.
fn build_symmetric_adjacency<T: Scalar>(a: &CsrMatrix<T>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        for idx in start..end {
            let j = a.col_indices()[idx];
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adj
}
/// Build symmetric adjacency list from CSC matrix.
fn build_symmetric_adjacency_csc<T: Scalar>(a: &CscMatrix<T>) -> Vec<Vec<usize>> {
    let n = a.nrows();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for j in 0..a.ncols() {
        let start = a.col_ptrs()[j];
        let end = a.col_ptrs()[j + 1];
        for idx in start..end {
            let i = a.row_indices()[idx];
            if i != j {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }
    for neighbors in &mut adj {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adj
}
/// Compute bandwidth and profile of a sparse matrix.
///
/// The bandwidth measures how far non-zero elements are from the diagonal.
/// The profile (envelope) is the sum of the row bandwidths.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format)
///
/// # Returns
///
/// Bandwidth and profile metrics.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::graph::bandwidth_profile;
///
/// let result = bandwidth_profile(&matrix);
/// println!("Bandwidth: {}", result.bandwidth);
/// println!("Profile: {}", result.profile);
/// ```
pub fn bandwidth_profile<T: Scalar>(a: &CsrMatrix<T>) -> BandwidthProfileResult {
    let n = a.nrows();
    if n == 0 {
        return BandwidthProfileResult {
            bandwidth: 0,
            lower_bandwidth: 0,
            upper_bandwidth: 0,
            profile: 0,
            average_bandwidth: 0.0,
        };
    }
    let mut bandwidth = 0usize;
    let mut lower_bandwidth = 0usize;
    let mut upper_bandwidth = 0usize;
    let mut profile = 0usize;
    for i in 0..n {
        let start = a.row_ptrs()[i];
        let end = a.row_ptrs()[i + 1];
        if start == end {
            continue;
        }
        let mut min_col = i;
        let mut max_col = i;
        for idx in start..end {
            let j = a.col_indices()[idx];
            min_col = min_col.min(j);
            max_col = max_col.max(j);
        }
        if min_col < i {
            lower_bandwidth = lower_bandwidth.max(i - min_col);
        }
        if max_col > i {
            upper_bandwidth = upper_bandwidth.max(max_col - i);
        }
        let row_bandwidth = max_col - min_col;
        profile += row_bandwidth;
        bandwidth = bandwidth.max(i.abs_diff(min_col));
        bandwidth = bandwidth.max(i.abs_diff(max_col));
    }
    let average_bandwidth = if n > 0 {
        profile as f64 / n as f64
    } else {
        0.0
    };
    BandwidthProfileResult {
        bandwidth,
        lower_bandwidth,
        upper_bandwidth,
        profile,
        average_bandwidth,
    }
}
/// Compute bandwidth and profile from CSC matrix.
pub fn bandwidth_profile_csc<T: Scalar>(a: &CscMatrix<T>) -> BandwidthProfileResult {
    let n = a.nrows();
    let m = a.ncols();
    if n == 0 || m == 0 {
        return BandwidthProfileResult {
            bandwidth: 0,
            lower_bandwidth: 0,
            upper_bandwidth: 0,
            profile: 0,
            average_bandwidth: 0.0,
        };
    }
    let mut bandwidth = 0usize;
    let mut lower_bandwidth = 0usize;
    let mut upper_bandwidth = 0usize;
    let mut row_min_col = vec![usize::MAX; n];
    let mut row_max_col = vec![0usize; n];
    for j in 0..m {
        let start = a.col_ptrs()[j];
        let end = a.col_ptrs()[j + 1];
        for idx in start..end {
            let i = a.row_indices()[idx];
            row_min_col[i] = row_min_col[i].min(j);
            row_max_col[i] = row_max_col[i].max(j);
            if i > j {
                lower_bandwidth = lower_bandwidth.max(i - j);
            }
            if j > i {
                upper_bandwidth = upper_bandwidth.max(j - i);
            }
            bandwidth = bandwidth.max(i.abs_diff(j));
        }
    }
    let mut profile = 0usize;
    for i in 0..n {
        if row_min_col[i] != usize::MAX {
            profile += row_max_col[i] - row_min_col[i];
        }
    }
    let average_bandwidth = if n > 0 {
        profile as f64 / n as f64
    } else {
        0.0
    };
    BandwidthProfileResult {
        bandwidth,
        lower_bandwidth,
        upper_bandwidth,
        profile,
        average_bandwidth,
    }
}
/// Construct level sets from a root vertex.
///
/// Level sets are the vertices at each distance from the root,
/// computed using breadth-first search.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format)
/// * `root` - Starting vertex
///
/// # Returns
///
/// Level set structure with levels and vertex lists.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::graph::level_sets;
///
/// let result = level_sets(&matrix, 0);
/// println!("Max level: {}", result.max_level);
/// ```
pub fn level_sets<T: Scalar>(a: &CsrMatrix<T>, root: usize) -> LevelSetResult {
    let n = a.nrows();
    if n == 0 || root >= n {
        return LevelSetResult {
            levels: Vec::new(),
            level_sets: Vec::new(),
            max_level: 0,
        };
    }
    let adj = build_symmetric_adjacency(a);
    let mut levels = vec![usize::MAX; n];
    levels[root] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    let mut max_level = 0;
    while let Some(v) = queue.pop_front() {
        let next_level = levels[v] + 1;
        for &u in &adj[v] {
            if levels[u] == usize::MAX {
                levels[u] = next_level;
                max_level = max_level.max(next_level);
                queue.push_back(u);
            }
        }
    }
    let mut level_sets: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for (v, &level) in levels.iter().enumerate() {
        if level != usize::MAX {
            level_sets[level].push(v);
        }
    }
    LevelSetResult {
        levels,
        level_sets,
        max_level,
    }
}
/// Find a pseudo-peripheral vertex.
///
/// A pseudo-peripheral vertex is one that is approximately at the
/// maximum distance from all other vertices. Useful for RCM ordering.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format)
///
/// # Returns
///
/// Index of a pseudo-peripheral vertex, or 0 if matrix is empty.
pub fn pseudo_peripheral_vertex<T: Scalar>(a: &CsrMatrix<T>) -> usize {
    let n = a.nrows();
    if n == 0 {
        return 0;
    }
    let adj = build_symmetric_adjacency(a);
    let mut start = 0;
    let mut min_degree = usize::MAX;
    for (v, neighbors) in adj.iter().enumerate() {
        if neighbors.len() < min_degree {
            min_degree = neighbors.len();
            start = v;
        }
    }
    for _ in 0..5 {
        let result = level_sets_internal(&adj, start, n);
        let mut farthest = start;
        let mut min_degree_at_max = usize::MAX;
        for (v, &level) in result.0.iter().enumerate() {
            if level == result.1 {
                let degree = adj[v].len();
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
/// Internal level set computation (returns levels and max_level).
fn level_sets_internal(adj: &[Vec<usize>], root: usize, n: usize) -> (Vec<usize>, usize) {
    let mut levels = vec![usize::MAX; n];
    levels[root] = 0;
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(root);
    let mut max_level = 0;
    while let Some(v) = queue.pop_front() {
        let next_level = levels[v] + 1;
        for &u in &adj[v] {
            if levels[u] == usize::MAX {
                levels[u] = next_level;
                max_level = max_level.max(next_level);
                queue.push_back(u);
            }
        }
    }
    (levels, max_level)
}
/// Check if a sparse matrix is structurally symmetric.
///
/// Returns true if for every non-zero a_ij, there exists a non-zero a_ji.
pub fn is_structurally_symmetric<T: Scalar>(a: &CsrMatrix<T>) -> bool {
    let n = a.nrows();
    if n != a.ncols() {
        return false;
    }
    for i in 0..n {
        let start_i = a.row_ptrs()[i];
        let end_i = a.row_ptrs()[i + 1];
        for idx in start_i..end_i {
            let j = a.col_indices()[idx];
            if i == j {
                continue;
            }
            let start_j = a.row_ptrs()[j];
            let end_j = a.row_ptrs()[j + 1];
            let found = a.col_indices()[start_j..end_j].binary_search(&i).is_ok();
            if !found {
                return false;
            }
        }
    }
    true
}
/// Compute the degree sequence of the graph.
///
/// Returns the degree (number of neighbors) for each vertex.
pub fn degree_sequence<T: Scalar>(a: &CsrMatrix<T>) -> Vec<usize> {
    let _n = a.nrows();
    let adj = build_symmetric_adjacency(a);
    adj.iter().map(|neighbors| neighbors.len()).collect()
}
/// Check if a sparse matrix viewed as a graph is bipartite.
///
/// A graph is bipartite if its vertices can be divided into two disjoint sets
/// such that every edge connects a vertex in one set to a vertex in the other.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format)
///
/// # Returns
///
/// `BipartiteResult` with the bipartiteness status and partition.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::graph::is_bipartite;
///
/// let result = is_bipartite(&matrix);
/// if result.is_bipartite {
///     println!("Left partition: {:?}", result.left);
///     println!("Right partition: {:?}", result.right);
/// }
/// ```
pub fn is_bipartite<T: Scalar>(a: &CsrMatrix<T>) -> BipartiteResult {
    let n = a.nrows();
    if n == 0 {
        return BipartiteResult {
            is_bipartite: true,
            partition: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
        };
    }
    let adj = build_symmetric_adjacency(a);
    let mut partition = vec![usize::MAX; n];
    let mut is_bipartite_flag = true;
    for start in 0..n {
        if partition[start] != usize::MAX {
            continue;
        }
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        partition[start] = 0;
        while let Some(v) = queue.pop_front() {
            let current_color = partition[v];
            let next_color = 1 - current_color;
            for &u in &adj[v] {
                if partition[u] == usize::MAX {
                    partition[u] = next_color;
                    queue.push_back(u);
                } else if partition[u] == current_color {
                    is_bipartite_flag = false;
                }
            }
        }
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    if is_bipartite_flag {
        for (v, &p) in partition.iter().enumerate() {
            if p == 0 {
                left.push(v);
            } else {
                right.push(v);
            }
        }
    }
    BipartiteResult {
        is_bipartite: is_bipartite_flag,
        partition,
        left,
        right,
    }
}
/// Find maximum bipartite matching using Hopcroft-Karp algorithm.
///
/// Given a bipartite graph, finds a maximum cardinality matching.
/// The algorithm runs in O(E * sqrt(V)) time.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format) representing a bipartite graph
///
/// # Returns
///
/// `BipartiteMatchingResult` with the matching information, or `None`
/// if the graph is not bipartite.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::graph::bipartite_matching;
///
/// if let Some(result) = bipartite_matching(&matrix) {
///     println!("Matching size: {}", result.matching_size);
///     for (l, r) in &result.edges {
///         println!("Edge: {} - {}", l, r);
///     }
/// }
/// ```
pub fn bipartite_matching<T: Scalar>(a: &CsrMatrix<T>) -> Option<BipartiteMatchingResult> {
    let n = a.nrows();
    if n == 0 {
        return Some(BipartiteMatchingResult {
            left_match: Vec::new(),
            right_match: Vec::new(),
            matching_size: 0,
            edges: Vec::new(),
            is_perfect: true,
        });
    }
    let bipartite_result = is_bipartite(a);
    if !bipartite_result.is_bipartite {
        return None;
    }
    let adj = build_symmetric_adjacency(a);
    let left = &bipartite_result.left;
    let right = &bipartite_result.right;
    let num_left = left.len();
    let num_right = right.len();
    let mut vertex_to_left_idx = vec![usize::MAX; n];
    let mut vertex_to_right_idx = vec![usize::MAX; n];
    for (idx, &v) in left.iter().enumerate() {
        vertex_to_left_idx[v] = idx;
    }
    for (idx, &v) in right.iter().enumerate() {
        vertex_to_right_idx[v] = idx;
    }
    let mut left_adj: Vec<Vec<usize>> = vec![Vec::new(); num_left];
    for (left_idx, &v) in left.iter().enumerate() {
        for &u in &adj[v] {
            if vertex_to_right_idx[u] != usize::MAX {
                left_adj[left_idx].push(vertex_to_right_idx[u]);
            }
        }
    }
    let result = hopcroft_karp(&left_adj, num_left, num_right);
    let mut left_match_orig = vec![None; n];
    let mut right_match_orig = vec![None; n];
    let mut edges = Vec::new();
    for (left_idx, &right_opt) in result.0.iter().enumerate() {
        if let Some(right_idx) = right_opt {
            let left_v = left[left_idx];
            let right_v = right[right_idx];
            left_match_orig[left_v] = Some(right_v);
            right_match_orig[right_v] = Some(left_v);
            edges.push((left_v, right_v));
        }
    }
    let matching_size = edges.len();
    let is_perfect =
        matching_size == num_left.min(num_right) && (num_left == 0 || matching_size > 0);
    Some(BipartiteMatchingResult {
        left_match: left_match_orig,
        right_match: right_match_orig,
        matching_size,
        edges,
        is_perfect,
    })
}
/// Hopcroft-Karp algorithm implementation.
///
/// Returns (left_match, right_match) where each entry is Some(matched_idx) or None.
fn hopcroft_karp(
    adj: &[Vec<usize>],
    num_left: usize,
    num_right: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut left_match: Vec<Option<usize>> = vec![None; num_left];
    let mut right_match: Vec<Option<usize>> = vec![None; num_right];
    if num_left == 0 || num_right == 0 {
        return (left_match, right_match);
    }
    let mut dist = vec![usize::MAX; num_left + 1];
    const NIL: usize = usize::MAX;
    loop {
        let mut queue = std::collections::VecDeque::new();
        for u in 0..num_left {
            if left_match[u].is_none() {
                dist[u] = 0;
                queue.push_back(u);
            } else {
                dist[u] = NIL;
            }
        }
        dist[num_left] = NIL;
        while let Some(u) = queue.pop_front() {
            if u == num_left {
                continue;
            }
            for &v in &adj[u] {
                let pair_v = right_match[v];
                let next_u = pair_v.unwrap_or(num_left);
                if dist[next_u] == NIL {
                    dist[next_u] = dist[u] + 1;
                    if next_u != num_left {
                        queue.push_back(next_u);
                    }
                }
            }
        }
        if dist[num_left] == NIL {
            break;
        }
        for u in 0..num_left {
            if left_match[u].is_none() {
                dfs_augment(
                    u,
                    adj,
                    &mut left_match,
                    &mut right_match,
                    &mut dist,
                    num_left,
                );
            }
        }
    }
    (left_match, right_match)
}
/// DFS to find and augment along shortest path.
fn dfs_augment(
    u: usize,
    adj: &[Vec<usize>],
    left_match: &mut [Option<usize>],
    right_match: &mut [Option<usize>],
    dist: &mut [usize],
    num_left: usize,
) -> bool {
    if u == num_left {
        return true;
    }
    for &v in &adj[u] {
        let pair_v = right_match[v];
        let next_u = pair_v.unwrap_or(num_left);
        if dist[next_u] == dist[u] + 1
            && dfs_augment(next_u, adj, left_match, right_match, dist, num_left)
        {
            left_match[u] = Some(v);
            right_match[v] = Some(u);
            return true;
        }
    }
    dist[u] = usize::MAX;
    false
}
/// Find maximum weighted bipartite matching using Hungarian algorithm.
///
/// Given a bipartite graph with edge weights, finds a maximum weight matching.
/// Uses the Hungarian (Kuhn-Munkres) algorithm in O(V^3) time.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format) where values are edge weights
///
/// # Returns
///
/// `WeightedMatchingResult` with the matching and total weight, or `None`
/// if the graph is not bipartite.
pub fn weighted_bipartite_matching<T: Scalar + PartialOrd>(
    a: &CsrMatrix<T>,
) -> Option<WeightedMatchingResult<T>>
where
    T: std::ops::Sub<Output = T> + std::ops::Add<Output = T> + Copy,
{
    let n = a.nrows();
    if n == 0 {
        return Some(WeightedMatchingResult {
            left_match: Vec::new(),
            right_match: Vec::new(),
            matching_size: 0,
            edges: Vec::new(),
            total_weight: T::zero(),
        });
    }
    let bipartite_result = is_bipartite(a);
    if !bipartite_result.is_bipartite {
        return None;
    }
    let left = &bipartite_result.left;
    let right = &bipartite_result.right;
    let num_left = left.len();
    let num_right = right.len();
    if num_left == 0 || num_right == 0 {
        return Some(WeightedMatchingResult {
            left_match: vec![None; n],
            right_match: vec![None; n],
            matching_size: 0,
            edges: Vec::new(),
            total_weight: T::zero(),
        });
    }
    let mut vertex_to_left_idx = vec![usize::MAX; n];
    let mut vertex_to_right_idx = vec![usize::MAX; n];
    for (idx, &v) in left.iter().enumerate() {
        vertex_to_left_idx[v] = idx;
    }
    for (idx, &v) in right.iter().enumerate() {
        vertex_to_right_idx[v] = idx;
    }
    let mut weights: Vec<Vec<Option<T>>> = vec![vec![None; num_right]; num_left];
    for (i, &left_v) in left.iter().enumerate() {
        let start = a.row_ptrs()[left_v];
        let end = a.row_ptrs()[left_v + 1];
        for idx in start..end {
            let right_v = a.col_indices()[idx];
            if vertex_to_right_idx[right_v] != usize::MAX {
                let j = vertex_to_right_idx[right_v];
                let weight = a.values()[idx];
                weights[i][j] = Some(weight);
            }
        }
    }
    let (assignment, total_weight) = hungarian_algorithm(&weights, num_left, num_right);
    let mut left_match_orig = vec![None; n];
    let mut right_match_orig = vec![None; n];
    let mut edges = Vec::new();
    for (left_idx, &right_opt) in assignment.iter().enumerate() {
        if let Some(right_idx) = right_opt {
            let left_v = left[left_idx];
            let right_v = right[right_idx];
            left_match_orig[left_v] = Some(right_v);
            right_match_orig[right_v] = Some(left_v);
            edges.push((left_v, right_v));
        }
    }
    let matching_size = edges.len();
    Some(WeightedMatchingResult {
        left_match: left_match_orig,
        right_match: right_match_orig,
        matching_size,
        edges,
        total_weight,
    })
}
/// Hungarian algorithm for maximum weight bipartite matching.
fn hungarian_algorithm<T: Scalar + PartialOrd + Copy>(
    weights: &[Vec<Option<T>>],
    num_left: usize,
    num_right: usize,
) -> (Vec<Option<usize>>, T)
where
    T: std::ops::Sub<Output = T> + std::ops::Add<Output = T>,
{
    if num_left == 0 || num_right == 0 {
        return (vec![None; num_left], T::zero());
    }
    let dim = num_left.max(num_right);
    let mut u = vec![T::zero(); dim];
    let mut v = vec![T::zero(); dim];
    for i in 0..num_left {
        let mut max_weight = T::zero();
        for j in 0..num_right {
            if let Some(w) = weights[i][j] {
                if w > max_weight {
                    max_weight = w;
                }
            }
        }
        u[i] = max_weight;
    }
    let mut left_match: Vec<Option<usize>> = vec![None; dim];
    let mut right_match: Vec<Option<usize>> = vec![None; dim];
    for i in 0..dim {
        let mut minv = vec![T::zero(); dim];
        let mut visited_right = vec![false; dim];
        let mut way = vec![usize::MAX; dim];
        let mut j0 = 0usize;
        right_match[0] = Some(i);
        loop {
            visited_right[j0] = true;
            let i0 = right_match[j0].unwrap_or(dim);
            if i0 >= dim {
                break;
            }
            let mut delta = T::zero();
            let mut delta_initialized = false;
            let mut j1 = 0usize;
            for j in 0..dim {
                if visited_right[j] {
                    continue;
                }
                let cost = get_weight(weights, i0, j, num_left, num_right, u[i0], v[j]);
                if !delta_initialized || cost < minv[j] {
                    minv[j] = cost;
                    way[j] = j0;
                }
                if minv[j] < delta || !delta_initialized {
                    delta = minv[j];
                    j1 = j;
                    delta_initialized = true;
                }
            }
            if !delta_initialized {
                break;
            }
            for j in 0..dim {
                if visited_right[j] {
                    if let Some(matched_i) = right_match[j] {
                        if matched_i < dim {
                            u[matched_i] = u[matched_i] - delta;
                        }
                    }
                    v[j] = v[j] + delta;
                } else {
                    minv[j] = minv[j] - delta;
                }
            }
            j0 = j1;
            if right_match[j0].is_none() {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            right_match[j0] = right_match.get(j1).copied().flatten();
            if j1 == usize::MAX {
                break;
            }
            if let Some(prev_i) = right_match[j1] {
                left_match[prev_i] = Some(j0);
            }
            j0 = j1;
            if way[j0] == usize::MAX {
                break;
            }
        }
        left_match[i] = Some(j0);
        right_match[j0] = Some(i);
    }
    let mut total_weight = T::zero();
    let mut valid_left_match = vec![None; num_left];
    for i in 0..num_left {
        if let Some(j) = left_match[i] {
            if j < num_right {
                if let Some(w) = weights[i][j] {
                    valid_left_match[i] = Some(j);
                    total_weight = total_weight + w;
                }
            }
        }
    }
    (valid_left_match, total_weight)
}
/// Get weight from weight matrix, handling padding and computing reduced cost.
fn get_weight<T: Scalar + PartialOrd + Copy>(
    weights: &[Vec<Option<T>>],
    i: usize,
    j: usize,
    num_left: usize,
    num_right: usize,
    ui: T,
    vj: T,
) -> T
where
    T: std::ops::Sub<Output = T> + std::ops::Add<Output = T>,
{
    if i >= num_left || j >= num_right {
        return T::zero();
    }
    match weights[i][j] {
        Some(w) => ui + vj - w,
        None => ui + vj,
    }
}
/// Partition a graph into k parts using recursive bisection.
///
/// Uses Kernighan-Lin inspired algorithm for bisection and refines
/// using local search. Good for domain decomposition and load balancing.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format) representing graph adjacency
/// * `k` - Number of partitions (must be power of 2 for recursive bisection)
///
/// # Returns
///
/// Partition result with assignments and edge cut size.
///
/// # Example
///
/// ```ignore
/// use oxiblas_sparse::graph::partition_graph_kway;
///
/// let result = partition_graph_kway(&matrix, 4);
/// println!("Edge cut: {}", result.edge_cut);
/// ```
pub fn partition_graph_kway<T: Scalar>(a: &CsrMatrix<T>, k: usize) -> PartitionResult {
    let n = a.nrows();
    if n == 0 || k == 0 {
        return PartitionResult {
            partition: Vec::new(),
            num_partitions: 0,
            partition_sizes: Vec::new(),
            edge_cut: 0,
        };
    }
    if k == 1 {
        return PartitionResult {
            partition: vec![0; n],
            num_partitions: 1,
            partition_sizes: vec![n],
            edge_cut: 0,
        };
    }
    let adj = build_symmetric_adjacency(a);
    let mut partition = vec![0; n];
    recursive_bisection(&adj, &mut partition, 0, k, 0);
    let mut partition_sizes = vec![0; k];
    for &p in &partition {
        if p < k {
            partition_sizes[p] += 1;
        }
    }
    let edge_cut = compute_edge_cut(&adj, &partition);
    PartitionResult {
        partition,
        num_partitions: k,
        partition_sizes,
        edge_cut,
    }
}
/// Bisect a graph into 2 parts using Kernighan-Lin inspired algorithm.
///
/// # Arguments
///
/// * `a` - Sparse matrix (CSR format) representing graph adjacency
///
/// # Returns
///
/// Partition result with 2 partitions.
pub fn partition_graph_bisect<T: Scalar>(a: &CsrMatrix<T>) -> PartitionResult {
    partition_graph_kway(a, 2)
}
/// Recursive bisection helper.
fn recursive_bisection(
    adj: &[Vec<usize>],
    partition: &mut [usize],
    current_label: usize,
    target_partitions: usize,
    depth: usize,
) {
    let n = adj.len();
    if n == 0 || target_partitions <= 1 {
        return;
    }
    let vertices: Vec<usize> = (0..n).filter(|&i| partition[i] == current_label).collect();
    if vertices.len() <= 1 {
        return;
    }
    let bisect_result = bisect_vertices(adj, &vertices);
    let next_label = current_label + (target_partitions / 2);
    for (i, &v) in vertices.iter().enumerate() {
        if bisect_result[i] == 1 {
            partition[v] = next_label;
        }
    }
    if target_partitions > 2 {
        let half = target_partitions / 2;
        recursive_bisection(adj, partition, current_label, half, depth + 1);
        recursive_bisection(adj, partition, next_label, half, depth + 1);
    }
}
/// Bisect a subset of vertices into two parts.
fn bisect_vertices(adj: &[Vec<usize>], vertices: &[usize]) -> Vec<usize> {
    if vertices.len() <= 1 {
        return vec![0; vertices.len()];
    }
    let mut vertex_map = vec![usize::MAX; adj.len()];
    for (i, &v) in vertices.iter().enumerate() {
        vertex_map[v] = i;
    }
    let m = vertices.len();
    let mut sub_adj: Vec<Vec<usize>> = vec![Vec::new(); m];
    for (i, &v) in vertices.iter().enumerate() {
        for &u in &adj[v] {
            if let Some(j) = vertex_map.get(u).copied() {
                if j != usize::MAX && j != i {
                    sub_adj[i].push(j);
                }
            }
        }
    }
    let mut part = greedy_graph_growing(&sub_adj, m / 2);
    kernighan_lin_refine(&sub_adj, &mut part, 10);
    part
}
/// Greedy graph growing to create initial bisection.
///
/// Grows partition from a seed vertex using BFS until target size reached.
fn greedy_graph_growing(adj: &[Vec<usize>], target_size: usize) -> Vec<usize> {
    let n = adj.len();
    if n == 0 {
        return Vec::new();
    }
    let mut part = vec![1; n];
    let mut visited = vec![false; n];
    let seed = find_pseudo_peripheral_internal(adj);
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(seed);
    visited[seed] = true;
    part[seed] = 0;
    let mut size_0 = 1;
    while let Some(v) = queue.pop_front() {
        if size_0 >= target_size {
            break;
        }
        for &u in &adj[v] {
            if !visited[u] && size_0 < target_size {
                visited[u] = true;
                part[u] = 0;
                size_0 += 1;
                queue.push_back(u);
            }
        }
    }
    part
}
/// Find a pseudo-peripheral vertex using BFS on adjacency list.
fn find_pseudo_peripheral_internal(adj: &[Vec<usize>]) -> usize {
    let n = adj.len();
    if n == 0 {
        return 0;
    }
    let mut current = 0;
    let mut max_dist = 0;
    for _ in 0..5 {
        let (farthest, dist) = bfs_farthest(adj, current);
        if dist <= max_dist {
            break;
        }
        max_dist = dist;
        current = farthest;
    }
    current
}
/// BFS to find farthest vertex from start.
fn bfs_farthest(adj: &[Vec<usize>], start: usize) -> (usize, usize) {
    let n = adj.len();
    let mut dist = vec![usize::MAX; n];
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(start);
    dist[start] = 0;
    let mut farthest = start;
    let mut max_dist = 0;
    while let Some(v) = queue.pop_front() {
        for &u in &adj[v] {
            if dist[u] == usize::MAX {
                dist[u] = dist[v] + 1;
                queue.push_back(u);
                if dist[u] > max_dist {
                    max_dist = dist[u];
                    farthest = u;
                }
            }
        }
    }
    (farthest, max_dist)
}
/// Kernighan-Lin style refinement for bisection.
///
/// Iteratively swaps vertices between partitions to reduce edge cut.
fn kernighan_lin_refine(adj: &[Vec<usize>], part: &mut [usize], max_passes: usize) {
    let n = adj.len();
    if n <= 1 {
        return;
    }
    for _ in 0..max_passes {
        let initial_cut = compute_edge_cut_subgraph(adj, part);
        let mut gains = vec![0i32; n];
        for v in 0..n {
            gains[v] = compute_gain(adj, part, v);
        }
        let mut locked = vec![false; n];
        let mut improved = false;
        for _step in 0..n.min(n / 2) {
            let mut best_v = None;
            let mut best_gain = i32::MIN;
            for v in 0..n {
                if !locked[v] && gains[v] > best_gain {
                    best_gain = gains[v];
                    best_v = Some(v);
                }
            }
            if let Some(v) = best_v {
                if best_gain > 0 {
                    part[v] = 1 - part[v];
                    locked[v] = true;
                    improved = true;
                    for &u in &adj[v] {
                        if !locked[u] {
                            gains[u] = compute_gain(adj, part, u);
                        }
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        let final_cut = compute_edge_cut_subgraph(adj, part);
        if !improved || final_cut >= initial_cut {
            break;
        }
    }
}
/// Compute gain for moving vertex v to the other partition.
fn compute_gain(adj: &[Vec<usize>], part: &[usize], v: usize) -> i32 {
    let my_part = part[v];
    let mut internal = 0i32;
    let mut external = 0i32;
    for &u in &adj[v] {
        if part[u] == my_part {
            internal += 1;
        } else {
            external += 1;
        }
    }
    external - internal
}
/// Compute edge cut for subgraph partition.
fn compute_edge_cut_subgraph(adj: &[Vec<usize>], part: &[usize]) -> usize {
    let mut cut = 0;
    for (v, neighbors) in adj.iter().enumerate() {
        for &u in neighbors {
            if v < u && part[v] != part[u] {
                cut += 1;
            }
        }
    }
    cut
}
/// Compute edge cut for full graph partition.
fn compute_edge_cut(adj: &[Vec<usize>], partition: &[usize]) -> usize {
    let mut cut = 0;
    for (v, neighbors) in adj.iter().enumerate() {
        for &u in neighbors {
            if v < u && partition[v] != partition[u] {
                cut += 1;
            }
        }
    }
    cut
}
#[cfg(test)]
mod tests {
    use super::*;
    fn make_test_matrix() -> CsrMatrix<f64> {
        let values = vec![
            2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
        ];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4];
        let row_ptrs = vec![0, 2, 5, 8, 11, 13];
        CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap()
    }
    fn make_disconnected_matrix() -> CsrMatrix<f64> {
        let values = vec![2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0];
        let col_indices = vec![0, 1, 0, 1, 2, 1, 2, 3, 4, 3, 4];
        let row_ptrs = vec![0, 2, 5, 7, 9, 11];
        CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap()
    }
    #[test]
    fn test_connected_components_single() {
        let a = make_test_matrix();
        let result = connected_components(&a);
        assert_eq!(result.num_components, 1);
        assert_eq!(result.component_sizes, vec![5]);
        assert!(result.labels.iter().all(|&l| l == 0));
    }
    #[test]
    fn test_connected_components_multiple() {
        let a = make_disconnected_matrix();
        let result = connected_components(&a);
        assert_eq!(result.num_components, 2);
        assert_eq!(result.component_sizes.iter().sum::<usize>(), 5);
        assert_eq!(result.labels[0], result.labels[1]);
        assert_eq!(result.labels[1], result.labels[2]);
        assert_eq!(result.labels[3], result.labels[4]);
        assert_ne!(result.labels[0], result.labels[3]);
    }
    #[test]
    fn test_connected_components_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::new(0, 0, vec![0], vec![], vec![]).unwrap();
        let result = connected_components(&a);
        assert_eq!(result.num_components, 0);
        assert!(result.labels.is_empty());
    }
    #[test]
    fn test_bandwidth_profile_tridiagonal() {
        let a = make_test_matrix();
        let result = bandwidth_profile(&a);
        assert_eq!(result.bandwidth, 1);
        assert_eq!(result.lower_bandwidth, 1);
        assert_eq!(result.upper_bandwidth, 1);
    }
    #[test]
    fn test_bandwidth_profile_diagonal() {
        let values = vec![1.0, 2.0, 3.0];
        let col_indices = vec![0, 1, 2];
        let row_ptrs = vec![0, 1, 2, 3];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        let result = bandwidth_profile(&a);
        assert_eq!(result.bandwidth, 0);
        assert_eq!(result.profile, 0);
    }
    #[test]
    fn test_bandwidth_profile_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::new(0, 0, vec![0], vec![], vec![]).unwrap();
        let result = bandwidth_profile(&a);
        assert_eq!(result.bandwidth, 0);
        assert_eq!(result.profile, 0);
    }
    #[test]
    fn test_level_sets() {
        let a = make_test_matrix();
        let result = level_sets(&a, 0);
        assert_eq!(result.levels[0], 0);
        assert_eq!(result.levels[1], 1);
        assert_eq!(result.levels[2], 2);
        assert_eq!(result.levels[3], 3);
        assert_eq!(result.levels[4], 4);
        assert_eq!(result.max_level, 4);
    }
    #[test]
    fn test_level_sets_middle() {
        let a = make_test_matrix();
        let result = level_sets(&a, 2);
        assert_eq!(result.levels[2], 0);
        assert_eq!(result.max_level, 2);
    }
    #[test]
    fn test_pseudo_peripheral() {
        let a = make_test_matrix();
        let pp = pseudo_peripheral_vertex(&a);
        assert!(pp == 0 || pp == 4);
    }
    #[test]
    fn test_is_structurally_symmetric() {
        let a = make_test_matrix();
        assert!(is_structurally_symmetric(&a));
    }
    #[test]
    fn test_is_not_structurally_symmetric() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let col_indices = vec![0, 1, 2, 1, 2, 2];
        let row_ptrs = vec![0, 3, 5, 6];
        let a = CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap();
        assert!(!is_structurally_symmetric(&a));
    }
    #[test]
    fn test_degree_sequence() {
        let a = make_test_matrix();
        let degrees = degree_sequence(&a);
        assert_eq!(degrees[0], 1);
        assert_eq!(degrees[1], 2);
        assert_eq!(degrees[2], 2);
        assert_eq!(degrees[3], 2);
        assert_eq!(degrees[4], 1);
    }
    fn make_bipartite_graph() -> CsrMatrix<f64> {
        let values = vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ];
        let col_indices = vec![1, 3, 0, 2, 1, 3, 5, 0, 2, 4, 3, 5, 2, 4];
        let row_ptrs = vec![0, 2, 4, 7, 10, 12, 14];
        CsrMatrix::new(6, 6, row_ptrs, col_indices, values).unwrap()
    }
    fn make_non_bipartite_graph() -> CsrMatrix<f64> {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 2, 0, 2, 0, 1];
        let row_ptrs = vec![0, 2, 4, 6];
        CsrMatrix::new(3, 3, row_ptrs, col_indices, values).unwrap()
    }
    #[test]
    fn test_is_bipartite_true() {
        let a = make_bipartite_graph();
        let result = is_bipartite(&a);
        assert!(result.is_bipartite);
        assert_eq!(result.left.len() + result.right.len(), 6);
        for i in 0..6 {
            let start = a.row_ptrs()[i];
            let end = a.row_ptrs()[i + 1];
            for idx in start..end {
                let j = a.col_indices()[idx];
                assert_ne!(result.partition[i], result.partition[j]);
            }
        }
    }
    #[test]
    fn test_is_bipartite_false() {
        let a = make_non_bipartite_graph();
        let result = is_bipartite(&a);
        assert!(!result.is_bipartite);
    }
    #[test]
    fn test_is_bipartite_path_graph() {
        let a = make_test_matrix();
        let result = is_bipartite(&a);
        assert!(result.is_bipartite);
    }
    #[test]
    fn test_is_bipartite_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::new(0, 0, vec![0], vec![], vec![]).unwrap();
        let result = is_bipartite(&a);
        assert!(result.is_bipartite);
        assert!(result.left.is_empty());
        assert!(result.right.is_empty());
    }
    #[test]
    fn test_bipartite_matching_basic() {
        let a = make_bipartite_graph();
        let result = bipartite_matching(&a);
        assert!(result.is_some());
        let matching = result.unwrap();
        assert_eq!(matching.matching_size, 3);
        assert!(matching.is_perfect);
        for &(l, r) in &matching.edges {
            assert_eq!(matching.left_match[l], Some(r));
            assert_eq!(matching.right_match[r], Some(l));
        }
    }
    #[test]
    fn test_bipartite_matching_non_bipartite() {
        let a = make_non_bipartite_graph();
        let result = bipartite_matching(&a);
        assert!(result.is_none());
    }
    #[test]
    fn test_bipartite_matching_empty() {
        let a: CsrMatrix<f64> = CsrMatrix::new(0, 0, vec![0], vec![], vec![]).unwrap();
        let result = bipartite_matching(&a);
        assert!(result.is_some());
        let matching = result.unwrap();
        assert_eq!(matching.matching_size, 0);
        assert!(matching.is_perfect);
    }
    #[test]
    fn test_bipartite_matching_path_graph() {
        let a = make_test_matrix();
        let result = bipartite_matching(&a);
        assert!(result.is_some());
        let matching = result.unwrap();
        assert_eq!(matching.matching_size, 2);
    }
    #[test]
    fn test_weighted_matching_basic() {
        let values = vec![3.0, 2.0, 3.0, 1.0, 1.0, 4.0, 2.0, 4.0];
        let col_indices = vec![1, 3, 0, 2, 1, 3, 0, 2];
        let row_ptrs = vec![0, 2, 4, 6, 8];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let result = weighted_bipartite_matching(&a);
        assert!(result.is_some());
        let matching = result.unwrap();
        assert_eq!(matching.matching_size, 2);
        assert!(matching.total_weight >= 3.0);
    }
    #[test]
    fn test_weighted_matching_non_bipartite() {
        let a = make_non_bipartite_graph();
        let result = weighted_bipartite_matching(&a);
        assert!(result.is_none());
    }
    #[test]
    fn test_bipartite_matching_complete_bipartite() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![2, 3, 4, 2, 3, 4, 0, 1, 0, 1, 0, 1];
        let row_ptrs = vec![0, 3, 6, 8, 10, 12];
        let a = CsrMatrix::new(5, 5, row_ptrs, col_indices, values).unwrap();
        let result = bipartite_matching(&a);
        assert!(result.is_some());
        let matching = result.unwrap();
        assert_eq!(matching.matching_size, 2);
    }
    #[test]
    fn test_bipartite_disconnected_components() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 0, 3, 2];
        let row_ptrs = vec![0, 1, 2, 3, 4];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let result = bipartite_matching(&a);
        assert!(result.is_some());
        let matching = result.unwrap();
        assert_eq!(matching.matching_size, 2);
        assert!(matching.is_perfect);
    }
    #[test]
    fn test_partition_bisect_basic() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 0, 2, 1, 3, 2];
        let row_ptrs = vec![0, 1, 3, 5, 6];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let result = partition_graph_bisect(&a);
        assert_eq!(result.num_partitions, 2);
        assert_eq!(result.partition.len(), 4);
        assert_eq!(result.partition_sizes.len(), 2);
        let total_size: usize = result.partition_sizes.iter().sum();
        assert_eq!(total_size, 4);
        assert!(result.edge_cut <= 2, "Edge cut should be small");
    }
    #[test]
    fn test_partition_kway_four_parts() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 2, 0, 3, 0, 3, 1, 2];
        let row_ptrs = vec![0, 2, 4, 6, 8];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let result = partition_graph_kway(&a, 4);
        assert_eq!(result.num_partitions, 4);
        assert_eq!(result.partition.len(), 4);
        let mut used_partitions: Vec<usize> = result.partition.clone();
        used_partitions.sort();
        used_partitions.dedup();
        assert!(used_partitions.len() >= 2, "Should use multiple partitions");
        let total_size: usize = result.partition_sizes.iter().sum();
        assert_eq!(total_size, 4);
    }
    #[test]
    fn test_partition_single_partition() {
        let a = make_test_matrix();
        let result = partition_graph_kway(&a, 1);
        assert_eq!(result.num_partitions, 1);
        assert_eq!(result.partition, vec![0; a.nrows()]);
        assert_eq!(result.edge_cut, 0);
    }
    #[test]
    fn test_partition_empty_matrix() {
        let values: Vec<f64> = vec![];
        let col_indices: Vec<usize> = vec![];
        let row_ptrs = vec![0];
        let a = CsrMatrix::new(0, 0, row_ptrs, col_indices, values).unwrap();
        let result = partition_graph_bisect(&a);
        assert_eq!(result.num_partitions, 0);
        assert!(result.partition.is_empty());
    }
    #[test]
    fn test_partition_disconnected_components() {
        let values = vec![1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 0, 3, 2];
        let row_ptrs = vec![0, 1, 2, 3, 4];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let result = partition_graph_bisect(&a);
        assert_eq!(result.num_partitions, 2);
        assert_eq!(result.partition.len(), 4);
        let partition0_count = result.partition.iter().filter(|&&p| p == 0).count();
        let partition1_count = result.partition.iter().filter(|&&p| p == 1).count();
        assert!(partition0_count >= 1);
        assert!(partition1_count >= 1);
    }
    #[test]
    fn test_partition_star_graph() {
        let values = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let col_indices = vec![1, 2, 3, 0, 0, 0];
        let row_ptrs = vec![0, 3, 4, 5, 6];
        let a = CsrMatrix::new(4, 4, row_ptrs, col_indices, values).unwrap();
        let result = partition_graph_bisect(&a);
        assert_eq!(result.num_partitions, 2);
        assert_eq!(result.partition.len(), 4);
        let part0_count = result.partition.iter().filter(|&&p| p == 0).count();
        let part1_count = result.partition.iter().filter(|&&p| p == 1).count();
        assert!(part0_count >= 1 || part1_count >= 1);
    }
    #[test]
    fn test_partition_result_validity() {
        let a = make_test_matrix();
        let result = partition_graph_kway(&a, 2);
        for &p in &result.partition {
            assert!(p < result.num_partitions);
        }
        let total: usize = result.partition_sizes.iter().sum();
        assert_eq!(total, a.nrows());
        for k in 0..result.num_partitions {
            let count = result.partition.iter().filter(|&&p| p == k).count();
            assert_eq!(count, result.partition_sizes[k]);
        }
    }
}
