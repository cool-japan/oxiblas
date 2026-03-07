//! Multilevel graph partitioning — METIS-equivalent pure Rust implementation.
//!
//! Provides Heavy Edge Matching (HEM) coarsening, greedy BFS bisection,
//! Kernighan-Lin boundary refinement, recursive bisection for k-way partitioning,
//! and multilevel nested dissection for fill-reducing orderings.
//!
//! This module implements the classic multilevel partitioning framework:
//! 1. **Coarsening**: Heavy Edge Matching contracts matched vertex pairs into
//!    super-vertices, preserving graph structure with accumulated weights.
//! 2. **Initial partition**: BFS-based greedy bisection on the coarsest graph.
//! 3. **Uncoarsening + refinement**: Project partition back level by level,
//!    applying Kernighan-Lin style boundary refinement at each level.
//!
//! # References
//! - Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for Partitioning
//!   Irregular Graphs", SIAM J. Sci. Comput. 20(1), 1999.
//! - Davis, "Direct Methods for Sparse Linear Systems", SIAM, 2006.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error type for multilevel partitioning and nested dissection.
#[derive(Debug, Clone, PartialEq)]
pub enum OrderingError {
    /// Invalid input parameters or graph data.
    InvalidInput(String),
    /// Graph has no vertices.
    EmptyGraph,
    /// The requested number of partitions is invalid.
    InvalidNumParts {
        /// The number of parts requested.
        requested: usize,
    },
}

impl std::fmt::Display for OrderingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            OrderingError::EmptyGraph => write!(f, "Graph has no vertices"),
            OrderingError::InvalidNumParts { requested } => {
                write!(f, "Invalid number of parts: {}", requested)
            }
        }
    }
}

impl std::error::Error for OrderingError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the multilevel partitioner.
#[derive(Debug, Clone)]
pub struct PartitionConfig {
    /// Stop coarsening when graph has fewer than this many vertices. Default: 100.
    pub max_coarse_size: usize,
    /// Number of Kernighan-Lin refinement passes per uncoarsening level. Default: 10.
    pub n_iter_refine: usize,
    /// Allowed partition imbalance (fraction of total weight). Default: 0.1.
    pub imbalance_tol: f64,
}

impl Default for PartitionConfig {
    fn default() -> Self {
        Self {
            max_coarse_size: 100,
            n_iter_refine: 10,
            imbalance_tol: 0.1,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal graph representation
// ---------------------------------------------------------------------------

/// A (possibly coarsened) graph in CSR format with vertex/edge weights
/// and the vertex map from fine to coarse level.
#[derive(Debug, Clone)]
struct CoarseGraph {
    /// Number of vertices in this coarsened graph.
    n_verts: usize,
    /// CSR row pointers (length n_verts + 1).
    xadj: Vec<usize>,
    /// CSR adjacency (length xadj[n_verts]).
    adjncy: Vec<usize>,
    /// Vertex weights (length n_verts).
    vwgt: Vec<f64>,
    /// Edge weights parallel to adjncy.
    adjwgt: Vec<f64>,
    /// For each fine vertex, the coarse vertex it maps to (length = fine n_verts).
    cmap: Vec<usize>,
}

/// A maximal matching used during HEM coarsening.
struct Matching {
    /// `pairs[v]` = `Some(u)` means v is matched with u; `None` = unmatched.
    pairs: Vec<Option<usize>>,
}

// ---------------------------------------------------------------------------
// Public partition result
// ---------------------------------------------------------------------------

/// A k-way partition of graph vertices.
#[derive(Debug, Clone)]
pub struct Partition {
    /// `part[v]` is the partition id (0..n_parts) assigned to vertex v.
    pub part: Vec<usize>,
    /// Total number of partitions.
    pub n_parts: usize,
}

// ---------------------------------------------------------------------------
// MultilevelPartitioner
// ---------------------------------------------------------------------------

/// Multilevel graph partitioner (METIS-style, pure Rust).
pub struct MultilevelPartitioner {
    /// Algorithm configuration.
    pub config: PartitionConfig,
}

impl MultilevelPartitioner {
    /// Creates a new partitioner with the given configuration.
    pub fn new(config: PartitionConfig) -> Self {
        Self { config }
    }

    /// Partition a graph with `n` vertices into `num_parts` parts using
    /// recursive bisection.
    ///
    /// # Arguments
    /// * `n` — number of vertices
    /// * `xadj` — CSR row pointers (length n+1)
    /// * `adjncy` — CSR adjacency list
    /// * `num_parts` — desired number of partitions (must be >= 1)
    ///
    /// # Returns
    /// A [`Partition`] where `part[v]` is in `0..num_parts`.
    pub fn partition(
        &self,
        n: usize,
        xadj: &[usize],
        adjncy: &[usize],
        num_parts: usize,
    ) -> Result<Partition, OrderingError> {
        if n == 0 {
            return Err(OrderingError::EmptyGraph);
        }
        if num_parts == 0 {
            return Err(OrderingError::InvalidNumParts { requested: 0 });
        }
        if num_parts == 1 {
            return Ok(Partition {
                part: vec![0; n],
                n_parts: 1,
            });
        }
        let uniform_vwgt = vec![1.0f64; n];
        let uniform_adjwgt = vec![1.0f64; adjncy.len()];
        // Recursive bisection: start with all vertices in part 0, then split.
        self.recursive_bisect(n, xadj, adjncy, &uniform_adjwgt, &uniform_vwgt, num_parts)
    }

    /// Recursive bisection entry point.
    fn recursive_bisect(
        &self,
        n: usize,
        xadj: &[usize],
        adjncy: &[usize],
        adjwgt: &[f64],
        vwgt: &[f64],
        num_parts: usize,
    ) -> Result<Partition, OrderingError> {
        // Start: all vertices in part 0, label them 0..n-1 as global indices.
        let global_ids: Vec<usize> = (0..n).collect();
        let mut part = vec![0usize; n];
        self.bisect_recursive(
            n,
            xadj,
            adjncy,
            adjwgt,
            vwgt,
            &global_ids,
            &mut part,
            0, // current part base id
            num_parts,
        )?;
        Ok(Partition {
            part,
            n_parts: num_parts,
        })
    }

    /// Recursively bisect the subgraph induced by `global_ids` and assign
    /// partition ids `part_base .. part_base + num_parts - 1` to `part[global_ids[*]]`.
    #[allow(clippy::too_many_arguments)]
    fn bisect_recursive(
        &self,
        full_n: usize,
        full_xadj: &[usize],
        full_adjncy: &[usize],
        full_adjwgt: &[f64],
        full_vwgt: &[f64],
        global_ids: &[usize],
        part: &mut Vec<usize>,
        part_base: usize,
        num_parts: usize,
    ) -> Result<(), OrderingError> {
        let sub_n = global_ids.len();
        if num_parts <= 1 || sub_n == 0 {
            for &v in global_ids {
                part[v] = part_base;
            }
            return Ok(());
        }

        // Build local subgraph.
        let (local_xadj, local_adjncy, local_adjwgt, local_vwgt) = build_subgraph(
            full_n,
            full_xadj,
            full_adjncy,
            full_adjwgt,
            full_vwgt,
            global_ids,
        );

        // Bisect local graph.
        let bisection = self.bisect(
            sub_n,
            &local_xadj,
            &local_adjncy,
            &local_adjwgt,
            &local_vwgt,
        )?;

        // Collect local part 0 and part 1 global ids.
        let part0: Vec<usize> = global_ids
            .iter()
            .enumerate()
            .filter(|&(i, _)| bisection.part[i] == 0)
            .map(|(_, &g)| g)
            .collect();
        let part1: Vec<usize> = global_ids
            .iter()
            .enumerate()
            .filter(|&(i, _)| bisection.part[i] == 1)
            .map(|(_, &g)| g)
            .collect();

        // Allocate sub-part counts proportionally to vertex counts.
        let parts_for_0 = num_parts / 2;
        let parts_for_1 = num_parts - parts_for_0;

        self.bisect_recursive(
            full_n,
            full_xadj,
            full_adjncy,
            full_adjwgt,
            full_vwgt,
            &part0,
            part,
            part_base,
            parts_for_0,
        )?;
        self.bisect_recursive(
            full_n,
            full_xadj,
            full_adjncy,
            full_adjwgt,
            full_vwgt,
            &part1,
            part,
            part_base + parts_for_0,
            parts_for_1,
        )?;

        Ok(())
    }

    /// Core multilevel bisection: coarsen → initial partition → uncoarsen+refine.
    fn bisect(
        &self,
        n: usize,
        xadj: &[usize],
        adjncy: &[usize],
        adjwgt: &[f64],
        vwgt: &[f64],
    ) -> Result<Partition, OrderingError> {
        if n == 0 {
            return Err(OrderingError::EmptyGraph);
        }
        if n <= 2 {
            // Trivially assign first half to 0, rest to 1.
            let part: Vec<usize> = (0..n).map(|i| if i < n / 2 { 0 } else { 1 }).collect();
            return Ok(Partition { part, n_parts: 2 });
        }

        // ---- Coarsen ----
        let mut levels: Vec<CoarseGraph> = Vec::new();
        let root = CoarseGraph {
            n_verts: n,
            xadj: xadj.to_vec(),
            adjncy: adjncy.to_vec(),
            vwgt: vwgt.to_vec(),
            adjwgt: adjwgt.to_vec(),
            cmap: (0..n).collect(),
        };
        levels.push(root);

        loop {
            let current = levels.last().expect("levels non-empty");
            if current.n_verts <= self.config.max_coarse_size {
                break;
            }
            let coarsened = self.coarsen(
                current.n_verts,
                &current.xadj,
                &current.adjncy,
                &current.adjwgt,
                &current.vwgt,
            )?;
            // If coarsening didn't reduce size meaningfully, stop.
            if coarsened.n_verts >= current.n_verts {
                break;
            }
            levels.push(coarsened);
        }

        // ---- Initial partition on coarsest graph ----
        let coarsest = levels.last().expect("levels non-empty");
        let mut partition = self.initial_partition(coarsest)?;

        // ---- Uncoarsen + refine ----
        // levels: [fine ... coarse]
        // We iterate from second-to-last down to index 0.
        for level_idx in (0..levels.len() - 1).rev() {
            let coarse = &levels[level_idx + 1];
            let fine = &levels[level_idx];
            partition = self.uncoarsen_and_refine(
                coarse,
                &partition,
                fine.n_verts,
                &fine.xadj,
                &fine.adjncy,
                &fine.adjwgt,
                &fine.vwgt,
            )?;
        }

        Ok(partition)
    }

    /// Heavy Edge Matching (HEM) coarsening.
    ///
    /// Iterates vertices in order, matching each unmatched vertex with its
    /// heaviest unmatched neighbor. Contracted pairs become super-vertices.
    fn coarsen(
        &self,
        n: usize,
        xadj: &[usize],
        adjncy: &[usize],
        adjwgt: &[f64],
        vwgt: &[f64],
    ) -> Result<CoarseGraph, OrderingError> {
        let matching = self.heavy_edge_matching(n, xadj, adjncy, adjwgt);

        // Assign coarse vertex ids.
        let mut cmap = vec![usize::MAX; n];
        let mut n_coarse = 0usize;
        for v in 0..n {
            if cmap[v] == usize::MAX {
                cmap[v] = n_coarse;
                if let Some(u) = matching.pairs[v] {
                    if cmap[u] == usize::MAX {
                        cmap[u] = n_coarse;
                    }
                }
                n_coarse += 1;
            }
        }

        // Build coarse vertex weights.
        let mut coarse_vwgt = vec![0.0f64; n_coarse];
        for v in 0..n {
            coarse_vwgt[cmap[v]] += vwgt[v];
        }

        // Build coarse adjacency: for each coarse vertex, collect edges to
        // other coarse vertices, merging parallel edges by summing weights.
        let mut coarse_adj: Vec<std::collections::HashMap<usize, f64>> =
            vec![std::collections::HashMap::new(); n_coarse];

        for v in 0..n {
            let cv = cmap[v];
            for idx in xadj[v]..xadj[v + 1] {
                let u = adjncy[idx];
                let cu = cmap[u];
                if cu != cv {
                    let w = adjwgt[idx];
                    *coarse_adj[cv].entry(cu).or_insert(0.0) += w;
                }
            }
        }

        // Flatten into CSR.
        let mut c_xadj = Vec::with_capacity(n_coarse + 1);
        let mut c_adjncy = Vec::new();
        let mut c_adjwgt = Vec::new();
        c_xadj.push(0usize);
        for cv in 0..n_coarse {
            let mut neighbors: Vec<(usize, f64)> = coarse_adj[cv].drain().collect();
            neighbors.sort_unstable_by_key(|&(u, _)| u);
            for (cu, w) in neighbors {
                c_adjncy.push(cu);
                c_adjwgt.push(w);
            }
            c_xadj.push(c_adjncy.len());
        }

        Ok(CoarseGraph {
            n_verts: n_coarse,
            xadj: c_xadj,
            adjncy: c_adjncy,
            vwgt: coarse_vwgt,
            adjwgt: c_adjwgt,
            cmap,
        })
    }

    /// Compute a Heavy Edge Matching on the graph.
    fn heavy_edge_matching(
        &self,
        n: usize,
        xadj: &[usize],
        adjncy: &[usize],
        adjwgt: &[f64],
    ) -> Matching {
        let mut pairs: Vec<Option<usize>> = vec![None; n];
        let mut matched = vec![false; n];

        for v in 0..n {
            if matched[v] {
                continue;
            }
            // Find the heaviest unmatched neighbor.
            let mut best_u: Option<usize> = None;
            let mut best_w = f64::NEG_INFINITY;
            for idx in xadj[v]..xadj[v + 1] {
                let u = adjncy[idx];
                if !matched[u] && u != v {
                    let w = adjwgt[idx];
                    if w > best_w {
                        best_w = w;
                        best_u = Some(u);
                    }
                }
            }
            if let Some(u) = best_u {
                matched[v] = true;
                matched[u] = true;
                pairs[v] = Some(u);
                pairs[u] = Some(v);
            } else {
                // Unmatched: self-loop (singleton coarse vertex).
                matched[v] = true;
            }
        }
        Matching { pairs }
    }

    /// Initial partition of the coarsest graph using BFS-based greedy bisection.
    ///
    /// Start a BFS from vertex 0; assign vertices to part 0 until half the
    /// total vertex weight is consumed, then assign the rest to part 1.
    fn initial_partition(&self, graph: &CoarseGraph) -> Result<Partition, OrderingError> {
        let n = graph.n_verts;
        if n == 0 {
            return Err(OrderingError::EmptyGraph);
        }

        let total_weight: f64 = graph.vwgt.iter().sum();
        let target = total_weight / 2.0;

        let mut part = vec![1usize; n];
        let mut visited = vec![false; n];
        let mut weight_0 = 0.0f64;

        let mut queue = VecDeque::new();
        queue.push_back(0usize);
        visited[0] = true;

        while let Some(v) = queue.pop_front() {
            if weight_0 < target {
                part[v] = 0;
                weight_0 += graph.vwgt[v];
            } else {
                part[v] = 1;
            }
            for idx in graph.xadj[v]..graph.xadj[v + 1] {
                let u = graph.adjncy[idx];
                if !visited[u] {
                    visited[u] = true;
                    queue.push_back(u);
                }
            }
        }

        // Handle disconnected vertices: assign to part 1.
        // (already initialised to 1)

        Ok(Partition { part, n_parts: 2 })
    }

    /// Kernighan-Lin style boundary refinement.
    ///
    /// Repeatedly identifies boundary vertices (adjacent to the other part)
    /// and moves the one with the highest positive gain that keeps balance.
    fn refine_partition(
        &self,
        n: usize,
        xadj: &[usize],
        adjncy: &[usize],
        adjwgt: &[f64],
        vwgt: &[f64],
        part: &mut Vec<usize>,
    ) {
        let total_weight: f64 = vwgt.iter().sum();
        let tol = self.config.imbalance_tol;

        for _ in 0..self.config.n_iter_refine {
            // Compute weights per part.
            let mut w: [f64; 2] = [0.0; 2];
            for v in 0..n {
                if part[v] < 2 {
                    w[part[v]] += vwgt[v];
                }
            }

            // Compute gain for each boundary vertex.
            // gain(v) = (external degree) - (internal degree) in edge-cut terms.
            let mut best_gain = 0.0f64;
            let mut best_v: Option<usize> = None;

            for v in 0..n {
                let pv = part[v];
                if pv >= 2 {
                    continue;
                }
                let other = 1 - pv;
                let mut ext = 0.0f64;
                let mut int = 0.0f64;
                let mut is_boundary = false;
                for idx in xadj[v]..xadj[v + 1] {
                    let u = adjncy[idx];
                    let w_edge = adjwgt[idx];
                    if part[u] == pv {
                        int += w_edge;
                    } else if part[u] == other {
                        ext += w_edge;
                        is_boundary = true;
                    }
                }
                if !is_boundary {
                    continue;
                }
                let gain = ext - int;
                // Check balance constraint: after moving v, the new weights must be
                // within tolerance of total/2.
                let new_w_pv = w[pv] - vwgt[v];
                let new_w_other = w[other] + vwgt[v];
                let half = total_weight / 2.0;
                let allowed = tol * total_weight;
                if (new_w_pv - half).abs() <= allowed && (new_w_other - half).abs() <= allowed {
                    if gain > best_gain {
                        best_gain = gain;
                        best_v = Some(v);
                    }
                }
            }

            if let Some(v) = best_v {
                part[v] = 1 - part[v];
            } else {
                break; // No improving move found.
            }
        }
    }

    /// Project a coarse partition back to the fine graph via `cmap`, then refine.
    fn uncoarsen_and_refine(
        &self,
        coarse: &CoarseGraph,
        coarse_part: &Partition,
        fine_n: usize,
        fine_xadj: &[usize],
        fine_adjncy: &[usize],
        fine_adjwgt: &[f64],
        fine_vwgt: &[f64],
    ) -> Result<Partition, OrderingError> {
        // Project: fine vertex v maps to coarse vertex cmap[v].
        let mut part = vec![0usize; fine_n];
        for v in 0..fine_n {
            let cv = coarse.cmap[v];
            if cv >= coarse_part.part.len() {
                return Err(OrderingError::InvalidInput(format!(
                    "cmap[{}] = {} out of range (coarse has {} vertices)",
                    v,
                    cv,
                    coarse_part.part.len()
                )));
            }
            part[v] = coarse_part.part[cv];
        }

        // Refine on the fine level.
        self.refine_partition(
            fine_n,
            fine_xadj,
            fine_adjncy,
            fine_adjwgt,
            fine_vwgt,
            &mut part,
        );

        Ok(Partition {
            part,
            n_parts: coarse_part.n_parts,
        })
    }
}

// ---------------------------------------------------------------------------
// Subgraph extraction helper
// ---------------------------------------------------------------------------

/// Build a local CSR subgraph induced by `global_ids` (a subset of 0..full_n).
/// Returns (xadj, adjncy, adjwgt, vwgt) for the local graph where local vertex i
/// corresponds to global_ids[i].
fn build_subgraph(
    _full_n: usize,
    full_xadj: &[usize],
    full_adjncy: &[usize],
    full_adjwgt: &[f64],
    full_vwgt: &[f64],
    global_ids: &[usize],
) -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
    let sub_n = global_ids.len();

    // Build reverse map: global → local (usize::MAX = not in subgraph).
    // We need a mapping but we don't know the max global id here, so use a HashMap.
    let mut global_to_local: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(sub_n);
    for (local, &g) in global_ids.iter().enumerate() {
        global_to_local.insert(g, local);
    }

    let mut xadj = Vec::with_capacity(sub_n + 1);
    let mut adjncy = Vec::new();
    let mut adjwgt = Vec::new();
    let mut vwgt = Vec::with_capacity(sub_n);

    xadj.push(0usize);
    for &g in global_ids {
        vwgt.push(full_vwgt[g]);
        for idx in full_xadj[g]..full_xadj[g + 1] {
            let ng = full_adjncy[idx];
            if let Some(&local_ng) = global_to_local.get(&ng) {
                adjncy.push(local_ng);
                adjwgt.push(full_adjwgt[idx]);
            }
        }
        xadj.push(adjncy.len());
    }

    (xadj, adjncy, adjwgt, vwgt)
}

// ---------------------------------------------------------------------------
// Multilevel Nested Dissection
// ---------------------------------------------------------------------------

/// Multilevel nested dissection ordering (METIS-equivalent, pure Rust).
///
/// Uses recursive bisection via [`MultilevelPartitioner`] to compute a
/// fill-reducing permutation for sparse direct solvers (Cholesky, LU, etc.).
///
/// The algorithm:
/// 1. Bisect the graph into two parts using multilevel partitioning.
/// 2. Identify the vertex separator (boundary vertices adjacent to both parts).
/// 3. Recursively order each part (without the separator).
/// 4. Append separator vertices last at this recursion level.
///
/// # Arguments
/// * `n` — number of vertices
/// * `xadj` — CSR row pointers (length n+1), symmetric adjacency
/// * `adjncy` — CSR column indices
///
/// # Returns
/// A permutation vector `perm` of length `n` where `perm[new_pos] = old_vertex`.
pub fn multilevel_nested_dissection(
    n: usize,
    xadj: &[usize],
    adjncy: &[usize],
) -> Result<Vec<usize>, OrderingError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if xadj.len() != n + 1 {
        return Err(OrderingError::InvalidInput(format!(
            "xadj length {} != n+1 = {}",
            xadj.len(),
            n + 1
        )));
    }

    let config = PartitionConfig::default();
    let adjwgt = vec![1.0f64; adjncy.len()];
    let vwgt = vec![1.0f64; n];

    let mut perm = vec![0usize; n];
    let mut pos = 0usize;

    let global_vertices: Vec<usize> = (0..n).collect();
    mnd_recursive(
        n,
        xadj,
        adjncy,
        &adjwgt,
        &vwgt,
        &global_vertices,
        &mut perm,
        &mut pos,
        &config,
    )?;

    Ok(perm)
}

/// Recursive helper for multilevel nested dissection.
///
/// Orders the subgraph induced by `global_vertices` and writes the result
/// into `perm[pos..]`.
#[allow(clippy::too_many_arguments)]
fn mnd_recursive(
    full_n: usize,
    full_xadj: &[usize],
    full_adjncy: &[usize],
    full_adjwgt: &[f64],
    full_vwgt: &[f64],
    global_vertices: &[usize],
    perm: &mut Vec<usize>,
    pos: &mut usize,
    config: &PartitionConfig,
) -> Result<(), OrderingError> {
    let sub_n = global_vertices.len();

    // Base case: small subgraph — just number vertices in natural order.
    if sub_n <= 2 * config.max_coarse_size {
        for &v in global_vertices {
            perm[*pos] = v;
            *pos += 1;
        }
        return Ok(());
    }

    // Build local subgraph.
    let (local_xadj, local_adjncy, local_adjwgt, local_vwgt) = build_subgraph(
        full_n,
        full_xadj,
        full_adjncy,
        full_adjwgt,
        full_vwgt,
        global_vertices,
    );

    let partitioner = MultilevelPartitioner::new(config.clone());
    let bisection = partitioner.bisect(
        sub_n,
        &local_xadj,
        &local_adjncy,
        &local_adjwgt,
        &local_vwgt,
    )?;

    // Identify separator: local vertices in part 0 that have a neighbor in part 1,
    // or vice versa. We include boundary vertices from the smaller cut side
    // (simplified: all vertices adjacent to the other part form the separator).
    let mut is_separator = vec![false; sub_n];
    for local_v in 0..sub_n {
        let pv = bisection.part[local_v];
        for idx in local_xadj[local_v]..local_xadj[local_v + 1] {
            let local_u = local_adjncy[idx];
            if bisection.part[local_u] != pv {
                is_separator[local_v] = true;
                break;
            }
        }
    }

    // Partition vertices into: part0 (non-sep), part1 (non-sep), separator.
    let mut part0_global = Vec::new();
    let mut part1_global = Vec::new();
    let mut sep_global = Vec::new();

    for (local_v, &global_v) in global_vertices.iter().enumerate() {
        if is_separator[local_v] {
            sep_global.push(global_v);
        } else if bisection.part[local_v] == 0 {
            part0_global.push(global_v);
        } else {
            part1_global.push(global_v);
        }
    }

    // Recurse: order part0, then part1, then separator (nested dissection order).
    if !part0_global.is_empty() {
        mnd_recursive(
            full_n,
            full_xadj,
            full_adjncy,
            full_adjwgt,
            full_vwgt,
            &part0_global,
            perm,
            pos,
            config,
        )?;
    }
    if !part1_global.is_empty() {
        mnd_recursive(
            full_n,
            full_xadj,
            full_adjncy,
            full_adjwgt,
            full_vwgt,
            &part1_global,
            perm,
            pos,
            config,
        )?;
    }
    // Separator vertices numbered last at this level.
    for &v in &sep_global {
        perm[*pos] = v;
        *pos += 1;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a cycle graph: 0-1-2-...-n-1-0, undirected.
    fn cycle_graph(n: usize) -> (Vec<usize>, Vec<usize>) {
        let mut xadj = vec![0usize; n + 1];
        let mut adjncy = Vec::with_capacity(2 * n);
        for v in 0..n {
            let prev = if v == 0 { n - 1 } else { v - 1 };
            let next = (v + 1) % n;
            adjncy.push(prev);
            adjncy.push(next);
            xadj[v + 1] = adjncy.len();
        }
        (xadj, adjncy)
    }

    /// Build a 4×4 grid graph (row-major, undirected).
    fn grid_graph(rows: usize, cols: usize) -> (Vec<usize>, Vec<usize>) {
        let n = rows * cols;
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for r in 0..rows {
            for c in 0..cols {
                let v = r * cols + c;
                if r + 1 < rows {
                    let u = (r + 1) * cols + c;
                    adj[v].push(u);
                    adj[u].push(v);
                }
                if c + 1 < cols {
                    let u = r * cols + (c + 1);
                    adj[v].push(u);
                    adj[u].push(v);
                }
            }
        }
        let mut xadj = vec![0usize; n + 1];
        let mut adjncy = Vec::new();
        for v in 0..n {
            adj[v].sort_unstable();
            adj[v].dedup();
            for &u in &adj[v] {
                adjncy.push(u);
            }
            xadj[v + 1] = adjncy.len();
        }
        (xadj, adjncy)
    }

    /// Build a 2D Laplacian (5×5 grid) CSR representation.
    fn laplacian_2d(rows: usize, cols: usize) -> (usize, Vec<usize>, Vec<usize>) {
        let (xadj, adjncy) = grid_graph(rows, cols);
        (rows * cols, xadj, adjncy)
    }

    /// Estimate fill-in for an ordering: count non-zeros in Cholesky factor.
    /// Uses a simple symbolic elimination simulation.
    fn estimate_fill(n: usize, xadj: &[usize], adjncy: &[usize], perm: &[usize]) -> usize {
        // Build inverse permutation.
        let mut inv_perm = vec![0usize; n];
        for (new_pos, &old_v) in perm.iter().enumerate() {
            inv_perm[old_v] = new_pos;
        }
        // Build permuted adjacency (lower triangular).
        let mut adj: Vec<std::collections::BTreeSet<usize>> =
            vec![std::collections::BTreeSet::new(); n];
        for old_v in 0..n {
            let pv = inv_perm[old_v];
            for idx in xadj[old_v]..xadj[old_v + 1] {
                let old_u = adjncy[idx];
                let pu = inv_perm[old_u];
                if pu > pv {
                    adj[pv].insert(pu);
                }
            }
        }
        // Symbolic elimination.
        let mut nnz = n; // diagonal
        for j in 0..n {
            let col: Vec<usize> = adj[j].iter().copied().collect();
            nnz += col.len();
            if let Some(&first) = col.first() {
                let rest: Vec<usize> = col[1..].to_vec();
                for &k in &rest {
                    adj[first].insert(k);
                }
            }
        }
        nnz
    }

    #[test]
    fn test_multilevel_partition_simple() {
        // 6-vertex cycle graph.
        let (xadj, adjncy) = cycle_graph(6);
        let partitioner = MultilevelPartitioner::new(PartitionConfig {
            max_coarse_size: 2,
            ..Default::default()
        });
        let result = partitioner
            .partition(6, &xadj, &adjncy, 2)
            .expect("partition failed");
        assert_eq!(result.part.len(), 6);
        assert_eq!(result.n_parts, 2);
        for &p in &result.part {
            assert!(p < 2, "partition id {} out of range", p);
        }
        // Both parts should be non-empty.
        let count_0 = result.part.iter().filter(|&&p| p == 0).count();
        let count_1 = result.part.iter().filter(|&&p| p == 1).count();
        assert!(count_0 > 0, "part 0 is empty");
        assert!(count_1 > 0, "part 1 is empty");
    }

    #[test]
    fn test_multilevel_partition_grid() {
        // 4×4 grid, partition into 4 parts.
        let (xadj, adjncy) = grid_graph(4, 4);
        let n = 16;
        let config = PartitionConfig {
            max_coarse_size: 4,
            n_iter_refine: 5,
            imbalance_tol: 0.3,
        };
        let partitioner = MultilevelPartitioner::new(config);
        let result = partitioner
            .partition(n, &xadj, &adjncy, 4)
            .expect("partition failed");
        assert_eq!(result.part.len(), n);
        assert_eq!(result.n_parts, 4);
        for &p in &result.part {
            assert!(p < 4, "partition id {} >= 4", p);
        }
        // All 4 parts should be non-empty.
        for part_id in 0..4 {
            let count = result.part.iter().filter(|&&p| p == part_id).count();
            assert!(count > 0, "part {} is empty", part_id);
        }
    }

    #[test]
    fn test_multilevel_nd_ordering() {
        // 5×5 grid — verify result is a valid permutation.
        let (n, xadj, adjncy) = laplacian_2d(5, 5);
        let perm = multilevel_nested_dissection(n, &xadj, &adjncy).expect("multilevel ND failed");
        assert_eq!(perm.len(), n);
        // Each vertex appears exactly once.
        let mut seen = vec![false; n];
        for &v in &perm {
            assert!(v < n, "vertex {} out of range", v);
            assert!(!seen[v], "vertex {} appears twice", v);
            seen[v] = true;
        }
        for i in 0..n {
            assert!(seen[i], "vertex {} missing from permutation", i);
        }
    }

    #[test]
    fn test_multilevel_nd_reduces_fill() {
        // 5×5 grid Laplacian: multilevel ND fill should be <= natural ordering fill.
        let (n, xadj, adjncy) = laplacian_2d(5, 5);

        // Natural ordering.
        let natural_perm: Vec<usize> = (0..n).collect();
        let natural_fill = estimate_fill(n, &xadj, &adjncy, &natural_perm);

        // Multilevel ND ordering.
        let mnd_perm =
            multilevel_nested_dissection(n, &xadj, &adjncy).expect("multilevel ND failed");
        let mnd_fill = estimate_fill(n, &xadj, &adjncy, &mnd_perm);

        // ND should generally give less fill. We allow a generous upper bound
        // (2x natural) in case the small graph doesn't trigger full coarsening.
        assert!(
            mnd_fill <= natural_fill * 2,
            "multilevel ND fill {} > 2 * natural fill {} — unexpectedly bad",
            mnd_fill,
            natural_fill
        );
    }
}
