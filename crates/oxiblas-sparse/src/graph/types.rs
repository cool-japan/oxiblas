//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

/// Result of bipartite check.
#[derive(Debug, Clone)]
pub struct BipartiteResult {
    /// Whether the graph is bipartite.
    pub is_bipartite: bool,
    /// Partition assignment for each vertex (0 or 1).
    /// Only valid if `is_bipartite` is true.
    pub partition: Vec<usize>,
    /// Vertices in partition 0.
    pub left: Vec<usize>,
    /// Vertices in partition 1.
    pub right: Vec<usize>,
}
/// Result of graph partitioning.
#[derive(Debug, Clone)]
pub struct PartitionResult {
    /// Partition assignment for each vertex (0 to k-1 for k-way partitioning).
    pub partition: Vec<usize>,
    /// Number of partitions.
    pub num_partitions: usize,
    /// Size of each partition (number of vertices).
    pub partition_sizes: Vec<usize>,
    /// Edge cut size (number of edges crossing partitions).
    pub edge_cut: usize,
}
/// Result of bandwidth and profile analysis.
#[derive(Debug, Clone)]
pub struct BandwidthProfileResult {
    /// Bandwidth: max |i - j| for all non-zero a_ij.
    pub bandwidth: usize,
    /// Lower bandwidth: max (i - j) for all non-zero a_ij where i > j.
    pub lower_bandwidth: usize,
    /// Upper bandwidth: max (j - i) for all non-zero a_ij where j > i.
    pub upper_bandwidth: usize,
    /// Profile (envelope): sum of row bandwidths.
    pub profile: usize,
    /// Average bandwidth per row.
    pub average_bandwidth: f64,
}
/// Result of level set construction.
#[derive(Debug, Clone)]
pub struct LevelSetResult {
    /// Level (distance from root) for each vertex.
    pub levels: Vec<usize>,
    /// List of vertices at each level.
    pub level_sets: Vec<Vec<usize>>,
    /// Maximum level (eccentricity of root).
    pub max_level: usize,
}
/// Result of bipartite matching.
#[derive(Debug, Clone)]
pub struct BipartiteMatchingResult {
    /// For each vertex in left partition, the matched vertex in right partition.
    /// `None` if the vertex is unmatched.
    pub left_match: Vec<Option<usize>>,
    /// For each vertex in right partition, the matched vertex in left partition.
    /// `None` if the vertex is unmatched.
    pub right_match: Vec<Option<usize>>,
    /// Size of the maximum matching (number of edges).
    pub matching_size: usize,
    /// List of edges in the matching as (left_vertex, right_vertex) pairs.
    pub edges: Vec<(usize, usize)>,
    /// Whether a perfect matching exists (all vertices matched).
    pub is_perfect: bool,
}
/// Result of weighted bipartite matching.
#[derive(Debug, Clone)]
pub struct WeightedMatchingResult<T> {
    /// For each vertex in left partition, the matched vertex in right partition.
    pub left_match: Vec<Option<usize>>,
    /// For each vertex in right partition, the matched vertex in left partition.
    pub right_match: Vec<Option<usize>>,
    /// Size of the maximum matching.
    pub matching_size: usize,
    /// List of edges in the matching.
    pub edges: Vec<(usize, usize)>,
    /// Total weight of the matching.
    pub total_weight: T,
}
/// Result of connected components analysis.
#[derive(Debug, Clone)]
pub struct ConnectedComponentsResult {
    /// Component label for each vertex (0-indexed).
    pub labels: Vec<usize>,
    /// Number of connected components.
    pub num_components: usize,
    /// Size of each component.
    pub component_sizes: Vec<usize>,
}
