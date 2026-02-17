// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

// SpillTree node.
struct SpillNode {
  int split_dim;        // -1 for leaf
  double split_val;     // median value on split_dim
  double left_bound;    // left child includes points <= split_val + tau_abs
  double right_bound;   // right child includes points >= split_val - tau_abs
  int left_child;       // index of left child (-1 if leaf)
  int right_child;      // index of right child (-1 if leaf)
  std::vector<int> point_indices;  // non-empty only for leaves
};

// Build the spill tree recursively.
static int build_spill_tree(
    const arma::mat& data,
    std::vector<int>& indices,
    double tau,
    int max_leaf_size,
    std::vector<SpillNode>& nodes
) {
  int n = static_cast<int>(indices.size());
  int idx = static_cast<int>(nodes.size());
  nodes.emplace_back();

  // Leaf node.
  if (n <= max_leaf_size) {
    nodes[idx].split_dim = -1;
    nodes[idx].split_val = 0.0;
    nodes[idx].left_bound = 0.0;
    nodes[idx].right_bound = 0.0;
    nodes[idx].left_child = -1;
    nodes[idx].right_child = -1;
    nodes[idx].point_indices = indices;
    return idx;
  }

  int d = static_cast<int>(data.n_cols);

  // Find dimension of maximum variance.
  int best_dim = 0;
  double best_var = -1.0;
  for (int f = 0; f < d; f++) {
    double sum = 0.0, sum2 = 0.0;
    for (int i : indices) {
      double v = data(i, f);
      sum += v;
      sum2 += v * v;
    }
    double var = sum2 / n - (sum / n) * (sum / n);
    if (var > best_var) {
      best_var = var;
      best_dim = f;
    }
  }

  // Find median on best_dim.
  std::vector<double> vals(n);
  for (int i = 0; i < n; i++) vals[i] = data(indices[i], best_dim);
  std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
  double median_val = vals[n / 2];

  // Compute tau_abs: tau * range on this dimension.
  double min_val = *std::min_element(vals.begin(), vals.end());
  double max_val = *std::max_element(vals.begin(), vals.end());
  double tau_abs = tau * (max_val - min_val);

  // Partition with overlap.
  std::vector<int> left_indices, right_indices;
  for (int i : indices) {
    double v = data(i, best_dim);
    if (v <= median_val + tau_abs) left_indices.push_back(i);
    if (v >= median_val - tau_abs) right_indices.push_back(i);
  }

  // If either side has all points (no split progress), make leaf.
  if (static_cast<int>(left_indices.size()) == n ||
      static_cast<int>(right_indices.size()) == n) {
    nodes[idx].split_dim = -1;
    nodes[idx].split_val = 0.0;
    nodes[idx].left_bound = 0.0;
    nodes[idx].right_bound = 0.0;
    nodes[idx].left_child = -1;
    nodes[idx].right_child = -1;
    nodes[idx].point_indices = indices;
    return idx;
  }

  nodes[idx].split_dim = best_dim;
  nodes[idx].split_val = median_val;
  nodes[idx].left_bound = median_val + tau_abs;
  nodes[idx].right_bound = median_val - tau_abs;

  int left_child = build_spill_tree(data, left_indices, tau, max_leaf_size, nodes);
  nodes[idx].left_child = left_child;

  int right_child = build_spill_tree(data, right_indices, tau, max_leaf_size, nodes);
  nodes[idx].right_child = right_child;

  return idx;
}

// KNN search: defeatist search with priority queue.
struct Neighbor {
  int index;
  double distance;
  bool operator>(const Neighbor& other) const {
    return distance > other.distance;
  }
  bool operator<(const Neighbor& other) const {
    return distance < other.distance;
  }
};

static void search_spill_tree(
    const arma::mat& data,
    const arma::rowvec& query,
    const std::vector<SpillNode>& nodes,
    int node_idx,
    int k,
    std::priority_queue<Neighbor>& heap  // max-heap of k best
) {
  const SpillNode& node = nodes[node_idx];

  if (node.split_dim == -1) {
    // Leaf: check all points.
    for (int pi : node.point_indices) {
      double dist = arma::norm(data.row(pi) - query, 2);
      if (static_cast<int>(heap.size()) < k) {
        heap.push({pi, dist});
      } else if (dist < heap.top().distance) {
        heap.pop();
        heap.push({pi, dist});
      }
    }
    return;
  }

  double val = query(node.split_dim);

  // Defeatist search: visit the closer child first, then the other
  // if the query could be within the overlap region.
  bool go_left_first = (val <= node.split_val);

  int first_child = go_left_first ? node.left_child : node.right_child;
  int second_child = go_left_first ? node.right_child : node.left_child;

  search_spill_tree(data, query, nodes, first_child, k, heap);

  // Visit second child if query is in the overlap zone or we don't have k neighbors yet.
  double dist_to_split = std::abs(val - node.split_val);
  double worst_dist = (static_cast<int>(heap.size()) < k) ?
    std::numeric_limits<double>::infinity() : heap.top().distance;

  if (dist_to_split < worst_dist ||
      (val >= node.right_bound && val <= node.left_bound)) {
    search_spill_tree(data, query, nodes, second_child, k, heap);
  }
}

// [[Rcpp::export]]
Rcpp::List cpp_build_and_search_spill_tree(
    const arma::mat& reference,
    const arma::mat& query,
    int k,
    double tau,
    int max_leaf_size
) {
  int n_query = static_cast<int>(query.n_rows);
  int n_ref = static_cast<int>(reference.n_rows);

  // Build tree.
  std::vector<SpillNode> nodes;
  std::vector<int> all_indices(n_ref);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  build_spill_tree(reference, all_indices, tau, max_leaf_size, nodes);

  // Ensure k <= n_ref.
  int actual_k = std::min(k, n_ref);

  // Search for each query point.
  arma::imat neighbor_indices(n_query, actual_k);
  arma::mat neighbor_distances(n_query, actual_k);

  for (int i = 0; i < n_query; i++) {
    std::priority_queue<Neighbor> heap;
    search_spill_tree(reference, query.row(i), nodes, 0, actual_k, heap);

    // Extract from heap (comes out in reverse order).
    int found = static_cast<int>(heap.size());
    for (int j = found - 1; j >= 0; j--) {
      neighbor_indices(i, j) = heap.top().index + 1; // 1-indexed for R
      neighbor_distances(i, j) = heap.top().distance;
      heap.pop();
    }
    // Fill remaining if fewer than k found (shouldn't happen).
    for (int j = found; j < actual_k; j++) {
      neighbor_indices(i, j) = -1;
      neighbor_distances(i, j) = std::numeric_limits<double>::infinity();
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("neighbors") = neighbor_indices,
    Rcpp::Named("distances") = neighbor_distances,
    Rcpp::Named("n_nodes") = static_cast<int>(nodes.size())
  );
}
