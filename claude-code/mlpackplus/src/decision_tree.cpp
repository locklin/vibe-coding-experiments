// [[Rcpp::depends(RcppArmadillo)]]
#include "tree_utils.h"

// BFS-based CART regression tree builder.
// Returns a list with tree node arrays.
// [[Rcpp::export]]
Rcpp::List cpp_fit_decision_tree(
    const arma::mat& X,
    const arma::vec& y,
    int max_depth,
    int min_leaf_size
) {
  int n = static_cast<int>(X.n_rows);

  TreeNodeArrays tree;

  // Compute root mean.
  double root_mean = arma::mean(y);

  // Create root node.
  tree.add_node(-1, 0.0, root_mean, n);

  // BFS queue: (node_index, indices, depth).
  struct QueueItem {
    int node_idx;
    std::vector<int> indices;
    int depth;
  };

  std::queue<QueueItem> bfs;
  std::vector<int> all_indices(n);
  std::iota(all_indices.begin(), all_indices.end(), 0);
  bfs.push({0, all_indices, 0});

  while (!bfs.empty()) {
    QueueItem item = std::move(bfs.front());
    bfs.pop();

    int node_idx = item.node_idx;
    int depth = item.depth;

    if (depth >= max_depth) continue;
    if (static_cast<int>(item.indices.size()) < 2 * min_leaf_size) continue;

    SplitResult split = find_best_split(X, y, item.indices, min_leaf_size);
    if (!split.found) continue;

    // Partition indices.
    std::vector<int> left_indices, right_indices;
    for (int idx : item.indices) {
      if (X(idx, split.feature) <= split.threshold) {
        left_indices.push_back(idx);
      } else {
        right_indices.push_back(idx);
      }
    }

    // Compute means.
    double left_mean = 0.0, right_mean = 0.0;
    for (int idx : left_indices) left_mean += y(idx);
    left_mean /= left_indices.size();
    for (int idx : right_indices) right_mean += y(idx);
    right_mean /= right_indices.size();

    // Create child nodes.
    int left_idx = tree.add_node(-1, 0.0, left_mean,
                                  static_cast<int>(left_indices.size()));
    int right_idx = tree.add_node(-1, 0.0, right_mean,
                                   static_cast<int>(right_indices.size()));

    // Update parent.
    tree.feature[node_idx] = split.feature;
    tree.threshold[node_idx] = split.threshold;
    tree.left_child[node_idx] = left_idx;
    tree.right_child[node_idx] = right_idx;

    // Enqueue children.
    bfs.push({left_idx, std::move(left_indices), depth + 1});
    bfs.push({right_idx, std::move(right_indices), depth + 1});
  }

  return Rcpp::List::create(
    Rcpp::Named("feature") = Rcpp::wrap(tree.feature),
    Rcpp::Named("threshold") = Rcpp::wrap(tree.threshold),
    Rcpp::Named("value") = Rcpp::wrap(tree.value),
    Rcpp::Named("left_child") = Rcpp::wrap(tree.left_child),
    Rcpp::Named("right_child") = Rcpp::wrap(tree.right_child),
    Rcpp::Named("n_samples") = Rcpp::wrap(tree.n_samples)
  );
}

// Predict using a fitted tree (root-to-leaf traversal).
// [[Rcpp::export]]
arma::vec cpp_predict_decision_tree(
    const arma::mat& X,
    const Rcpp::IntegerVector& feature,
    const Rcpp::NumericVector& threshold,
    const Rcpp::NumericVector& value,
    const Rcpp::IntegerVector& left_child,
    const Rcpp::IntegerVector& right_child
) {
  int n = static_cast<int>(X.n_rows);
  arma::vec predictions(n);

  for (int i = 0; i < n; i++) {
    int node = 0;
    while (feature[node] != -1) {
      if (X(i, feature[node]) <= threshold[node]) {
        node = left_child[node];
      } else {
        node = right_child[node];
      }
    }
    predictions(i) = value[node];
  }

  return predictions;
}
