// [[Rcpp::depends(RcppArmadillo)]]
#include "tree_utils.h"

// Build a single regression tree with random feature subsets and bootstrap.
// Returns tree node arrays as a list.
static Rcpp::List build_one_tree(
    const arma::mat& X,
    const arma::vec& y,
    const std::vector<int>& sample_indices,
    int max_depth,
    int min_leaf_size,
    int max_features,
    std::mt19937& rng
) {
  int n = static_cast<int>(sample_indices.size());

  TreeNodeArrays tree;

  // Root mean.
  double root_mean = 0.0;
  for (int idx : sample_indices) root_mean += y(idx);
  root_mean /= n;

  tree.add_node(-1, 0.0, root_mean, n);

  struct QueueItem {
    int node_idx;
    std::vector<int> indices;
    int depth;
  };

  std::queue<QueueItem> bfs;
  bfs.push({0, sample_indices, 0});

  while (!bfs.empty()) {
    QueueItem item = std::move(bfs.front());
    bfs.pop();

    int node_idx = item.node_idx;
    int depth = item.depth;

    if (depth >= max_depth) continue;
    if (static_cast<int>(item.indices.size()) < 2 * min_leaf_size) continue;

    SplitResult split = find_best_split_random_features(
      X, y, item.indices, min_leaf_size, max_features, rng);

    if (!split.found) continue;

    std::vector<int> left_indices, right_indices;
    for (int idx : item.indices) {
      if (X(idx, split.feature) <= split.threshold) {
        left_indices.push_back(idx);
      } else {
        right_indices.push_back(idx);
      }
    }

    double left_mean = 0.0, right_mean = 0.0;
    for (int idx : left_indices) left_mean += y(idx);
    left_mean /= left_indices.size();
    for (int idx : right_indices) right_mean += y(idx);
    right_mean /= right_indices.size();

    int left_idx = tree.add_node(-1, 0.0, left_mean,
                                  static_cast<int>(left_indices.size()));
    int right_idx = tree.add_node(-1, 0.0, right_mean,
                                   static_cast<int>(right_indices.size()));

    tree.feature[node_idx] = split.feature;
    tree.threshold[node_idx] = split.threshold;
    tree.left_child[node_idx] = left_idx;
    tree.right_child[node_idx] = right_idx;

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

// Predict a single observation through a single tree.
static double predict_one_tree(
    const arma::rowvec& x_row,
    const Rcpp::List& tree
) {
  Rcpp::IntegerVector feature = tree["feature"];
  Rcpp::NumericVector threshold = tree["threshold"];
  Rcpp::NumericVector value = tree["value"];
  Rcpp::IntegerVector left_child = tree["left_child"];
  Rcpp::IntegerVector right_child = tree["right_child"];

  int node = 0;
  while (feature[node] != -1) {
    if (x_row(feature[node]) <= threshold[node]) {
      node = left_child[node];
    } else {
      node = right_child[node];
    }
  }
  return value[node];
}

// [[Rcpp::export]]
Rcpp::List cpp_fit_random_forest(
    const arma::mat& X,
    const arma::vec& y,
    int n_trees,
    int max_depth,
    int min_leaf_size,
    int max_features,
    double sample_fraction,
    int seed
) {
  int n = static_cast<int>(X.n_rows);
  int sample_size = static_cast<int>(std::round(n * sample_fraction));

  std::mt19937 rng(seed);

  Rcpp::List trees(n_trees);

  // OOB tracking: for each sample, accumulate predictions and count.
  arma::vec oob_sum(n, arma::fill::zeros);
  arma::ivec oob_count(n, arma::fill::zeros);

  for (int t = 0; t < n_trees; t++) {
    // Bootstrap sample.
    std::vector<bool> in_bag(n, false);
    std::vector<int> sample_indices(sample_size);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int i = 0; i < sample_size; i++) {
      int idx = dist(rng);
      sample_indices[i] = idx;
      in_bag[idx] = true;
    }

    trees[t] = build_one_tree(X, y, sample_indices, max_depth,
                               min_leaf_size, max_features, rng);

    // OOB predictions.
    Rcpp::List tree_t = trees[t];
    for (int i = 0; i < n; i++) {
      if (!in_bag[i]) {
        double pred = predict_one_tree(X.row(i), tree_t);
        oob_sum(i) += pred;
        oob_count(i) += 1;
      }
    }
  }

  // Compute OOB MSE.
  double oob_mse = 0.0;
  int oob_n = 0;
  for (int i = 0; i < n; i++) {
    if (oob_count(i) > 0) {
      double oob_pred = oob_sum(i) / oob_count(i);
      oob_mse += (y(i) - oob_pred) * (y(i) - oob_pred);
      oob_n++;
    }
  }
  if (oob_n > 0) oob_mse /= oob_n;

  return Rcpp::List::create(
    Rcpp::Named("trees") = trees,
    Rcpp::Named("oob_mse") = oob_mse,
    Rcpp::Named("oob_n") = oob_n
  );
}

// [[Rcpp::export]]
arma::vec cpp_predict_random_forest(
    const arma::mat& X,
    const Rcpp::List& trees
) {
  int n = static_cast<int>(X.n_rows);
  int n_trees = trees.size();
  arma::vec predictions(n, arma::fill::zeros);

  for (int t = 0; t < n_trees; t++) {
    Rcpp::List tree_t = trees[t];
    for (int i = 0; i < n; i++) {
      predictions(i) += predict_one_tree(X.row(i), tree_t);
    }
  }

  predictions /= n_trees;
  return predictions;
}
