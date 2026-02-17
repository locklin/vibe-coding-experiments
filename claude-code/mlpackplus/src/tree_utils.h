#ifndef MLPACKPLUS_TREE_UTILS_H
#define MLPACKPLUS_TREE_UTILS_H

#include <RcppArmadillo.h>
#include <vector>
#include <queue>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

// Parallel-array tree node storage.
// Each node is stored at an index; children referenced by index.
struct TreeNodeArrays {
  std::vector<int> feature;       // split feature (-1 for leaf)
  std::vector<double> threshold;  // split threshold
  std::vector<double> value;      // prediction value (mean of leaf)
  std::vector<int> left_child;    // index of left child (-1 if none)
  std::vector<int> right_child;   // index of right child (-1 if none)
  std::vector<int> n_samples;     // number of samples in node

  int add_node(int feat, double thresh, double val, int n) {
    int idx = static_cast<int>(feature.size());
    feature.push_back(feat);
    threshold.push_back(thresh);
    value.push_back(val);
    left_child.push_back(-1);
    right_child.push_back(-1);
    n_samples.push_back(n);
    return idx;
  }

  int size() const {
    return static_cast<int>(feature.size());
  }
};

// Result of a split search.
struct SplitResult {
  bool found;
  int feature;
  double threshold;
  double improvement;
};

// Find the best split across all features using MSE criterion.
// Sorted-scan approach: O(n * d).
inline SplitResult find_best_split(
    const arma::mat& X,
    const arma::vec& y,
    const std::vector<int>& indices,
    int min_leaf_size
) {
  SplitResult result;
  result.found = false;
  result.feature = -1;
  result.threshold = 0.0;
  result.improvement = 0.0;

  int n = static_cast<int>(indices.size());
  int d = static_cast<int>(X.n_cols);

  if (n < 2 * min_leaf_size) return result;

  // Compute parent MSE.
  double sum_y = 0.0, sum_y2 = 0.0;
  for (int i = 0; i < n; i++) {
    double yi = y(indices[i]);
    sum_y += yi;
    sum_y2 += yi * yi;
  }
  double parent_mse = sum_y2 / n - (sum_y / n) * (sum_y / n);

  double best_score = parent_mse;  // We want to minimize weighted child MSE.
  // Actually: track best impurity decrease.

  // For each feature, sort indices by feature value and scan.
  std::vector<int> sorted_idx(indices.begin(), indices.end());

  for (int f = 0; f < d; f++) {
    // Sort by feature f.
    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&X, f](int a, int b) { return X(a, f) < X(b, f); });

    double left_sum = 0.0, left_sum2 = 0.0;
    double right_sum = sum_y, right_sum2 = sum_y2;

    for (int i = 0; i < n - 1; i++) {
      double yi = y(sorted_idx[i]);
      left_sum += yi;
      left_sum2 += yi * yi;
      right_sum -= yi;
      right_sum2 -= yi * yi;

      int left_n = i + 1;
      int right_n = n - left_n;

      if (left_n < min_leaf_size || right_n < min_leaf_size) continue;

      // Skip if same feature value (can't split here).
      if (X(sorted_idx[i], f) == X(sorted_idx[i + 1], f)) continue;

      double left_mse = left_sum2 / left_n - (left_sum / left_n) * (left_sum / left_n);
      double right_mse = right_sum2 / right_n - (right_sum / right_n) * (right_sum / right_n);

      double weighted_mse = (left_mse * left_n + right_mse * right_n) / n;
      double improvement = parent_mse - weighted_mse;

      if (improvement > result.improvement) {
        result.found = true;
        result.feature = f;
        result.threshold = (X(sorted_idx[i], f) + X(sorted_idx[i + 1], f)) / 2.0;
        result.improvement = improvement;
      }
    }
  }

  return result;
}

// Find best split considering only a random subset of features.
inline SplitResult find_best_split_random_features(
    const arma::mat& X,
    const arma::vec& y,
    const std::vector<int>& indices,
    int min_leaf_size,
    int max_features,
    std::mt19937& rng
) {
  SplitResult result;
  result.found = false;
  result.feature = -1;
  result.threshold = 0.0;
  result.improvement = 0.0;

  int n = static_cast<int>(indices.size());
  int d = static_cast<int>(X.n_cols);

  if (n < 2 * min_leaf_size) return result;

  // Select random features.
  int nf = std::min(max_features, d);
  std::vector<int> all_features(d);
  std::iota(all_features.begin(), all_features.end(), 0);
  std::shuffle(all_features.begin(), all_features.end(), rng);
  std::vector<int> selected_features(all_features.begin(), all_features.begin() + nf);

  // Compute parent stats.
  double sum_y = 0.0, sum_y2 = 0.0;
  for (int i = 0; i < n; i++) {
    double yi = y(indices[i]);
    sum_y += yi;
    sum_y2 += yi * yi;
  }
  double parent_mse = sum_y2 / n - (sum_y / n) * (sum_y / n);

  std::vector<int> sorted_idx(indices.begin(), indices.end());

  for (int fi = 0; fi < nf; fi++) {
    int f = selected_features[fi];

    std::sort(sorted_idx.begin(), sorted_idx.end(),
              [&X, f](int a, int b) { return X(a, f) < X(b, f); });

    double left_sum = 0.0, left_sum2 = 0.0;
    double right_sum = sum_y, right_sum2 = sum_y2;

    for (int i = 0; i < n - 1; i++) {
      double yi = y(sorted_idx[i]);
      left_sum += yi;
      left_sum2 += yi * yi;
      right_sum -= yi;
      right_sum2 -= yi * yi;

      int left_n = i + 1;
      int right_n = n - left_n;

      if (left_n < min_leaf_size || right_n < min_leaf_size) continue;
      if (X(sorted_idx[i], f) == X(sorted_idx[i + 1], f)) continue;

      double left_mse = left_sum2 / left_n - (left_sum / left_n) * (left_sum / left_n);
      double right_mse = right_sum2 / right_n - (right_sum / right_n) * (right_sum / right_n);

      double weighted_mse = (left_mse * left_n + right_mse * right_n) / n;
      double improvement = parent_mse - weighted_mse;

      if (improvement > result.improvement) {
        result.found = true;
        result.feature = f;
        result.threshold = (X(sorted_idx[i], f) + X(sorted_idx[i + 1], f)) / 2.0;
        result.improvement = improvement;
      }
    }
  }

  return result;
}

#endif // MLPACKPLUS_TREE_UTILS_H
