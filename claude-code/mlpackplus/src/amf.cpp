// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

// NMF via Alternating Least Squares with non-negativity projection.
// V ~ W * H, where V is m x n, W is m x rank, H is rank x n.
// [[Rcpp::export]]
Rcpp::List cpp_fit_amf(
    const arma::mat& V,
    int rank,
    int max_iter,
    double tol,
    double lambda
) {
  int m = static_cast<int>(V.n_rows);
  int n_cols = static_cast<int>(V.n_cols);

  // Initialize W and H with random uniform values using R's RNG.
  arma::mat W(m, rank);
  arma::mat H(rank, n_cols);
  for (int i = 0; i < m; i++)
    for (int j = 0; j < rank; j++)
      W(i, j) = R::runif(0.0, 1.0);
  for (int i = 0; i < rank; i++)
    for (int j = 0; j < n_cols; j++)
      H(i, j) = R::runif(0.0, 1.0);

  double prev_norm = arma::norm(V - W * H, "fro");
  double initial_norm = prev_norm;
  int iterations = 0;

  arma::mat I_reg = lambda * arma::eye<arma::mat>(rank, rank);

  for (int iter = 0; iter < max_iter; iter++) {
    // Update H: solve (W'W + lambda*I) H = W'V
    arma::mat WtW = W.t() * W + I_reg;
    arma::mat WtV = W.t() * V;
    H = arma::solve(WtW, WtV);
    // Clamp negatives to zero.
    H.clamp(0.0, arma::datum::inf);

    // Update W: solve (HH' + lambda*I) W' = H V'
    arma::mat HHt = H * H.t() + I_reg;
    arma::mat HVt = H * V.t();
    arma::mat Wt = arma::solve(HHt, HVt);
    W = Wt.t();
    // Clamp negatives.
    W.clamp(0.0, arma::datum::inf);

    double cur_norm = arma::norm(V - W * H, "fro");
    iterations = iter + 1;

    // Check convergence (relative change).
    if (prev_norm > 0) {
      double rel_change = std::abs(prev_norm - cur_norm) / prev_norm;
      if (rel_change < tol) break;
    }
    prev_norm = cur_norm;
  }

  double final_norm = arma::norm(V - W * H, "fro");

  return Rcpp::List::create(
    Rcpp::Named("W") = W,
    Rcpp::Named("H") = H,
    Rcpp::Named("iterations") = iterations,
    Rcpp::Named("initial_residual") = initial_norm,
    Rcpp::Named("final_residual") = final_norm
  );
}
