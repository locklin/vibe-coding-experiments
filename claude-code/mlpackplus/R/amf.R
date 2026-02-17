#' Alternating Matrix Factorization (NMF via ALS)
#'
#' Performs Non-negative Matrix Factorization using Alternating Least Squares.
#' Factorizes V ~ W * H where all entries of W and H are non-negative.
#'
#' @param V Numeric matrix to factorize (m x n). Should have non-negative entries.
#' @param rank Factorization rank (number of latent factors).
#' @param max_iter Maximum iterations (default 100).
#' @param tol Convergence tolerance on relative Frobenius norm change (default 1e-5).
#' @param lambda Ridge regularization parameter (default 0.01).
#' @param seed Random seed for reproducibility (default NULL).
#' @return An S3 object of class \code{AMF}.
#' @export
#' @examples
#' set.seed(42)
#' W_true <- matrix(runif(30), nrow = 10, ncol = 3)
#' H_true <- matrix(runif(15), nrow = 3, ncol = 5)
#' V <- W_true %*% H_true
#' model <- AMF(V, rank = 3, seed = 42)
#' reconstruction <- predict(model)
AMF <- function(V, rank, max_iter = 100L, tol = 1e-5, lambda = 0.01,
                seed = NULL) {
  V <- as.matrix(V)
  rank <- as.integer(rank)

  if (rank < 1L) stop("rank must be >= 1")
  if (rank > min(nrow(V), ncol(V))) {
    stop("rank must be <= min(nrow(V), ncol(V))")
  }
  if (max_iter < 1L) stop("max_iter must be >= 1")
  if (tol < 0) stop("tol must be >= 0")

  if (!is.null(seed)) set.seed(seed)

  result <- cpp_fit_amf(V, rank, as.integer(max_iter), tol, lambda)

  structure(
    list(
      W = result$W,
      H = result$H,
      rank = rank,
      iterations = result$iterations,
      initial_residual = result$initial_residual,
      final_residual = result$final_residual,
      m = nrow(V),
      n = ncol(V)
    ),
    class = "AMF"
  )
}

#' @export
predict.AMF <- function(object, ...) {
  object$W %*% object$H
}

#' @export
print.AMF <- function(x, ...) {
  cat("AMF (Non-negative Matrix Factorization)\n")
  cat("  Dimensions:       ", x$m, "x", x$n, "\n")
  cat("  Rank:             ", x$rank, "\n")
  cat("  Iterations:       ", x$iterations, "\n")
  cat("  Initial residual: ", sprintf("%.6f", x$initial_residual), "\n")
  cat("  Final residual:   ", sprintf("%.6f", x$final_residual), "\n")
  invisible(x)
}
