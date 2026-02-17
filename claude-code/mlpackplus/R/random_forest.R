#' Random Forest Regressor
#'
#' Fits a random forest for regression using bagged CART trees with
#' random feature subsets.
#'
#' @param x Numeric matrix of predictors (n x d).
#' @param y Numeric vector of responses (length n).
#' @param n_trees Number of trees (default 100).
#' @param max_depth Maximum tree depth (default 10).
#' @param min_leaf_size Minimum samples per leaf (default 5).
#' @param max_features Number of features to consider at each split
#'   (default floor(d/3)).
#' @param sample_fraction Fraction of data to sample per tree (default 1.0).
#' @param seed Random seed for reproducibility (default NULL).
#' @return An S3 object of class \code{RandomForestRegressor}.
#' @export
#' @examples
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- sin(x[,1]) + 0.5 * cos(x[,2]) + rnorm(100, sd = 0.1)
#' model <- RandomForestRegressor(x, y, n_trees = 50, seed = 42)
#' preds <- predict(model, x)
RandomForestRegressor <- function(x, y, n_trees = 100L, max_depth = 10L,
                                   min_leaf_size = 5L, max_features = NULL,
                                   sample_fraction = 1.0, seed = NULL) {
  x <- as.matrix(x)
  y <- as.numeric(y)

  if (nrow(x) != length(y)) {
    stop("Number of rows in x must match length of y")
  }
  if (n_trees < 1L) stop("n_trees must be >= 1")
  if (max_depth < 1L) stop("max_depth must be >= 1")
  if (min_leaf_size < 1L) stop("min_leaf_size must be >= 1")
  if (sample_fraction <= 0 || sample_fraction > 1) {
    stop("sample_fraction must be in (0, 1]")
  }

  d <- ncol(x)
  if (is.null(max_features)) {
    max_features <- max(1L, as.integer(floor(d / 3)))
  }
  max_features <- as.integer(min(max_features, d))

  if (is.null(seed)) seed <- sample.int(.Machine$integer.max, 1L)

  result <- cpp_fit_random_forest(
    x, y,
    as.integer(n_trees),
    as.integer(max_depth),
    as.integer(min_leaf_size),
    max_features,
    sample_fraction,
    as.integer(seed)
  )

  structure(
    list(
      trees = result$trees,
      oob_mse = result$oob_mse,
      oob_n = result$oob_n,
      n_trees = n_trees,
      max_depth = max_depth,
      min_leaf_size = min_leaf_size,
      max_features = max_features,
      sample_fraction = sample_fraction,
      n_features = d,
      n_samples = nrow(x),
      seed = seed
    ),
    class = "RandomForestRegressor"
  )
}

#' @export
predict.RandomForestRegressor <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  if (ncol(newdata) != object$n_features) {
    stop("newdata must have ", object$n_features, " columns")
  }
  as.numeric(cpp_predict_random_forest(newdata, object$trees))
}

#' @export
print.RandomForestRegressor <- function(x, ...) {
  cat("RandomForestRegressor\n")
  cat("  Trees:         ", x$n_trees, "\n")
  cat("  Max depth:     ", x$max_depth, "\n")
  cat("  Min leaf size: ", x$min_leaf_size, "\n")
  cat("  Max features:  ", x$max_features, "\n")
  cat("  Sample fraction:", x$sample_fraction, "\n")
  cat("  Features:      ", x$n_features, "\n")
  cat("  Train samples: ", x$n_samples, "\n")
  cat("  OOB MSE:       ", sprintf("%.6f", x$oob_mse), "\n")
  invisible(x)
}
