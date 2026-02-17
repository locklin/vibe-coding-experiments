#' Decision Tree Regressor
#'
#' Fits a CART regression tree using MSE splitting criterion.
#'
#' @param x Numeric matrix of predictors (n x d).
#' @param y Numeric vector of responses (length n).
#' @param max_depth Maximum tree depth (default 10).
#' @param min_leaf_size Minimum samples per leaf (default 5).
#' @return An S3 object of class \code{DecisionTreeRegressor}.
#' @export
#' @examples
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- x[,1]^2 + rnorm(100, sd = 0.1)
#' model <- DecisionTreeRegressor(x, y, max_depth = 5)
#' preds <- predict(model, x)
DecisionTreeRegressor <- function(x, y, max_depth = 10L, min_leaf_size = 5L) {
  x <- as.matrix(x)
  y <- as.numeric(y)

  if (nrow(x) != length(y)) {
    stop("Number of rows in x must match length of y")
  }
  if (max_depth < 1L) stop("max_depth must be >= 1")
  if (min_leaf_size < 1L) stop("min_leaf_size must be >= 1")

  tree <- cpp_fit_decision_tree(x, y, as.integer(max_depth),
                                 as.integer(min_leaf_size))

  structure(
    list(
      tree = tree,
      max_depth = max_depth,
      min_leaf_size = min_leaf_size,
      n_features = ncol(x),
      n_samples = nrow(x),
      n_nodes = length(tree$feature)
    ),
    class = "DecisionTreeRegressor"
  )
}

#' @export
predict.DecisionTreeRegressor <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  if (ncol(newdata) != object$n_features) {
    stop("newdata must have ", object$n_features, " columns")
  }
  as.numeric(cpp_predict_decision_tree(
    newdata,
    object$tree$feature,
    object$tree$threshold,
    object$tree$value,
    object$tree$left_child,
    object$tree$right_child
  ))
}

#' @export
print.DecisionTreeRegressor <- function(x, ...) {
  cat("DecisionTreeRegressor\n")
  cat("  Nodes:        ", x$n_nodes, "\n")
  cat("  Max depth:    ", x$max_depth, "\n")
  cat("  Min leaf size:", x$min_leaf_size, "\n")
  cat("  Features:     ", x$n_features, "\n")
  cat("  Train samples:", x$n_samples, "\n")
  invisible(x)
}
