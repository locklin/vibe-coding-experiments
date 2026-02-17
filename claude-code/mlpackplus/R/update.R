#' Update (Incremental Training) Methods
#'
#' @name update-methods
NULL

#' Update a DecisionTreeRegressor
#'
#' CART trees cannot be updated incrementally. This method errors informatively.
#'
#' @param object A DecisionTreeRegressor.
#' @param ... Ignored.
#' @export
update.DecisionTreeRegressor <- function(object, ...) {
  stop("DecisionTreeRegressor does not support incremental updates. ",
       "Refit the model with the full dataset instead.")
}

#' Update a RandomForestRegressor (Online Bagging)
#'
#' Adds new trees trained on new data to an existing forest.
#'
#' @param object A RandomForestRegressor.
#' @param x New predictor matrix.
#' @param y New response vector.
#' @param n_new_trees Number of new trees to add (default 10).
#' @param ... Additional arguments (ignored).
#' @return An updated RandomForestRegressor.
#' @export
update.RandomForestRegressor <- function(object, x, y, n_new_trees = 10L,
                                          ...) {
  x <- as.matrix(x)
  y <- as.numeric(y)

  if (nrow(x) != length(y)) {
    stop("Number of rows in x must match length of y")
  }
  if (ncol(x) != object$n_features) {
    stop("New data must have ", object$n_features, " columns")
  }

  # Train new trees on the new data.
  new_forest <- RandomForestRegressor(
    x, y,
    n_trees = n_new_trees,
    max_depth = object$max_depth,
    min_leaf_size = object$min_leaf_size,
    max_features = object$max_features,
    sample_fraction = object$sample_fraction
  )

  # Append new trees.
  all_trees <- c(object$trees, new_forest$trees)

  object$trees <- all_trees
  object$n_trees <- object$n_trees + n_new_trees
  object$n_samples <- object$n_samples + nrow(x)
  object
}

#' Update an AMF Model
#'
#' Extends W (new rows) or H (new columns) by solving one factor while
#' fixing the other.
#'
#' @param object An AMF model.
#' @param new_rows Numeric matrix of new rows to append to V (optional).
#'   Must have the same number of columns as the original V.
#' @param new_cols Numeric matrix of new columns to append to V (optional).
#'   Must have the same number of rows as the original V.
#' @param lambda Ridge regularization (default 0.01).
#' @param ... Additional arguments (ignored).
#' @return An updated AMF model.
#' @export
update.AMF <- function(object, new_rows = NULL, new_cols = NULL,
                        lambda = 0.01, ...) {
  if (is.null(new_rows) && is.null(new_cols)) {
    stop("Must provide new_rows and/or new_cols")
  }

  W <- object$W
  H <- object$H
  r <- object$rank
  I_reg <- lambda * diag(r)

  if (!is.null(new_rows)) {
    new_rows <- as.matrix(new_rows)
    if (ncol(new_rows) != object$n) {
      stop("new_rows must have ", object$n, " columns")
    }
    # Solve for new W rows: W_new = new_rows %*% t(H) %*% solve(H %*% t(H) + lambda*I)
    HHt <- H %*% t(H) + I_reg
    W_new <- t(solve(HHt, H %*% t(new_rows)))
    W_new[W_new < 0] <- 0
    W <- rbind(W, W_new)
    object$m <- object$m + nrow(new_rows)
  }

  if (!is.null(new_cols)) {
    new_cols <- as.matrix(new_cols)
    if (nrow(new_cols) != object$m) {
      stop("new_cols must have ", object$m, " rows")
    }
    # Solve for new H columns: H_new = solve(t(W) %*% W + lambda*I, t(W) %*% new_cols)
    WtW <- t(W) %*% W + I_reg
    H_new <- solve(WtW, t(W) %*% new_cols)
    H_new[H_new < 0] <- 0
    H <- cbind(H, H_new)
    object$n <- object$n + ncol(new_cols)
  }

  object$W <- W
  object$H <- H
  object$final_residual <- NA_real_  # stale after update
  object
}
