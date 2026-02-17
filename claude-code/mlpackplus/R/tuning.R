#' Grid Search Hyperparameter Tuning
#'
#' Evaluates all combinations of hyperparameters using k-fold cross-validation.
#'
#' @param model_factory A function that takes named hyperparameter values and
#'   returns a \code{model_fn} (a function(x, y) -> model).
#' @param x Numeric matrix of predictors.
#' @param y Numeric vector of responses.
#' @param param_grid Named list of parameter vectors to search over.
#' @param k Number of CV folds (default 5).
#' @param metric_fn Metric function (default \code{mse}).
#' @return A list with \code{best_params}, \code{best_score}, and
#'   \code{results} data.frame.
#' @export
#' @examples
#' set.seed(42)
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- x[,1]^2 + rnorm(100, sd = 0.1)
#' factory <- function(max_depth, min_leaf_size) {
#'   function(x, y) DecisionTreeRegressor(x, y,
#'     max_depth = max_depth, min_leaf_size = min_leaf_size)
#' }
#' result <- grid_search(factory, x, y,
#'   param_grid = list(max_depth = c(3, 5), min_leaf_size = c(2, 5)), k = 3)
grid_search <- function(model_factory, x, y, param_grid, k = 5L,
                         metric_fn = mse) {
  x <- as.matrix(x)
  y <- as.numeric(y)

  grid <- expand.grid(param_grid, stringsAsFactors = FALSE)
  n_combos <- nrow(grid)

  scores <- numeric(n_combos)
  for (i in seq_len(n_combos)) {
    params <- as.list(grid[i, , drop = FALSE])
    model_fn <- do.call(model_factory, params)
    cv_result <- kfold_cv(model_fn, x, y, k = k, metric_fn = metric_fn)
    scores[i] <- cv_result$mean_score
  }

  grid$score <- scores
  best_idx <- which.min(scores)

  best_params <- as.list(grid[best_idx, setdiff(names(grid), "score"),
                               drop = FALSE])

  list(
    best_params = best_params,
    best_score = scores[best_idx],
    results = grid
  )
}

#' Random Search Hyperparameter Tuning
#'
#' Evaluates random hyperparameter configurations using k-fold cross-validation.
#'
#' @param model_factory A function that takes named hyperparameter values and
#'   returns a \code{model_fn}.
#' @param x Numeric matrix of predictors.
#' @param y Numeric vector of responses.
#' @param param_distributions Named list of functions that each take no
#'   arguments and return a sampled parameter value.
#' @param n_iter Number of random configurations to try (default 20).
#' @param k Number of CV folds (default 5).
#' @param metric_fn Metric function (default \code{mse}).
#' @return A list with \code{best_params}, \code{best_score}, and
#'   \code{results} data.frame.
#' @export
#' @examples
#' set.seed(42)
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- x[,1]^2 + rnorm(100, sd = 0.1)
#' factory <- function(max_depth, min_leaf_size) {
#'   function(x, y) DecisionTreeRegressor(x, y,
#'     max_depth = max_depth, min_leaf_size = min_leaf_size)
#' }
#' result <- random_search(factory, x, y,
#'   param_distributions = list(
#'     max_depth = function() sample(3:10, 1),
#'     min_leaf_size = function() sample(2:10, 1)
#'   ), n_iter = 5, k = 3)
random_search <- function(model_factory, x, y, param_distributions,
                           n_iter = 20L, k = 5L, metric_fn = mse) {
  x <- as.matrix(x)
  y <- as.numeric(y)
  n_iter <- as.integer(n_iter)

  param_names <- names(param_distributions)
  results_list <- vector("list", n_iter)
  scores <- numeric(n_iter)

  for (i in seq_len(n_iter)) {
    params <- lapply(param_distributions, function(fn) fn())
    names(params) <- param_names

    model_fn <- do.call(model_factory, params)
    cv_result <- kfold_cv(model_fn, x, y, k = k, metric_fn = metric_fn)
    scores[i] <- cv_result$mean_score
    results_list[[i]] <- c(params, list(score = cv_result$mean_score))
  }

  results <- do.call(rbind, lapply(results_list, as.data.frame))

  best_idx <- which.min(scores)
  best_params <- as.list(results[best_idx, setdiff(names(results), "score"),
                                  drop = FALSE])

  list(
    best_params = best_params,
    best_score = scores[best_idx],
    results = results
  )
}
