#' K-Fold Cross-Validation
#'
#' Evaluates a model using k-fold cross-validation.
#'
#' @param model_fn A function(x, y) that returns a fitted model with a
#'   \code{predict} method.
#' @param x Numeric matrix of predictors.
#' @param y Numeric vector of responses.
#' @param k Number of folds (default 5).
#' @param metric_fn A function(actual, predicted) returning a scalar score
#'   (lower is better by convention).
#' @return A list with \code{mean_score} and \code{fold_scores}.
#' @export
#' @examples
#' set.seed(42)
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- x[,1]^2 + rnorm(100, sd = 0.1)
#' model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 5)
#' result <- kfold_cv(model_fn, x, y, k = 5, metric_fn = mse)
kfold_cv <- function(model_fn, x, y, k = 5L, metric_fn = mse) {
  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)
  k <- as.integer(k)

  if (k < 2L) stop("k must be >= 2")
  if (k > n) stop("k must be <= number of samples")

  # Create fold assignments.
  fold_ids <- rep(seq_len(k), length.out = n)
  fold_ids <- fold_ids[sample(n)]

  fold_scores <- numeric(k)
  for (i in seq_len(k)) {
    test_idx <- which(fold_ids == i)
    train_idx <- which(fold_ids != i)

    model <- model_fn(x[train_idx, , drop = FALSE], y[train_idx])
    preds <- predict(model, x[test_idx, , drop = FALSE])
    fold_scores[i] <- metric_fn(y[test_idx], preds)
  }

  list(
    mean_score = mean(fold_scores),
    fold_scores = fold_scores
  )
}

#' Simple Train/Test Split Cross-Validation
#'
#' Evaluates a model using a single train/test split.
#'
#' @param model_fn A function(x, y) that returns a fitted model.
#' @param x Numeric matrix of predictors.
#' @param y Numeric vector of responses.
#' @param train_fraction Fraction of data for training (default 0.8).
#' @param metric_fn A function(actual, predicted) returning a scalar score.
#' @return A list with \code{score}, \code{train_indices}, and
#'   \code{test_indices}.
#' @export
#' @examples
#' set.seed(42)
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- x[,1]^2 + rnorm(100, sd = 0.1)
#' model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 5)
#' result <- simple_cv(model_fn, x, y, train_fraction = 0.8, metric_fn = mse)
simple_cv <- function(model_fn, x, y, train_fraction = 0.8, metric_fn = mse) {
  x <- as.matrix(x)
  y <- as.numeric(y)
  n <- nrow(x)

  if (train_fraction <= 0 || train_fraction >= 1) {
    stop("train_fraction must be in (0, 1)")
  }

  n_train <- max(1L, as.integer(round(n * train_fraction)))
  if (n_train >= n) n_train <- n - 1L

  shuffled <- sample(n)
  train_indices <- shuffled[seq_len(n_train)]
  test_indices <- shuffled[(n_train + 1L):n]

  model <- model_fn(x[train_indices, , drop = FALSE], y[train_indices])
  preds <- predict(model, x[test_indices, , drop = FALSE])
  score <- metric_fn(y[test_indices], preds)

  list(
    score = score,
    train_indices = train_indices,
    test_indices = test_indices
  )
}
