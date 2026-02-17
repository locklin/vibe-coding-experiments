#' mlpack-style Decision Tree Interface
#'
#' Provides a functional interface similar to mlpack's CLI bindings.
#' Trains and/or predicts depending on which arguments are provided.
#'
#' @param training_data Numeric matrix of training predictors (optional).
#' @param training_labels Numeric vector of training responses (optional).
#' @param test_data Numeric matrix of test predictors (optional).
#' @param input_model A previously fitted DecisionTreeRegressor (optional).
#' @param max_depth Maximum tree depth (default 10).
#' @param min_leaf_size Minimum leaf size (default 5).
#' @return A list with \code{output_model} and optionally \code{predictions}.
#' @export
#' @examples
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- x[,1]^2 + rnorm(100, sd = 0.1)
#' # Train and predict in one call
#' result <- decision_tree(training_data = x, training_labels = y,
#'   test_data = x, max_depth = 5)
decision_tree <- function(training_data = NULL, training_labels = NULL,
                           test_data = NULL, input_model = NULL,
                           max_depth = 10L, min_leaf_size = 5L) {
  model <- input_model
  predictions <- NULL

  # Train if training data provided.
  if (!is.null(training_data) && !is.null(training_labels)) {
    model <- DecisionTreeRegressor(
      training_data, training_labels,
      max_depth = max_depth, min_leaf_size = min_leaf_size
    )
  }

  if (is.null(model)) {
    stop("Must provide either training data or an input_model")
  }

  # Predict if test data provided.
  if (!is.null(test_data)) {
    predictions <- predict(model, test_data)
  }

  result <- list(output_model = model)
  if (!is.null(predictions)) result$predictions <- predictions
  result
}

#' mlpack-style Random Forest Interface
#'
#' Provides a functional interface similar to mlpack's CLI bindings.
#'
#' @param training_data Numeric matrix of training predictors (optional).
#' @param training_labels Numeric vector of training responses (optional).
#' @param test_data Numeric matrix of test predictors (optional).
#' @param input_model A previously fitted RandomForestRegressor (optional).
#' @param n_trees Number of trees (default 100).
#' @param max_depth Maximum tree depth (default 10).
#' @param min_leaf_size Minimum leaf size (default 5).
#' @param max_features Features per split (default NULL = d/3).
#' @param sample_fraction Bootstrap sample fraction (default 1.0).
#' @param seed Random seed (default NULL).
#' @return A list with \code{output_model} and optionally \code{predictions}.
#' @export
#' @examples
#' x <- matrix(rnorm(200), ncol = 2)
#' y <- sin(x[,1]) + rnorm(100, sd = 0.1)
#' result <- random_forest(training_data = x, training_labels = y,
#'   test_data = x, n_trees = 50, seed = 42)
random_forest <- function(training_data = NULL, training_labels = NULL,
                           test_data = NULL, input_model = NULL,
                           n_trees = 100L, max_depth = 10L,
                           min_leaf_size = 5L, max_features = NULL,
                           sample_fraction = 1.0, seed = NULL) {
  model <- input_model
  predictions <- NULL

  if (!is.null(training_data) && !is.null(training_labels)) {
    model <- RandomForestRegressor(
      training_data, training_labels,
      n_trees = n_trees, max_depth = max_depth,
      min_leaf_size = min_leaf_size, max_features = max_features,
      sample_fraction = sample_fraction, seed = seed
    )
  }

  if (is.null(model)) {
    stop("Must provide either training data or an input_model")
  }

  if (!is.null(test_data)) {
    predictions <- predict(model, test_data)
  }

  result <- list(output_model = model)
  if (!is.null(predictions)) result$predictions <- predictions
  result
}
