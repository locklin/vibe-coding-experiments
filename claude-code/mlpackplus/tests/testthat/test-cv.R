test_that("kfold_cv returns correct structure", {
  data <- make_step_data(n = 100)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 3)
  result <- kfold_cv(model_fn, data$x, data$y, k = 5)
  expect_true(is.list(result))
  expect_true("mean_score" %in% names(result))
  expect_true("fold_scores" %in% names(result))
  expect_length(result$fold_scores, 5)
})

test_that("kfold_cv mean_score equals mean of fold_scores", {
  set.seed(42)
  data <- make_step_data(n = 100)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 3)
  result <- kfold_cv(model_fn, data$x, data$y, k = 5)
  expect_equal(result$mean_score, mean(result$fold_scores))
})

test_that("kfold_cv scores are positive for MSE", {
  set.seed(42)
  data <- make_sine_data(n = 100)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 5)
  result <- kfold_cv(model_fn, data$x, data$y, k = 3, metric_fn = mse)
  expect_true(all(result$fold_scores > 0))
})

test_that("kfold_cv input validation", {
  data <- make_step_data(n = 10)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 2,
                                                     min_leaf_size = 1)
  expect_error(kfold_cv(model_fn, data$x, data$y, k = 1), "k must be >= 2")
  expect_error(kfold_cv(model_fn, data$x, data$y, k = 20),
               "k must be <= number")
})

test_that("simple_cv returns correct structure", {
  set.seed(42)
  data <- make_step_data(n = 100)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 3)
  result <- simple_cv(model_fn, data$x, data$y, train_fraction = 0.8)
  expect_true(is.list(result))
  expect_true("score" %in% names(result))
  expect_true("train_indices" %in% names(result))
  expect_true("test_indices" %in% names(result))
})

test_that("simple_cv train/test indices are disjoint and cover all data", {
  set.seed(42)
  data <- make_step_data(n = 100)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 3)
  result <- simple_cv(model_fn, data$x, data$y, train_fraction = 0.8)
  all_idx <- sort(c(result$train_indices, result$test_indices))
  expect_equal(all_idx, 1:100)
  expect_equal(length(result$train_indices), 80)
})

test_that("simple_cv input validation", {
  data <- make_step_data(n = 10)
  model_fn <- function(x, y) DecisionTreeRegressor(x, y, max_depth = 2,
                                                     min_leaf_size = 1)
  expect_error(simple_cv(model_fn, data$x, data$y, train_fraction = 0),
               "train_fraction")
  expect_error(simple_cv(model_fn, data$x, data$y, train_fraction = 1),
               "train_fraction")
})
