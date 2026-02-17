test_that("grid_search returns correct structure", {
  set.seed(42)
  data <- make_step_data(n = 100)
  factory <- function(max_depth, min_leaf_size) {
    function(x, y) DecisionTreeRegressor(x, y, max_depth = max_depth,
                                          min_leaf_size = min_leaf_size)
  }
  result <- grid_search(factory, data$x, data$y,
                          param_grid = list(max_depth = c(2, 5),
                                            min_leaf_size = c(2, 5)),
                          k = 3)
  expect_true(is.list(result))
  expect_true("best_params" %in% names(result))
  expect_true("best_score" %in% names(result))
  expect_true("results" %in% names(result))
  expect_true(is.data.frame(result$results))
  expect_equal(nrow(result$results), 4)  # 2 x 2 grid
})

test_that("grid_search finds better params than worst", {
  set.seed(42)
  data <- make_sine_data(n = 200)
  factory <- function(max_depth) {
    function(x, y) DecisionTreeRegressor(x, y, max_depth = max_depth)
  }
  result <- grid_search(factory, data$x, data$y,
                          param_grid = list(max_depth = c(1, 3, 8)),
                          k = 3)
  # Best score should be <= worst score
  expect_lte(result$best_score, max(result$results$score))
})

test_that("random_search returns correct structure", {
  set.seed(42)
  data <- make_step_data(n = 100)
  factory <- function(max_depth, min_leaf_size) {
    function(x, y) DecisionTreeRegressor(x, y, max_depth = max_depth,
                                          min_leaf_size = min_leaf_size)
  }
  result <- random_search(factory, data$x, data$y,
                            param_distributions = list(
                              max_depth = function() sample(2:8, 1),
                              min_leaf_size = function() sample(2:10, 1)
                            ),
                            n_iter = 5, k = 3)
  expect_true(is.list(result))
  expect_equal(nrow(result$results), 5)
  expect_true("best_params" %in% names(result))
  expect_true("best_score" %in% names(result))
})

test_that("random_search best_score matches best in results", {
  set.seed(42)
  data <- make_sine_data(n = 100)
  factory <- function(max_depth) {
    function(x, y) DecisionTreeRegressor(x, y, max_depth = max_depth)
  }
  result <- random_search(factory, data$x, data$y,
                            param_distributions = list(
                              max_depth = function() sample(1:10, 1)
                            ),
                            n_iter = 8, k = 3)
  expect_equal(result$best_score, min(result$results$score))
})
