test_that("SpillTreeKNN returns correct S3 class", {
  ref <- make_grid_data(n = 50, d = 2)
  model <- SpillTreeKNN(ref, k = 3)
  expect_s3_class(model, "SpillTreeKNN")
})

test_that("SpillTreeKNN neighbor matrix has correct dimensions", {
  ref <- make_grid_data(n = 50, d = 2)
  model <- SpillTreeKNN(ref, k = 5)
  expect_equal(dim(model$neighbors), c(50, 5))
  expect_equal(dim(model$distances), c(50, 5))
})

test_that("SpillTreeKNN with tau=0 gives exact nearest neighbors", {
  set.seed(42)
  # Simple 2D data.
  ref <- matrix(c(0, 0, 1, 0, 0, 1, 1, 1), ncol = 2, byrow = TRUE)
  model <- SpillTreeKNN(ref, k = 2, tau = 0.0)

  # For point (0,0), nearest neighbors should be (1,0) and (0,1) at dist 1.
  nn <- model$neighbors[1, ]
  dist <- model$distances[1, ]

  # First neighbor is itself (distance 0).
  expect_equal(dist[1], 0.0)
  expect_equal(nn[1], 1)  # 1-indexed
})

test_that("SpillTreeKNN self-distances are zero for first neighbor", {
  ref <- make_grid_data(n = 30, d = 2)
  model <- SpillTreeKNN(ref, k = 3, tau = 0.0)
  # First neighbor of each point should be itself with distance 0.
  expect_true(all(model$distances[, 1] < 1e-10))
})

test_that("SpillTreeKNN predict returns results for new query points", {
  ref <- make_grid_data(n = 50, d = 2)
  model <- SpillTreeKNN(ref, k = 3)
  set.seed(99)
  query <- matrix(rnorm(20), ncol = 2)
  result <- predict(model, query)
  expect_equal(dim(result$neighbors), c(10, 3))
  expect_equal(dim(result$distances), c(10, 3))
})

test_that("SpillTreeKNN distances are non-decreasing per row", {
  ref <- make_grid_data(n = 100, d = 3)
  model <- SpillTreeKNN(ref, k = 5, tau = 0.0)
  for (i in 1:nrow(model$distances)) {
    diffs <- diff(model$distances[i, ])
    expect_true(all(diffs >= -1e-10))
  }
})

test_that("SpillTreeKNN input validation", {
  ref <- make_grid_data(n = 10, d = 2)
  expect_error(SpillTreeKNN(ref, k = 0), "k must be >= 1")
  expect_error(SpillTreeKNN(ref, k = 20), "k must be <= number")
  expect_error(SpillTreeKNN(ref, k = 3, tau = 0.5), "tau must be")
  expect_error(SpillTreeKNN(ref, k = 3, tau = -0.1), "tau must be")
})

test_that("print.SpillTreeKNN runs without error", {
  ref <- make_grid_data(n = 30, d = 2)
  model <- SpillTreeKNN(ref, k = 3)
  expect_output(print(model), "SpillTreeKNN")
})

test_that("SpillTreeKNN with tau > 0 still returns valid results", {
  ref <- make_grid_data(n = 100, d = 2)
  model <- SpillTreeKNN(ref, k = 5, tau = 0.1)
  expect_equal(dim(model$neighbors), c(100, 5))
  # All distances should be non-negative.
  expect_true(all(model$distances >= 0))
})
