# Tests for input validation

test_that("non-binary y is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- c(0, 1, 2, 0, 1, 2, 0, 1, 2, 0)

  expect_error(FastLogisticRegressionLowRank(X, y), "binary")
})

test_that("y with values other than 0 and 1 is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- c(-1, 1, -1, 1, -1, 1, -1, 1, -1, 1)

  expect_error(FastLogisticRegressionLowRank(X, y), "binary")
})

test_that("mismatched X and y dimensions are rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- c(0, 1, 0, 1, 0)

  expect_error(FastLogisticRegressionLowRank(X, y), "nrow")
})

test_that("negative epsilon is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, epsilon = -1), "positive")
})

test_that("negative lambda_ssr is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, lambda_ssr = -0.5), "non-negative")
})

test_that("negative gamma is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, gamma = -0.1), "non-negative")
})

test_that("invalid energy_percentile is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, energy_percentile = 0), "energy_percentile")
  expect_error(FastLogisticRegressionLowRank(X, y, energy_percentile = 101), "energy_percentile")
  expect_error(FastLogisticRegressionLowRank(X, y, energy_percentile = -5), "energy_percentile")
})

test_that("negative convergence_tolerance is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, convergence_tolerance = -1e-3),
               "positive")
})

test_that("minimum_iteration < 1 is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, minimum_iteration = 0),
               "minimum_iteration")
})

test_that("maximum_iteration < minimum_iteration is rejected", {
  X <- matrix(rnorm(30), 10, 3)
  y <- rbinom(10, 1, 0.5)

  expect_error(FastLogisticRegressionLowRank(X, y, minimum_iteration = 5,
                                              maximum_iteration = 3),
               "maximum_iteration")
})
