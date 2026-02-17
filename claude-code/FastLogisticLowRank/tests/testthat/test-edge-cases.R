# Tests for edge cases

test_that("works with many predictors relative to observations", {
  set.seed(42)
  n <- 50
  p <- 30
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
  expect_length(fit$coefficients, p)

  probs <- predict(fit, X, type = "response")
  expect_true(all(is.finite(probs)))
})

test_that("works with correlated predictors", {
  set.seed(42)
  n <- 200
  Z <- matrix(rnorm(n * 3), n, 3)
  # Create correlated columns
  X <- cbind(Z[, 1], Z[, 1] + rnorm(n, 0, 0.1),
             Z[, 2], Z[, 3], Z[, 3] + rnorm(n, 0, 0.1))
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
  expect_true(all(is.finite(fit$coefficients)))
})

test_that("works with all-zero predictor column", {
  set.seed(42)
  n <- 100
  X <- cbind(rnorm(n), rep(0, n), rnorm(n))
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
})

test_that("works with balanced classes", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rep(c(0, 1), each = n / 2)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
})

test_that("works with imbalanced classes", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- c(rep(0, 180), rep(1, 20))

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
  probs <- predict(fit, X, type = "response")
  expect_true(all(is.finite(probs)))
})

test_that("works with two observations per class (minimum viable)", {
  set.seed(42)
  n <- 10
  p <- 2
  X <- matrix(rnorm(n * p), n, p)
  y <- c(rep(0, 5), rep(1, 5))

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
})

test_that("works with standardized and unstandardized data", {
  set.seed(42)
  n <- 200
  p <- 5

  # Unstandardized with different scales
  X_raw <- cbind(rnorm(n, 100, 50), rnorm(n, 0, 0.01), rnorm(n, -5, 2),
                 rnorm(n), rnorm(n, 1000, 100))
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X_raw, y)
  expect_s3_class(fit, "FastLogisticLowRank")
  expect_true(all(is.finite(fit$coefficients)))
})

test_that("reproducibility: same data gives same results", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit1 <- FastLogisticRegressionLowRank(X, y)
  fit2 <- FastLogisticRegressionLowRank(X, y)

  expect_equal(fit1$coefficients, fit2$coefficients)
  expect_equal(fit1$intercept, fit2$intercept)
})
