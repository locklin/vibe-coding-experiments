# Tests for CvFastLogisticLowRank

test_that("CvFastLogisticLowRank works with CV method", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 150
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  result <- CvFastLogisticLowRank(X, y,
    lambda_values = c(0.01, 0.1, 1.0),
    method = "CV", fold = 3, verbose = FALSE)

  expect_type(result, "list")
  expect_s3_class(result, "FastLogisticLowRankCV")
  expect_true("best_lambda" %in% names(result))
  expect_true("best_params" %in% names(result))
  expect_true("lambda_values" %in% names(result))
  expect_true("method" %in% names(result))
  expect_equal(result$method, "CV")
  expect_true(result$best_lambda %in% c(0.01, 0.1, 1.0))
})

test_that("CvFastLogisticLowRank works with fastCV method", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 150
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  result <- CvFastLogisticLowRank(X, y,
    lambda_values = c(0.01, 0.1, 1.0),
    method = "fastCV", verbose = FALSE)

  expect_s3_class(result, "FastLogisticLowRankCV")
  expect_equal(result$method, "fastCV")
  expect_true(result$best_lambda %in% c(0.01, 0.1, 1.0))
})

test_that("CvFastLogisticLowRank returns stored lambda_values", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 100
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  lambdas <- c(0.001, 0.01, 0.1, 1.0, 10.0)
  result <- CvFastLogisticLowRank(X, y, lambda_values = lambdas,
    method = "CV", fold = 3, verbose = FALSE)

  expect_equal(result$lambda_values, lambdas)
})

test_that("optimal lambda produces a valid model", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  result <- CvFastLogisticLowRank(X, y,
    lambda_values = c(0.01, 0.1, 1.0),
    method = "CV", fold = 3, verbose = FALSE)

  fit <- FastLogisticRegressionLowRank(X, y,
    lambda_ssr = result$best_lambda)

  expect_s3_class(fit, "FastLogisticLowRank")
  preds <- predict(fit, X, type = "class")
  expect_true(all(preds %in% c(0, 1)))
})

test_that("CvFastLogisticLowRank passes extra arguments", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 100
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  result <- CvFastLogisticLowRank(X, y,
    lambda_values = c(0.01, 0.1),
    method = "CV", fold = 3, verbose = FALSE,
    gamma = 0.1, fit_intercept = FALSE)

  expect_true(result$best_lambda %in% c(0.01, 0.1))
})

test_that("CvFastLogisticLowRank has correct function signature", {
  expect_true(is.function(CvFastLogisticLowRank))
  fn_args <- formals(CvFastLogisticLowRank)
  expect_true("lambda_values" %in% names(fn_args))
  expect_true("method" %in% names(fn_args))
  expect_true("fold" %in% names(fn_args))
})

test_that("cv result best_lambda is a scalar numeric", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 100
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  result <- CvFastLogisticLowRank(X, y,
    lambda_values = c(0.1, 1.0),
    method = "CV", fold = 3, verbose = FALSE)

  expect_type(result$best_lambda, "double")
  expect_length(result$best_lambda, 1)
})

test_that("print method works for FastLogisticLowRankCV", {
  skip_if_not_installed("CVST")

  set.seed(42)
  n <- 100
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  result <- CvFastLogisticLowRank(X, y,
    lambda_values = c(0.1, 1.0),
    method = "CV", fold = 3, verbose = FALSE)

  expect_output(print(result), "Cross-Validation Result")
  expect_output(print(result), "Best lambda")
})
