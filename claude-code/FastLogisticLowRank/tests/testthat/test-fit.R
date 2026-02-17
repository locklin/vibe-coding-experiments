# Tests for FastLogisticRegressionLowRank fitting

test_that("basic fitting works on simple data", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y)

  expect_s3_class(fit, "FastLogisticLowRank")
  expect_length(fit$coefficients, p)
  expect_true(!is.null(fit$intercept))
  expect_true(fit$rank >= 1)
  expect_true(fit$n_iterations >= 1)
  expect_true(is.logical(fit$converged))
})

test_that("fitted model returns named coefficients", {
  set.seed(1)
  X <- matrix(rnorm(100 * 3), 100, 3)
  colnames(X) <- c("x1", "x2", "x3")
  y <- rbinom(100, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_equal(names(fit$coefficients), c("x1", "x2", "x3"))
})

test_that("fitting without colnames generates V1, V2, ... names", {
  set.seed(1)
  X <- matrix(rnorm(100 * 3), 100, 3)
  y <- rbinom(100, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_equal(names(fit$coefficients), c("V1", "V2", "V3"))
})

test_that("fit_intercept = FALSE works", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y, fit_intercept = FALSE)

  expect_null(fit$intercept)
  expect_length(fit$coefficients, p)
  expect_false(fit$params$fit_intercept)
})

test_that("model converges on linearly separable data", {
  set.seed(10)
  n <- 300
  X <- matrix(rnorm(n * 2), n, 2)
  y <- as.integer(X[, 1] + X[, 2] > 0)

  fit <- FastLogisticRegressionLowRank(X, y, maximum_iteration = 20)

  # Model should achieve reasonable accuracy on well-separated data
  preds <- predict(fit, X, type = "class")
  accuracy <- mean(preds == y)
  expect_gt(accuracy, 0.65)
})

test_that("L2 regularization (lambda_ssr) produces smaller coefficients", {
  set.seed(42)
  n <- 200
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(2, -1, 0.5, rep(0, 7))
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit_unreg <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0)
  fit_reg <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 1.0)

  # Regularized coefficients should have smaller L2 norm
  norm_unreg <- sqrt(sum(fit_unreg$coefficients^2))
  norm_reg <- sqrt(sum(fit_reg$coefficients^2))
  expect_lt(norm_reg, norm_unreg)
})

test_that("gamma regularization works", {
  set.seed(42)
  n <- 200
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(2, -1, 0.5, rep(0, 7))
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit_unreg <- FastLogisticRegressionLowRank(X, y)
  fit_reg <- FastLogisticRegressionLowRank(X, y, gamma = 1.0)

  # Regularized coefficients should have smaller norm
  norm_unreg <- sqrt(sum(fit_unreg$coefficients^2))
  norm_reg <- sqrt(sum(fit_reg$coefficients^2))
  expect_lt(norm_reg, norm_unreg)
})

test_that("combined lambda_ssr and gamma regularization works", {
  set.seed(42)
  n <- 200
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(2, -1, 0.5, rep(0, 7))
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.5, gamma = 0.5, f = 1)

  expect_s3_class(fit, "FastLogisticLowRank")
  expect_length(fit$coefficients, p)
})

test_that("convergence_tolerance affects number of iterations", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit_loose <- FastLogisticRegressionLowRank(X, y, convergence_tolerance = 0.1,
                                              minimum_iteration = 1)
  fit_tight <- FastLogisticRegressionLowRank(X, y, convergence_tolerance = 1e-8,
                                              minimum_iteration = 1,
                                              maximum_iteration = 50)

  # Tight tolerance should use at least as many iterations
  expect_gte(fit_tight$n_iterations, fit_loose$n_iterations)
})

test_that("maximum_iteration limits iterations", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y, maximum_iteration = 3,
                                        convergence_tolerance = 1e-15)

  # Should not exceed maximum_iteration + 1 (0-indexed loop)
  expect_lte(fit$n_iterations, 4)
})

test_that("energy_percentile affects rank", {
  set.seed(42)
  n <- 200
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit_low <- FastLogisticRegressionLowRank(X, y, energy_percentile = 50)
  fit_high <- FastLogisticRegressionLowRank(X, y, energy_percentile = 99.9999)

  # Higher percentile should give higher or equal rank
  expect_gte(fit_high$rank, fit_low$rank)
})

test_that("params are stored correctly in result", {
  set.seed(1)
  X <- matrix(rnorm(100 * 3), 100, 3)
  y <- rbinom(100, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.5, gamma = 0.1,
                                        f = 0.5, epsilon = 1e-8,
                                        energy_percentile = 99.0,
                                        convergence_tolerance = 1e-4,
                                        minimum_iteration = 3,
                                        maximum_iteration = 15,
                                        fit_intercept = FALSE)

  expect_equal(fit$params$lambda_ssr, 0.5)
  expect_equal(fit$params$gamma, 0.1)
  expect_equal(fit$params$f, 0.5)
  expect_equal(fit$params$epsilon, 1e-8)
  expect_equal(fit$params$energy_percentile, 99.0)
  expect_equal(fit$params$convergence_tolerance, 1e-4)
  expect_equal(fit$params$minimum_iteration, 3L)
  expect_equal(fit$params$maximum_iteration, 15L)
  expect_false(fit$params$fit_intercept)
})

test_that("single predictor works", {
  set.seed(42)
  n <- 100
  X <- matrix(rnorm(n), n, 1)
  y <- as.integer(X[, 1] > 0)

  fit <- FastLogisticRegressionLowRank(X, y)
  expect_s3_class(fit, "FastLogisticLowRank")
  expect_length(fit$coefficients, 1)
})

test_that("data.frame input is coerced to matrix", {
  set.seed(42)
  n <- 100
  df <- data.frame(x1 = rnorm(n), x2 = rnorm(n))
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(as.matrix(df), y)
  expect_s3_class(fit, "FastLogisticLowRank")
})

test_that("print method works without error", {
  set.seed(1)
  X <- matrix(rnorm(100 * 3), 100, 3)
  y <- rbinom(100, 1, 0.5)
  fit <- FastLogisticRegressionLowRank(X, y)

  expect_output(print(fit), "Fast Logistic Regression")
  expect_output(print(fit), "Coefficients")
})

test_that("summary method works without error", {
  set.seed(1)
  X <- matrix(rnorm(100 * 3), 100, 3)
  y <- rbinom(100, 1, 0.5)
  fit <- FastLogisticRegressionLowRank(X, y)

  expect_output(summary(fit), "Fast Logistic Regression")
  expect_output(summary(fit), "Regularization")
})
