# Tests for predict.FastLogisticLowRank

test_that("predict returns correct types for each prediction type", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y)

  probs <- predict(fit, X, type = "response")
  classes <- predict(fit, X, type = "class")
  link <- predict(fit, X, type = "link")

  expect_type(probs, "double")
  expect_type(classes, "integer")
  expect_type(link, "double")

  expect_length(probs, n)
  expect_length(classes, n)
  expect_length(link, n)
})

test_that("predicted probabilities are in [0, 1]", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  probs <- predict(fit, X, type = "response")

  expect_true(all(probs >= 0))
  expect_true(all(probs <= 1))
})

test_that("predicted classes are 0 or 1", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)
  classes <- predict(fit, X, type = "class")

  expect_true(all(classes %in% c(0, 1)))
})

test_that("threshold parameter works for class predictions", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y)

  # Low threshold -> more 1s
  classes_low <- predict(fit, X, type = "class", threshold = 0.2)
  # High threshold -> fewer 1s
  classes_high <- predict(fit, X, type = "class", threshold = 0.8)

  expect_gte(sum(classes_low == 1), sum(classes_high == 1))
})

test_that("link and response predictions are consistent", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y)

  probs <- predict(fit, X, type = "response")
  link <- predict(fit, X, type = "link")

  # Manual sigmoid of link should match response
  manual_prob <- exp(pmin(link, 100)) / (1 + exp(pmin(link, 100)))
  expect_equal(probs, manual_prob, tolerance = 1e-10)
})

test_that("prediction on new data works", {
  set.seed(42)
  n_train <- 200
  n_test <- 50
  p <- 5
  X_train <- matrix(rnorm(n_train * p), n_train, p)
  X_test <- matrix(rnorm(n_test * p), n_test, p)
  beta_true <- c(1, -0.5, 0.3, 0, 0.8)
  prob_train <- 1 / (1 + exp(-X_train %*% beta_true))
  y_train <- rbinom(n_train, 1, prob_train)

  fit <- FastLogisticRegressionLowRank(X_train, y_train)
  probs <- predict(fit, X_test, type = "response")

  expect_length(probs, n_test)
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("prediction without intercept works", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y, fit_intercept = FALSE)
  probs <- predict(fit, X, type = "response")
  classes <- predict(fit, X, type = "class")

  expect_length(probs, n)
  expect_true(all(probs >= 0 & probs <= 1))
  expect_true(all(classes %in% c(0, 1)))
})

test_that("predict handles extreme linear predictor values", {
  set.seed(42)
  n <- 100
  p <- 2
  # Create data with large separation to produce large linear predictors
  X <- rbind(matrix(rnorm(50 * p, mean = 5), 50, p),
             matrix(rnorm(50 * p, mean = -5), 50, p))
  y <- c(rep(1, 50), rep(0, 50))

  fit <- FastLogisticRegressionLowRank(X, y)
  probs <- predict(fit, X, type = "response")

  # No NaN or Inf
  expect_true(all(is.finite(probs)))
  expect_true(all(probs >= 0 & probs <= 1))
})

test_that("default predict type is 'response'", {
  set.seed(42)
  n <- 100
  p <- 3
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)

  # Default should be response
  default_pred <- predict(fit, X)
  response_pred <- predict(fit, X, type = "response")

  expect_equal(default_pred, response_pred)
})

test_that("single observation prediction works", {
  set.seed(42)
  n <- 200
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- rbinom(n, 1, 0.5)

  fit <- FastLogisticRegressionLowRank(X, y)

  # Predict for a single observation
  X_single <- matrix(rnorm(p), 1, p)
  prob <- predict(fit, X_single, type = "response")
  cls <- predict(fit, X_single, type = "class")

  expect_length(prob, 1)
  expect_length(cls, 1)
})

test_that("model achieves reasonable accuracy on structured data", {
  set.seed(123)
  n <- 500
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  beta_true <- c(1.5, -1, 0.8, -0.3, 0.5, rep(0, 5))
  prob <- 1 / (1 + exp(-X %*% beta_true))
  y <- rbinom(n, 1, prob)

  fit <- FastLogisticRegressionLowRank(X, y)
  preds <- predict(fit, X, type = "class")
  accuracy <- mean(preds == y)

  expect_gt(accuracy, 0.65)
})
