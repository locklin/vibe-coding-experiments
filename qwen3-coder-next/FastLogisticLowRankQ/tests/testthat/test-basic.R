test_that("Basic model fitting works", {
  set.seed(123)
  n <- 100
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  
  expect_s3_class(model, "FastLogisticRegressionLowRank")
  expect_true(model$converged)
  expect_is(model$coefficients, "numeric")
  expect_length(model$coefficients, p + 1)
  expect_is(model$rank, "integer")
  expect_is(model$n_iterations, "integer")
})

test_that("Model with intercept disabled works", {
  set.seed(456)
  n <- 50
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y, fit_intercept = FALSE)
  
  expect_length(model$coefficients, p)
  expect_null(model$intercept)
})

test_that("Predict method returns correct types", {
  set.seed(789)
  n <- 80
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  
  # Test class predictions
  class_pred <- predict(model, X, type = "class")
  expect_is(class_pred, "numeric")
  expect_length(class_pred, n)
  expect_true(all(class_pred %in% c(0, 1)))
  
  # Test probability predictions
  prob_pred <- predict(model, X, type = "prob")
  expect_is(prob_pred, "numeric")
  expect_length(prob_pred, n)
  expect_true(all(prob_pred >= 0 & prob_pred <= 1))
})

test_that("Prediction with new data works", {
  set.seed(111)
  n_train <- 100
  n_test <- 30
  p <- 6
  X_train <- matrix(rnorm(n_train * p), n_train, p)
  y_train <- sample(c(0, 1), n_train, replace = TRUE)
  X_test <- matrix(rnorm(n_test * p), n_test, p)
  
  model <- FastLogisticRegressionLowRank(X_train, y_train)
  
  pred <- predict(model, X_test)
  expect_length(pred, n_test)
})

test_that("Different regularization parameters work", {
  set.seed(222)
  n <- 60
  p <- 8
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Test with L2 regularization
  model1 <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.1)
  expect_true(model1$converged)
  
  # Test with L1-like regularization
  model2 <- FastLogisticRegressionLowRank(X, y, f = 0.5)
  expect_true(model2$converged)
  
  # Test with elastic net-like regularization
  model3 <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.05, gamma = 0.05)
  expect_true(model3$converged)
})

test_that("Energy percentile affects rank", {
  set.seed(333)
  n <- 100
  p <- 20
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model_low <- FastLogisticRegressionLowRank(X, y, energyPercentile = 90)
  model_high <- FastLogisticRegressionLowRank(X, y, energyPercentile = 99.9)
  
  expect_true(model_high$rank >= model_low$rank)
})

test_that("Convergence tolerance works", {
  set.seed(444)
  n <- 80
  p <- 10
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Tighter tolerance should require more iterations
  model_tight <- FastLogisticRegressionLowRank(X, y, convergenceTolerance = 1e-6)
  model_loose <- FastLogisticRegressionLowRank(X, y, convergenceTolerance = 1e-2)
  
  expect_gte(model_tight$n_iterations, model_loose$n_iterations)
})

test_that("Maximum iterations enforced", {
  set.seed(555)
  n <- 50
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Very tight tolerance to force max iterations
  model <- FastLogisticRegressionLowRank(X, y, convergenceTolerance = 1e-12, maximumIteration = 3)
  
  expect_lte(model$n_iterations, 4)  # 0-indexed + 1
})

test_that("Input validation works", {
  set.seed(666)
  n <- 50
  p <- 5
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Test mismatched dimensions
  expect_error(FastLogisticRegressionLowRank(X, y[1:25]), "Length of y must match")
  
  # Test non-binary y
  y_nonbinary <- sample(c(0, 1, 2), n, replace = TRUE)
  expect_error(FastLogisticRegressionLowRank(X, y_nonbinary), "y must contain only 0 and 1 values")
})

test_that("Print method works", {
  set.seed(777)
  n <- 30
  p <- 4
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  
  # Just test that print doesn't error
  expect_output(print(model), "Fast Logistic Regression")
})
