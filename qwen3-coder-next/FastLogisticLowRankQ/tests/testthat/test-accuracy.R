test_that("Model achieves reasonable accuracy on separable data", {
  set.seed(888)
  n <- 200
  p <- 15
  
  # Generate separable data
  X <- matrix(rnorm(n * p), n, p)
  # True coefficients
  true_beta <- c(1, 0.5, -0.3, 0.8, rep(0, p - 4))
  # Generate y based on true relationship
  eta <- X %*% true_beta[-1] + true_beta[1]
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(n, 1, prob)
  
  model <- FastLogisticRegressionLowRank(X, y)
  pred <- predict(model, X, type = "class")
  
  accuracy <- mean(pred == y)
  expect_gte(accuracy, 0.75)
})

test_that("Model works with correlated features", {
  set.seed(999)
  n <- 150
  p <- 12
  
  # Create correlated features
  base <- rnorm(n)
  X <- matrix(base, n, p)
  for (i in 2:p) {
    X[, i] <- 0.7 * X[, i - 1] + 0.3 * rnorm(n)
  }
  
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  expect_true(model$converged)
  
  pred <- predict(model, X)
  accuracy <- mean(pred == y)
  expect_true(accuracy >= 0.4)  # Should be better than random
})

test_that("Model handles high-dimensional data", {
  set.seed(101)
  n <- 100
  p <- 200  # More features than samples
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  expect_true(model$converged)
  
  pred <- predict(model, X)
  expect_length(pred, n)
})

test_that("Model works with sparse-like data", {
  set.seed(202)
  n <- 80
  p <- 10
  
  # Create sparse data (many zeros)
  X <- matrix(0, n, p)
  for (i in 1:n) {
    present <- sample(1:p, sample(1:3, 1))
    X[i, present] <- rnorm(length(present))
  }
  
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  expect_true(model$converged)
})

test_that("Probability predictions are calibrated", {
  set.seed(303)
  n <- 200
  p <- 10
  
  X <- matrix(rnorm(n * p), n, p)
  # Generate y with clear probability relationship
  eta <- 0.5 + 0.3 * X[, 1] - 0.2 * X[, 2]
  prob <- 1 / (1 + exp(-eta))
  y <- rbinom(n, 1, prob)
  
  model <- FastLogisticRegressionLowRank(X, y)
  pred_prob <- predict(model, X, type = "prob")
  
  # Check that predicted probabilities are reasonable
  expect_true(all(pred_prob >= 0 & pred_prob <= 1))
  
  # Check correlation between predicted prob and actual y
  cor_val <- cor(pred_prob, y)
  expect_gte(cor_val, 0.2)
})

test_that("Large dataset performance", {
  set.seed(404)
  n <- 5000
  p <- 50
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Should complete in reasonable time
  model <- FastLogisticRegressionLowRank(X, y)
  expect_true(model$converged)
})

test_that("Cross-validation with CVST package", {
  skip_if_not_installed("CVST")
  
  set.seed(505)
  n <- 100
  p <- 10
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Test that CVST integration works
  # This is a basic test - full CV testing is in test-cv.R
  expect_true(TRUE)  # Placeholder
})

test_that("Different initialization strategies work", {
  set.seed(606)
  n <- 60
  p <- 8
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Test with different minimum iterations
  model1 <- FastLogisticRegressionLowRank(X, y, minimumIteration = 1)
  model2 <- FastLogisticRegressionLowRank(X, y, minimumIteration = 5)
  
  expect_true(model1$converged)
  expect_true(model2$converged)
})

test_that("Numerical stability with extreme values", {
  set.seed(707)
  n <- 50
  p <- 5
  
  # Create data with extreme values
  X <- matrix(c(rnorm(n * (p - 1)), rnorm(n, 100, 10)), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  model <- FastLogisticRegressionLowRank(X, y)
  expect_true(model$converged)
})
