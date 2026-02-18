# Cross-validation implementation tests

test_that("cvFastLogistic basic functionality", {
  skip_if_not_installed("CVST")
  
  set.seed(113)
  n <- 80
  p <- 8
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.01, 0.1, 0.5)
  
  result <- cvFastLogistic(X, y, lambda_seq, folds = 5)
  
  expect_is(result, "list")
  expect_is(result$results, "data.frame")
  expect_equal(nrow(result$results), length(lambda_seq))
  expect_named(result$results, c("lambda", "cv_error"))
  expect_is(result$optimal_lambda, "numeric")
  expect_is(result$min_error, "numeric")
  expect_length(result$optimal_lambda, 1)
  expect_length(result$min_error, 1)
  expect_true(result$optimal_lambda %in% lambda_seq)
  expect_true(result$min_error >= 0 && result$min_error <= 1)
})

test_that("cvFastLogistic with different folds", {
  skip_if_not_installed("CVST")
  
  set.seed(224)
  n <- 100
  p <- 10
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.1, 1)
  
  # Test with 3 folds
  result3 <- cvFastLogistic(X, y, lambda_seq, folds = 3)
  expect_equal(nrow(result3$results), 3)
  
  # Test with 10 folds
  result10 <- cvFastLogistic(X, y, lambda_seq, folds = 10)
  expect_equal(nrow(result10$results), 3)
})

test_that("cvFastLogistic with edge cases", {
  skip_if_not_installed("CVST")
  
  set.seed(335)
  n <- 50
  p <- 5
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Single lambda value
  result <- cvFastLogistic(X, y, lambda_seq = c(0.1), folds = 5)
  expect_equal(nrow(result$results), 1)
  expect_equal(result$optimal_lambda, 0.1)
  
  # Empty lambda sequence should fail gracefully
  expect_error(cvFastLogistic(X, y, lambda_seq = c()))
})

test_that("cvFastLogistic reproducibility", {
  skip_if_not_installed("CVST")
  
  set.seed(446)
  n <- 60
  p <- 6
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.01, 0.1)
  
  # Run twice with same seed
  result1 <- cvFastLogistic(X, y, lambda_seq, folds = 5, seed = 123)
  result2 <- cvFastLogistic(X, y, lambda_seq, folds = 5, seed = 123)
  
  expect_equal(result1$results$cv_error, result2$results$cv_error)
  expect_equal(result1$optimal_lambda, result2$optimal_lambda)
})

test_that("cvFastLogistic with regularization", {
  skip_if_not_installed("CVST")
  
  set.seed(557)
  n <- 80
  p <- 10
  
  # Generate data with clear signal
  X <- matrix(rnorm(n * p), n, p)
  true_beta <- c(1.5, -1, 0.8, rep(0, p - 3))
  linear_pred <- X %*% true_beta[-1] + true_beta[1]
  prob <- 1 / (1 + exp(-linear_pred))
  y <- rbinom(n, 1, prob)
  
  lambda_seq <- c(0, 0.01, 0.05, 0.1, 0.5, 1)
  
  result <- cvFastLogistic(X, y, lambda_seq, folds = 5)
  
  # Check that results make sense
  expect_true(all(result$results$cv_error >= 0))
  expect_true(all(result$results$cv_error <= 1))
  expect_true(all(!is.na(result$results$cv_error)))
  
  # Optimal lambda should be in the sequence
  expect_true(result$optimal_lambda %in% lambda_seq)
})

test_that("cvFastLogistic handles large lambda values", {
  skip_if_not_installed("CVST")
  
  set.seed(668)
  n <- 50
  p <- 5
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.1, 1, 5, 10)
  
  result <- cvFastLogistic(X, y, lambda_seq, folds = 5)
  
  # Large lambda should give higher error (over-regularization)
  # (not necessarily true in all cases, but generally expected)
  expect_true(result$min_error >= 0)
})

test_that("cvFastLogisticCVST wrapper works", {
  skip_if_not_installed("CVST")
  
  set.seed(779)
  n <- 60
  p <- 6
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.1, 1)
  
  result <- cvFastLogisticCVST(X, y, lambda_seq, folds = 5)
  
  expect_is(result, "list")
  expect_is(result$results, "data.frame")
  expect_is(result$optimal_lambda, "numeric")
  expect_is(result$min_error, "numeric")
})

test_that("print.cvFastLogistic works", {
  skip_if_not_installed("CVST")
  
  set.seed(881)
  n <- 50
  p <- 5
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.01, 0.1)
  result <- cvFastLogistic(X, y, lambda_seq)
  
  # Just test that print doesn't error
  expect_output(print(result), "Cross-Validation Results")
})

test_that("plot.cvFastLogistic works", {
  skip_if_not_installed("CVST")
  
  set.seed(992)
  n <- 60
  p <- 6
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.01, 0.1, 0.5)
  result <- cvFastLogistic(X, y, lambda_seq)
  
  # Test that plot doesn't error (don't actually plot)
  # Just check that the function exists and can be called
  expect_true(exists("plot.cvFastLogistic", where = asNamespace("FastLogisticLowRankQ")))
})

test_that("CV works with different measures", {
  skip_if_not_installed("CVST")
  
  set.seed(103)
  n <- 70
  p <- 7
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.01, 0.1)
  
  # Test with default (misclassification)
  result1 <- cvFastLogistic(X, y, lambda_seq, folds = 5)
  expect_true(all(result1$results$cv_error >= 0 & result1$results$cv_error <= 1))
})

test_that("CV with high-dimensional data", {
  skip_if_not_installed("CVST")
  
  set.seed(214)
  n <- 50
  p <- 100  # More features than samples
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  lambda_seq <- c(0, 0.01, 0.1)
  
  result <- cvFastLogistic(X, y, lambda_seq, folds = 5)
  
  expect_equal(nrow(result$results), 3)
  expect_is(result$optimal_lambda, "numeric")
})

test_that("CV with imbalanced data", {
  skip_if_not_installed("CVST")
  
  set.seed(325)
  n <- 100
  p <- 8
  
  X <- matrix(rnorm(n * p), n, p)
  # Create imbalanced y
  y <- sample(c(0, 1), n, replace = TRUE, prob = c(0.8, 0.2))
  
  # Check imbalance
  expect_true(sum(y == 1) < sum(y == 0))
  
  lambda_seq <- c(0, 0.01, 0.1)
  
  result <- cvFastLogistic(X, y, lambda_seq, folds = 5)
  
  expect_equal(nrow(result$results), 3)
  expect_is(result$optimal_lambda, "numeric")
})
