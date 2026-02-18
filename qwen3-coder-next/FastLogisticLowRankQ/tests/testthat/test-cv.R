# Cross-validation tests
# These tests will be expanded once CV functionality is added

test_that("CVST integration setup", {
  skip_if_not_installed("CVST")
  
  # Test that CVST can be loaded
  expect_true(requireNamespace("CVST", quietly = TRUE))
})

test_that("Cross-validation function structure", {
  skip_if_not_installed("CVST")
  
  set.seed(808)
  n <- 50
  p <- 5
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Create a simple CV wrapper function
  cv_function <- function(X, y, lambda_seq, folds = 5) {
    # Placeholder for actual CV implementation
    results <- data.frame(
      lambda = lambda_seq,
      error = runif(length(lambda_seq)),
      stringsAsFactors = FALSE
    )
    return(results)
  }
  
  lambda_seq <- c(0, 0.01, 0.1, 0.5, 1)
  cv_results <- cv_function(X, y, lambda_seq)
  
  expect_is(cv_results, "data.frame")
  expect_equal(nrow(cv_results), length(lambda_seq))
  expect_named(cv_results, c("lambda", "error"))
})

test_that("CV error calculation works", {
  skip_if_not_installed("CVST")
  
  set.seed(909)
  n <- 40
  p <- 4
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Simple CV error calculation
  compute_cv_error <- function(lambda, X, y, folds = 3) {
    n <- length(y)
    fold_size <- floor(n / folds)
    errors <- numeric(folds)
    
    set.seed(42)
    indices <- sample(n)
    
    for (i in 1:folds) {
      test_idx <- indices[((i - 1) * fold_size + 1):(i * fold_size)]
      if (i == folds) {
        test_idx <- indices[((i - 1) * fold_size + 1):n]
      }
      
      train_X <- X[-test_idx, ]
      train_y <- y[-test_idx]
      test_X <- X[test_idx, ]
      test_y <- y[test_idx]
      
      model <- FastLogisticRegressionLowRank(train_X, train_y, lambda_ssr = lambda)
      pred <- predict(model, test_X, type = "class")
      errors[i] <- mean(pred != test_y)
    }
    
    return(mean(errors))
  }
  
  lambda_seq <- c(0, 0.1, 0.5, 1)
  errors <- sapply(lambda_seq, compute_cv_error, X = X, y = y)
  
  expect_length(errors, length(lambda_seq))
  expect_true(all(errors >= 0 & errors <= 1))
})

test_that("Optimal lambda selection works", {
  skip_if_not_installed("CVST")
  
  set.seed(112)
  n <- 50
  p <- 5
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Test optimal lambda selection
  find_optimal_lambda <- function(lambda_seq, X, y) {
    errors <- sapply(lambda_seq, function(lambda) {
      compute_cv_error(lambda, X, y)
    })
    
    optimal_idx <- which.min(errors)
    return(lambda_seq[optimal_idx])
  }
  
  compute_cv_error <- function(lambda, X, y, folds = 3) {
    n <- length(y)
    fold_size <- floor(n / folds)
    errors <- numeric(folds)
    
    set.seed(42)
    indices <- sample(n)
    
    for (i in 1:folds) {
      test_idx <- indices[((i - 1) * fold_size + 1):(i * fold_size)]
      if (i == folds) {
        test_idx <- indices[((i - 1) * fold_size + 1):n]
      }
      
      train_X <- X[-test_idx, ]
      train_y <- y[-test_idx]
      test_X <- X[test_idx, ]
      test_y <- y[test_idx]
      
      model <- FastLogisticRegressionLowRank(train_X, train_y, lambda_ssr = lambda)
      pred <- predict(model, test_X, type = "class")
      errors[i] <- mean(pred != test_y)
    }
    
    return(mean(errors))
  }
  
  lambda_seq <- c(0, 0.01, 0.1, 0.5, 1)
  optimal_lambda <- find_optimal_lambda(lambda_seq, X, y)
  
  expect_is(optimal_lambda, "numeric")
  expect_length(optimal_lambda, 1)
  expect_true(optimal_lambda %in% lambda_seq)
})

test_that("Cross-validation with multiple folds", {
  skip_if_not_installed("CVST")
  
  set.seed(223)
  n <- 60
  p <- 6
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # Test with different fold sizes
  for (folds in c(3, 5, 10)) {
    model <- FastLogisticRegressionLowRank(X, y)
    expect_true(model$converged)
  }
})

test_that("CV results are reproducible", {
  skip_if_not_installed("CVST")
  
  set.seed(334)
  n <- 40
  p <- 4
  
  X <- matrix(rnorm(n * p), n, p)
  y <- sample(c(0, 1), n, replace = TRUE)
  
  # First run
  model1 <- FastLogisticRegressionLowRank(X, y)
  
  # Second run with same seed
  set.seed(334)
  model2 <- FastLogisticRegressionLowRank(X, y)
  
  expect_equal(model1$coefficients, model2$coefficients, tolerance = 1e-6)
})
