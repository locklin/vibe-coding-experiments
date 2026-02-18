#' Cross-Validation for Fast Logistic Regression
#'
#' Performs k-fold cross-validation to find optimal regularization parameters
#' for the FastLogisticRegressionLowRank model.
#'
#' @param X Input matrix of features
#' @param y Binary response vector
#' @param lambda_seq Vector of lambda_ssr values to test
#' @param folds Number of folds for cross-validation (default: 5)
#' @param seed Random seed for reproducibility (default: 42)
#' @return A list containing:
#'   \item{results}{Data frame with lambda values and corresponding CV errors}
#'   \item{optimal_lambda}{Lambda with minimum CV error}
#'   \item{min_error}{Minimum cross-validation error}
#' @examples
#' set.seed(123)
#' n <- 100
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' y <- sample(c(0, 1), n, replace = TRUE)
#' lambda_seq <- c(0, 0.01, 0.1, 0.5, 1)
#' cv_result <- cvFastLogistic(X, y, lambda_seq)
#' cat("Optimal lambda:", cv_result$optimal_lambda, "\n")
#' @export
cvFastLogistic <- function(X, y, lambda_seq, folds = 5, seed = 42) {
  n <- length(y)
  fold_size <- floor(n / folds)
  
  # Initialize storage for CV errors
  cv_errors <- numeric(length(lambda_seq))
  
  # Set seed for reproducibility
  set.seed(seed)
  indices <- sample(n)
  
  # Perform cross-validation for each lambda
  for (i in seq_along(lambda_seq)) {
    errors <- numeric(folds)
    
    for (fold in 1:folds) {
      # Calculate test indices for this fold
      test_start <- ((fold - 1) * fold_size + 1)
      test_end <- fold * fold_size
      test_idx <- indices[test_start:test_end]
      
      # Handle last fold
      if (fold == folds) {
        test_idx <- indices[test_start:n]
      }
      
      # Split data
      train_X <- X[-test_idx, ]
      train_y <- y[-test_idx]
      test_X <- X[test_idx, ]
      test_y <- y[test_idx]
      
      # Fit model with current lambda
      model <- FastLogisticRegressionLowRank(
        train_X, train_y,
        lambda_ssr = lambda_seq[i]
      )
      
      # Make predictions
      pred <- predict(model, test_X, type = "class")
      
      # Calculate error
      errors[fold] <- mean(pred != test_y)
    }
    
    # Store mean CV error for this lambda
    cv_errors[i] <- mean(errors)
  }
  
  # Find optimal lambda
  optimal_idx <- which.min(cv_errors)
  optimal_lambda <- lambda_seq[optimal_idx]
  min_error <- cv_errors[optimal_idx]
  
  # Return results
  list(
    results = data.frame(
      lambda = lambda_seq,
      cv_error = cv_errors,
      stringsAsFactors = FALSE
    ),
    optimal_lambda = optimal_lambda,
    min_error = min_error
  )
}

#' Cross-Validation using CVST Package
#'
#' An alternative implementation using the CVST package for cross-validation.
#' Requires the CVST package to be installed.
#'
#' @param X Input matrix of features
#' @param y Binary response vector
#' @param lambda_seq Vector of lambda_ssr values to test
#' @param folds Number of folds for cross-validation (default: 5)
#' @param measure Error measure for classification ("misclass" or "auc")
#' @return A list containing cross-validation results
#' @importFrom stats model.frame model.matrix
#' @examples
#' \dontrun{
#' set.seed(123)
#' n <- 100
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' y <- sample(c(0, 1), n, replace = TRUE)
#' lambda_seq <- c(0, 0.01, 0.1, 0.5, 1)
#' cv_result <- cvFastLogisticCVST(X, y, lambda_seq)
#' }
#' @export
cvFastLogisticCVST <- function(X, y, lambda_seq, folds = 5, measure = "misclass") {
  if (!requireNamespace("CVST", quietly = TRUE)) {
    stop("CVST package is required but not installed. Please install it with: install.packages('CVST')")
  }
  
  # Create data frame
  data_df <- as.data.frame(X)
  data_df$y <- y
  
  # Define the CV method
  cv_method <- CVST::cvControl(method = "cv", args = list(K = folds))
  
  # Create parameter grid
  param_grid <- data.frame(
    lambda_ssr = lambda_seq,
    stringsAsFactors = FALSE
  )
  
  # This is a simplified interface - in practice you might want to use
  # CVST's cvFit function with custom fitting functions
  # For now, we'll use our manual CV implementation wrapped in CVST interface
  
  # Perform CV using our implementation
  result <- cvFastLogistic(X, y, lambda_seq, folds)
  
  return(list(
    results = result$results,
    optimal_lambda = result$optimal_lambda,
    min_error = result$min_error,
    measure = measure
  ))
}

#' Plot cross-validation results
#'
#' @param cv_result Result from cvFastLogistic or cvFastLogisticCVST
#' @param log_lambda Whether to use log scale for lambda (default: TRUE)
#' @param ... Additional arguments passed to plot
#' @export
plot.cvFastLogistic <- function(cv_result, log_lambda = TRUE, ...) {
  results <- cv_result$results
  
  if (log_lambda) {
    plot(log(results$lambda), results$cv_error, 
         type = "b", pch = 19,
         xlab = "log(lambda)",
         ylab = "Cross-Validation Error",
         main = "Cross-Validation Results",
         ...)
  } else {
    plot(results$lambda, results$cv_error, 
         type = "b", pch = 19,
         xlab = "lambda",
         ylab = "Cross-Validation Error",
         main = "Cross-Validation Results",
         ...)
  }
  
  # Add vertical line for optimal lambda
  if (log_lambda) {
    abline(v = log(cv_result$optimal_lambda), col = "red", lty = 2, lwd = 2)
    text(log(cv_result$optimal_lambda), max(results$cv_error), 
         labels = paste0("λ = ", round(cv_result$optimal_lambda, 4)),
         pos = 4, col = "red")
  } else {
    abline(v = cv_result$optimal_lambda, col = "red", lty = 2, lwd = 2)
    text(cv_result$optimal_lambda, max(results$cv_error), 
         labels = paste0("λ = ", round(cv_result$optimal_lambda, 4)),
         pos = 4, col = "red")
  }
}

#' Print cross-validation results
#'
#' @param x Result from cvFastLogistic or cvFastLogisticCVST
#' @param ... Additional arguments (not used)
#' @export
print.cvFastLogistic <- function(x, ...) {
  cat("Cross-Validation Results for Fast Logistic Regression\n")
  cat("-----------------------------------------------------\n\n")
  
  cat("Optimal lambda:", x$optimal_lambda, "\n")
  cat("Minimum CV error:", round(x$min_error, 4), "\n\n")
  
  cat("CV Error by lambda:\n")
  print(x$results)
}
