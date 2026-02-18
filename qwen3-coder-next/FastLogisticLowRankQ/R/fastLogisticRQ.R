#' Fast Logistic Regression with Low-Rank Approximation
#'
#' A fast implementation of binary logistic regression that uses low-rank
#' approximation and randomized SVD for efficient computation on large datasets.
#'
#' @param X Input matrix of features (n observations x p features)
#' @param y Binary response vector (n observations)
#' @param epsilon Numerical stability parameter (default: 1e-10)
#' @param lambda_ssr L2 regularization parameter (default: 0)
#' @param f L1 regularization exponent (default: 0)
#' @param gamma Elastic net mixing parameter (default: 0)
#' @param energyPercentile Energy threshold for rank determination (default: 99.9999)
#' @param convergenceTolerance Convergence threshold for iterations (default: 1e-3)
#' @param minimumIteration Minimum number of iterations (default: 2)
#' @param maximumIteration Maximum number of iterations (default: 10)
#' @param fit_intercept Whether to include intercept term (default: TRUE)
#' @param rank_override Override automatic rank determination (default: NULL)
#' @return A fitted model object of class 'FastLogisticRegressionLowRank'
#' @references Nurdan S. et al. (2023). Fast Binary Logistic Regression.
#' @examples
#' set.seed(123)
#' n <- 100
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' y <- sample(c(0, 1), n, replace = TRUE)
#' model <- FastLogisticRegressionLowRank(X, y)
#' predictions <- predict(model, X)
#' probabilities <- predict(model, X, type = "prob")
#' @export
FastLogisticRegressionLowRank <- function(X, y,
                                         epsilon = 1e-10,
                                         lambda_ssr = 0,
                                         f = 0,
                                         gamma = 0,
                                         energyPercentile = 99.9999,
                                         convergenceTolerance = 1e-3,
                                         minimumIteration = 2,
                                         maximumIteration = 10,
                                         fit_intercept = TRUE,
                                         rank_override = NULL) {
  # Input validation
  X <- as.matrix(X)
  y <- as.numeric(y)
  
  if (length(y) != nrow(X)) {
    stop("Length of y must match number of rows in X")
  }
  
  if (!all(y %in% c(0, 1))) {
    stop("y must contain only 0 and 1 values")
  }
  
  # Store parameters
  params <- list(
    epsilon = epsilon,
    lambda_ssr = lambda_ssr,
    f = f,
    gamma = gamma,
    energyPercentile = energyPercentile,
    convergenceTolerance = convergenceTolerance,
    minimumIteration = minimumIteration,
    maximumIteration = maximumIteration,
    fit_intercept = fit_intercept
  )
  
  # Fit the model
  result <- .fastLogisticRegressionLowRankInternal(
    X = X,
    y = y,
    params = params,
    rank_override = rank_override
  )
  
  # Return model object
  structure(
    list(
      coefficients = result$coefficients,
      intercept = if (fit_intercept) result$coefficients[1] else NULL,
      coef_no_intercept = if (fit_intercept) result$coefficients[-1] else result$coefficients,
      rank = result$rank,
      n_iterations = result$n_iterations,
      converged = result$converged,
      convergence_tolerance = convergenceTolerance,
      parameters = params
    ),
    class = "FastLogisticRegressionLowRank"
  )
}

#' Internal implementation of Fast Logistic Regression with Low-Rank Approximation
#'
#' @param X Input matrix
#' @param y Response vector
#' @param params List of parameters
#' @param rank_override Override rank (default: NULL)
#' @noRd
.fastLogisticRegressionLowRankInternal <- function(X, y, params, rank_override = NULL) {
  n <- nrow(X)
  d <- ncol(X) + if (params$fit_intercept) 1 else 0
  
  # Add intercept column if needed
  if (params$fit_intercept) {
    X <- cbind(1, X)
  }
  
  # Determine rank
  if (is.null(rank_override)) {
    rank <- .determineRank(X, params$energyPercentile)
  } else {
    rank <- rank_override
  }
  
  # Compute SVD
  svd_result <- tryCatch({
    if (n > 100000 && ncol(X) > 500) {
      # Use randomized SVD for large matrices
      library(Matrix, warn.conflicts = FALSE)
      if (requireNamespace("RSpectra", quietly = TRUE)) {
        # For very large matrices
        rsvd <- RSpectra::svds(X, k = rank)
        list(U = rsvd$u, D = rsvd$d, V = rsvd$v)
      } else {
        # Fallback to standard SVD
        svd(X, nu = rank, nv = rank)
      }
    } else {
      svd(X, nu = rank, nv = rank)
    }
  }, error = function(e) {
    svd(X, nu = rank, nv = rank)
  })
  
  U <- svd_result$U
  S <- svd_result$D
  V <- svd_result$V
  
  # Truncate to determined rank
  U <- U[, 1:rank, drop = FALSE]
  S <- S[1:rank]
  V <- V[, 1:rank, drop = FALSE]
  
  # Compute F = V * diag(1/S) * t(U)
  F <- V %*% diag(1 / S) %*% t(U)
  
  # Initialize
  w_0 <- F %*% y
  w <- w_0
  log2 <- log(2)
  t_val <- log2
  q <- 0.5
  epsilon <- params$epsilon
  n_iter <- params$maximumIteration
  tol <- params$convergenceTolerance
  min_iter <- params$minimumIteration
  
  do_regularization <- (params$lambda_ssr > 0 || params$gamma > 0)
  
  for (iteration in 0:n_iter) {
    w_hat <- w
    o <- X %*% w
    
    # Compute z for Newton-like update
    o_min <- pmin(o, 100)
    z <- (log(1 + exp(o_min)) - t_val - 0.5 * o) / (o^2 + epsilon)
    
    if (do_regularization) {
      d <- length(w)
      I <- diag(d)
      G <- t(V) * rep(S, each = rank)
      
      # v = (1/n) * t(G) %*% (t(U) %*% (y - q))
      v <- (1 / n) * t(G) %*% (t(U) %*% (y - q))
      
      p <- rep(1, d)
      if (params$fit_intercept) p[1] <- 0
      
      h <- p / ((abs(w_hat)^(2 - params$f) + epsilon))
      H <- diag(as.vector(h))
      
      # A = (2/n) * t(G) %*% (diag(z) %*% G) + (lambda_ssr/d) * I + (gamma/d) * H
      A <- (2 / n) * t(G) %*% (diag(z) %*% G) + (params$lambda_ssr / d) * I + (params$gamma / d) * H
      b <- (params$lambda_ssr / d) * w + v
      
      w <- solve(A, b)
    } else {
      # y_q = U %*% (t(U) %*% (y - q))
      y_q <- U %*% (t(U) %*% (y - q))
      w <- (1 / 2) * F %*% (y_q / z)
    }
    
    change <- max(abs(w - w_hat))
    
    if (iteration >= min_iter && change <= tol) {
      break
    }
  }
  
  list(
    coefficients = as.vector(w),
    rank = rank,
    n_iterations = iteration + 1,
    converged = change <= tol
  )
}

#' Determine optimal rank based on energy threshold
#'
#' @param X Input matrix
#' @param energy_percentile Energy threshold (0-100)
#' @noRd
.determineRank <- function(X, energy_percentile) {
  d <- min(nrow(X), ncol(X))
  percentile <- energy_percentile / 100.0
  
  # Compute SVD for rank determination
  s <- svd(X, nu = 0, nv = 0)$d
  log_s <- log(s + 1.0)
  cdf <- cumsum(log_s) / sum(log_s)
  
  r <- which(cdf > percentile & log_s > 1e-10)[1]
  if (is.na(r)) r <- d
  
  r <- max(min(r, d), 1)
  return(r)
}

#' Predict method for FastLogisticRegressionLowRank objects
#'
#' @param object Fitted model of class 'FastLogisticRegressionLowRank'
#' @param newdata New data matrix for prediction
#' @param type Type of prediction: "class" for class labels, "prob" for probabilities
#' @param ... Additional arguments (not used)
#' @return Predicted class labels or probabilities
#' @method predict FastLogisticRegressionLowRank
#' @export
predict.FastLogisticRegressionLowRank <- function(object, newdata, type = "class", ...) {
  newdata <- as.matrix(newdata)
  
  if (object$parameters$fit_intercept) {
    newdata <- cbind(1, newdata)
  }
  
  w <- if (object$parameters$fit_intercept) {
    c(object$intercept, object$coef_no_intercept)
  } else {
    object$coefficients
  }
  
  # Compute linear predictor
  eta <- newdata %*% w
  
  # Compute probability using logistic function
  prob <- 1 / (1 + exp(-pmin(eta, 100)))  # Clip for numerical stability
  
  if (type == "prob") {
    return(as.vector(prob))
  } else if (type == "class") {
    return(ifelse(prob >= 0.5, 1, 0))
  } else {
    stop("type must be 'class' or 'prob'")
  }
}

#' Print method for FastLogisticRegressionLowRank objects
#'
#' @param x Object of class 'FastLogisticRegressionLowRank'
#' @param ... Additional arguments (not used)
#' @export
print.FastLogisticRegressionLowRank <- function(x, ...) {
  cat("Fast Logistic Regression with Low-Rank Approximation\n")
  cat("--------------------------------------------------\n")
  cat("Rank:", x$rank, "\n")
  cat("Converged:", x$converged, "\n")
  cat("Iterations:", x$n_iterations, "\n")
  cat("Intercept:", x$intercept, "\n")
  cat("Coefficients (excluding intercept):\n")
  print(x$coef_no_intercept)
}
