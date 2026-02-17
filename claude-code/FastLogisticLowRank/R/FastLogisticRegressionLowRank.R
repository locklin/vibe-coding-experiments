# FastLogisticRegressionLowRank.R
#
# Core fitting, prediction, and internal S3 classes for the
# FastLogisticLowRank package.


# ---- FblrSvd S3 class ---------------------------------------------------

#' Constructor for the internal FblrSvd S3 class.
#'
#' Wraps an SVD decomposition (singular values and right singular vectors)
#' into a standardized S3 object used by the fitting algorithm.
#'
#' @param d Numeric vector of singular values.
#' @param vh Matrix of right singular vectors transposed (r x d).
#' @param u Matrix of left singular vectors (n x r), or NULL.
#' @return An object of class \code{"FblrSvd"}.
#' @keywords internal
.NewFblrSvd <- function(d, vh, u = NULL) {
  obj <- list(d = d, vh = vh, u = u)
  class(obj) <- "FblrSvd"
  return(obj)
}


#' Determine Rank from Singular Values
#'
#' S3 generic for determining the numerical rank of a decomposition
#' based on an energy percentile threshold.
#'
#' @param x An object to determine rank for.
#' @param ... Additional arguments passed to methods.
#' @return Integer rank.
#' @keywords internal
DetermineRank <- function(x, ...) {
  UseMethod("DetermineRank")
}


#' @describeIn DetermineRank Method for FblrSvd objects.
#'
#' Selects the rank such that the cumulative log-energy of the
#' singular values exceeds the specified percentile.
#'
#' @param x An \code{"FblrSvd"} object.
#' @param energy_percentile Numeric energy percentile threshold (0-100).
#' @param ... Additional arguments (unused).
#' @return Integer rank.
#' @keywords internal
DetermineRank.FblrSvd <- function(x, energy_percentile, ...) {
  singular_values <- x$d
  percentile <- energy_percentile / 100.0
  log_s <- log(singular_values + 1.0)
  cdf <- cumsum(log_s) / sum(log_s)

  valid <- which(cdf > percentile & log_s > 1e-10)
  if (length(valid) == 0) {
    r <- length(singular_values)
  } else {
    r <- min(valid)
  }
  r <- max(min(r, length(singular_values)), 1L)
  return(r)
}


#' Compute Left Singular Vectors from Data
#'
#' S3 generic for computing U given the original data matrix and an
#' SVD-like decomposition.
#'
#' @param x An object containing the decomposition.
#' @param ... Additional arguments passed to methods.
#' @return A matrix of left singular vectors.
#' @keywords internal
ComputeU <- function(x, ...) {
  UseMethod("ComputeU")
}


#' @describeIn ComputeU Method for FblrSvd objects.
#'
#' Computes U = X * Vh' * diag(1/S).
#'
#' @param x An \code{"FblrSvd"} object.
#' @param data_matrix The original data matrix X (n x d).
#' @param ... Additional arguments (unused).
#' @return Numeric matrix (n x r).
#' @keywords internal
ComputeU.FblrSvd <- function(x, data_matrix, ...) {
  s_inv <- 1 / x$d
  # t(Vh) is d x r; scale each column j by s_inv[j], then premultiply by X.
  return(data_matrix %*% sweep(t(x$vh), 2, s_inv, "*"))
}


#' Truncate an FblrSvd to a Given Rank
#'
#' S3 generic for truncating a decomposition to a specified rank.
#'
#' @param x An object containing the decomposition.
#' @param ... Additional arguments passed to methods.
#' @return A truncated decomposition.
#' @keywords internal
Truncate <- function(x, ...) {
  UseMethod("Truncate")
}


#' @describeIn Truncate Method for FblrSvd objects.
#'
#' Retains only the first \code{r} singular values and corresponding
#' vectors.
#'
#' @param x An \code{"FblrSvd"} object.
#' @param r Integer rank to truncate to.
#' @param ... Additional arguments (unused).
#' @return A truncated \code{"FblrSvd"} object.
#' @keywords internal
Truncate.FblrSvd <- function(x, r, ...) {
  u_trunc <- if (!is.null(x$u)) {
    x$u[, seq_len(r), drop = FALSE]
  } else {
    NULL
  }
  return(.NewFblrSvd(
    d = x$d[seq_len(r)],
    vh = x$vh[seq_len(r), , drop = FALSE],
    u = u_trunc
  ))
}


#' Randomized SVD (internal).
#'
#' Computes a truncated SVD via random projection and power iteration.
#' Returns an \code{"FblrSvd"} object.
#'
#' @param X Numeric matrix (n x d).
#' @param n_components Number of components to retain.
#' @param n_oversamples Number of oversampling columns. Default: 10.
#' @param n_iter Number of power iterations. Default: 4.
#' @return An \code{"FblrSvd"} object.
#' @keywords internal
.RandomizedSvd <- function(X, n_components,
                           n_oversamples = 10L, n_iter = 4L) {
  n <- nrow(X)
  d <- ncol(X)
  k <- min(n_components + n_oversamples, n, d)

  set.seed(12345)
  omega <- matrix(rnorm(d * k), nrow = d, ncol = k)
  Y <- X %*% omega

  for (i in seq_len(n_iter)) {
    Y <- qr.Q(qr(Y))
    Z <- t(X) %*% Y
    Z <- qr.Q(qr(Z))
    Y <- X %*% Z
  }

  Q <- qr.Q(qr(Y))
  B <- t(Q) %*% X

  svd_b <- svd(B)
  nc <- min(n_components, length(svd_b$d))

  return(.NewFblrSvd(
    d = svd_b$d[seq_len(nc)],
    vh = t(svd_b$v[, seq_len(nc), drop = FALSE]),
    u = Q %*% svd_b$u[, seq_len(nc), drop = FALSE]
  ))
}


#' Wrap a base R svd() result into an FblrSvd object (internal).
#'
#' @param svd_result Result from \code{svd()}.
#' @return An \code{"FblrSvd"} object.
#' @keywords internal
.WrapSvd <- function(svd_result) {
  return(.NewFblrSvd(
    d = svd_result$d,
    vh = t(svd_result$v),
    u = svd_result$u
  ))
}


# ---- Main fitting function -----------------------------------------------

#' Fast Binary Logistic Regression via Low-Rank Approximation
#'
#' Fits a binary logistic regression model using low-rank matrix approximation
#' via SVD. The algorithm projects the design matrix into a lower-dimensional
#' space to reduce computational cost while maintaining accuracy.
#'
#' @param X A numeric matrix of predictors (n x p).
#' @param y A binary response vector of length n with values in \{0, 1\}.
#' @param epsilon Numerical stability constant. Default: 1e-10.
#' @param lambda_ssr L2 regularization strength. Default: 0 (no
#'   regularization).
#' @param f Regularization exponent for adaptive penalty. Default: 0.
#' @param gamma Lp norm regularization strength. Default: 0 (no Lp
#'   regularization).
#' @param energy_percentile Energy percentile threshold (0-100) for
#'   determining the rank of the low-rank approximation. Higher values
#'   retain more singular values. Default: 99.9999.
#' @param convergence_tolerance Convergence criterion for the iterative
#'   optimization. Iteration stops when the maximum absolute change in
#'   weights falls below this value. Default: 1e-3.
#' @param minimum_iteration Minimum number of iterations before checking
#'   convergence. Default: 2.
#' @param maximum_iteration Maximum number of iterations allowed.
#'   Default: 10.
#' @param fit_intercept Logical; whether to include an intercept (bias)
#'   term. Default: TRUE.
#'
#' @return An object of class \code{"FastLogisticLowRank"} containing:
#'   \item{coefficients}{Named numeric vector of fitted coefficients.}
#'   \item{intercept}{The intercept value (NULL if
#'     \code{fit_intercept = FALSE}).}
#'   \item{rank}{The rank used in the low-rank approximation.}
#'   \item{classes}{The unique class labels found in \code{y}.}
#'   \item{data_reduction}{Logical: whether data reduction was applied.}
#'   \item{feature_reduction}{Logical: whether feature reduction
#'     (randomized SVD) was used.}
#'   \item{n_iterations}{Number of iterations performed.}
#'   \item{converged}{Logical: whether the algorithm converged.}
#'   \item{call}{The matched call.}
#'   \item{params}{A list of all fitting parameters.}
#'
#' @examples
#' set.seed(42)
#' n <- 200
#' p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(1, -0.5, 0.3, 0, 0.8)
#' prob <- 1 / (1 + exp(-X %*% beta_true))
#' y <- rbinom(n, 1, prob)
#'
#' # Fit model
#' fit <- FastLogisticRegressionLowRank(X, y)
#' print(fit)
#'
#' # Predict probabilities
#' probs <- predict(fit, X, type = "response")
#'
#' # Predict classes
#' classes <- predict(fit, X, type = "class")
#'
#' # With L2 regularization
#' fit_reg <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.1)
#'
#' @export
FastLogisticRegressionLowRank <- function(
    X, y,
    epsilon = 1e-10,
    lambda_ssr = 0,
    f = 0,
    gamma = 0,
    energy_percentile = 99.9999,
    convergence_tolerance = 1e-3,
    minimum_iteration = 2L,
    maximum_iteration = 10L,
    fit_intercept = TRUE) {

  cl <- match.call()

  # --- Input validation ---
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }
  if (!is.numeric(X)) {
    stop("X must be a numeric matrix")
  }
  y <- as.numeric(y)
  classes <- sort(unique(y))
  if (length(classes) != 2 || !all(classes %in% c(0, 1))) {
    stop("y must be a binary vector with values 0 and 1")
  }
  if (nrow(X) != length(y)) {
    stop("nrow(X) must equal length(y)")
  }
  if (epsilon <= 0) {
    stop("epsilon must be positive")
  }
  if (lambda_ssr < 0) {
    stop("lambda_ssr must be non-negative")
  }
  if (gamma < 0) {
    stop("gamma must be non-negative")
  }
  if (energy_percentile <= 0 || energy_percentile > 100) {
    stop("energy_percentile must be in (0, 100]")
  }
  if (convergence_tolerance <= 0) {
    stop("convergence_tolerance must be positive")
  }
  minimum_iteration <- as.integer(minimum_iteration)
  maximum_iteration <- as.integer(maximum_iteration)
  if (minimum_iteration < 1) {
    stop("minimum_iteration must be >= 1")
  }
  if (maximum_iteration < minimum_iteration) {
    stop("maximum_iteration must be >= minimum_iteration")
  }

  n <- nrow(X)
  d <- ncol(X) + ifelse(fit_intercept, 1L, 0L)

  # Add intercept column.
  if (fit_intercept) {
    x_train <- cbind(1, X)
  } else {
    x_train <- X
  }

  # --- Decide on data and feature reduction ---
  low_rank_threshold <- max(d * 100, 100000)
  do_data_reduction <- (n > 2 * low_rank_threshold)

  d_limit <- 500
  r_hat <- min(sqrt(d * d_limit), d)
  cost_ratio <- (n * d^2) /
    (n * d * r_hat + n * r_hat^2 + r_hat^2 * d)
  do_feature_reduction <- (cost_ratio <= 1.80)

  if (do_data_reduction) {
    x_subset <- x_train[seq_len(low_rank_threshold), , drop = FALSE]
  } else {
    x_subset <- x_train
  }

  # --- SVD computation → FblrSvd object ---
  if (do_feature_reduction) {
    d_hat <- max(min(as.integer(n / 50), d), 1L)
    svd_obj <- .RandomizedSvd(x_subset, n_components = d_hat)
    U <- ComputeU(svd_obj, data_matrix = x_train)
  } else {
    svd_obj <- .WrapSvd(svd(x_subset))
    U <- svd_obj$u
  }

  # Determine rank via energy percentile.
  r <- DetermineRank(svd_obj, energy_percentile = energy_percentile)
  svd_obj <- Truncate(svd_obj, r = r)

  # Adjust for data reduction.
  if (do_data_reduction) {
    scale_factor <- sqrt((n - 1) / (low_rank_threshold - 1))
    svd_obj <- .NewFblrSvd(
      d = svd_obj$d * scale_factor,
      vh = svd_obj$vh,
      u = NULL
    )
    U <- ComputeU(svd_obj, data_matrix = x_train)
  } else {
    U <- U[, seq_len(r), drop = FALSE]
  }

  singular_values <- svd_obj$d
  vh <- svd_obj$vh

  # --- Precompute projection matrix F (d x n) ---
  # F = t(Vh) %*% diag(1/S) %*% t(U)
  s_inv <- 1 / singular_values
  f_mat <- sweep(t(vh), 2, s_inv, "*") %*% t(U)

  # Initial weights.
  w <- as.vector(f_mat %*% y)

  log2_val <- log(2.0)
  q <- 0.5
  do_regularization <- (lambda_ssr > 0 || gamma > 0)

  if (do_regularization) {
    i_mat <- diag(d)
    # G = diag(S) %*% Vh (r x d).
    G <- sweep(vh, 1, singular_values, "*")
    v_vec <- (1 / n) * t(G) %*% (t(U) %*% (y - q))

    p_vec <- rep(1, d)
    if (fit_intercept) {
      p_vec[1] <- 0
    }
  } else {
    y_q <- U %*% (t(U) %*% (y - q))
  }

  n_iter <- 0L
  converged <- FALSE

  for (iteration in seq(0, maximum_iteration)) {
    w_hat <- w
    o <- as.vector(x_train %*% w_hat)

    o_clipped <- pmin(o, 100)
    z <- (log(1 + exp(o_clipped)) - log2_val - 0.5 * o) /
      (o^2 + epsilon)

    if (do_regularization) {
      h <- p_vec / (abs(w_hat)^(2 - f) + epsilon)
      H <- diag(h)

      z_U <- z * U
      mid <- t(U) %*% z_U
      A <- (2 / n) * t(G) %*% mid %*% G +
        (lambda_ssr / d) * i_mat + (gamma / d) * H
      b <- (lambda_ssr / d) * w + v_vec

      w <- as.vector(solve(A, b))
    } else {
      w <- as.vector((1 / 2) * f_mat %*% (y_q / z))
    }

    n_iter <- iteration + 1L
    change <- max(abs(w - w_hat))
    if (iteration >= minimum_iteration &&
        change <= convergence_tolerance) {
      converged <- TRUE
      break
    }
  }

  # --- Extract intercept and coefficients ---
  if (fit_intercept) {
    intercept <- w[1]
    coefs <- w[-1]
  } else {
    intercept <- NULL
    coefs <- w
  }

  if (!is.null(colnames(X))) {
    names(coefs) <- colnames(X)
  } else {
    names(coefs) <- paste0("V", seq_along(coefs))
  }

  result <- list(
    coefficients = coefs,
    intercept = intercept,
    rank = r,
    classes = classes,
    data_reduction = do_data_reduction,
    feature_reduction = do_feature_reduction,
    n_iterations = n_iter,
    converged = converged,
    call = cl,
    params = list(
      epsilon = epsilon,
      lambda_ssr = lambda_ssr,
      f = f,
      gamma = gamma,
      energy_percentile = energy_percentile,
      convergence_tolerance = convergence_tolerance,
      minimum_iteration = minimum_iteration,
      maximum_iteration = maximum_iteration,
      fit_intercept = fit_intercept
    )
  )
  class(result) <- "FastLogisticLowRank"
  return(result)
}


# ---- S3 methods for FastLogisticLowRank ----------------------------------

#' Predict Method for FastLogisticLowRank
#'
#' Generate predictions from a fitted FastLogisticLowRank model.
#'
#' @param object A fitted \code{"FastLogisticLowRank"} object.
#' @param newdata A numeric matrix of new predictor values.
#' @param type Character string specifying the type of prediction:
#'   \code{"response"} returns predicted probabilities (of class 1),
#'   \code{"class"} returns predicted class labels (0 or 1),
#'   \code{"link"} returns the linear predictor values.
#' @param threshold Decision threshold for class predictions when
#'   \code{type = "class"}. Default: 0.5.
#' @param ... Additional arguments (currently unused).
#'
#' @return Depending on \code{type}:
#'   \itemize{
#'     \item \code{"response"}: Numeric vector of predicted probabilities.
#'     \item \code{"class"}: Integer vector of predicted class labels.
#'     \item \code{"link"}: Numeric vector of linear predictor values.
#'   }
#'
#' @examples
#' set.seed(42)
#' n <- 200
#' p <- 5
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(1, -0.5, 0.3, 0, 0.8)
#' prob <- 1 / (1 + exp(-X %*% beta_true))
#' y <- rbinom(n, 1, prob)
#' fit <- FastLogisticRegressionLowRank(X, y)
#'
#' # Predicted probabilities
#' probs <- predict(fit, X, type = "response")
#' head(probs)
#'
#' # Predicted classes
#' classes <- predict(fit, X, type = "class")
#' table(classes, y)
#'
#' @export
predict.FastLogisticLowRank <- function(object, newdata,
                                        type = c("response",
                                                 "class",
                                                 "link"),
                                        threshold = 0.5, ...) {
  type <- match.arg(type)
  if (!is.matrix(newdata)) {
    newdata <- as.matrix(newdata)
  }

  w <- object$coefficients
  if (object$params$fit_intercept) {
    newdata <- cbind(1, newdata)
    w <- c(object$intercept, w)
  }

  eta <- as.vector(newdata %*% w)

  if (type == "link") {
    return(eta)
  }

  # Sigmoid with numerical stability.
  e <- exp(pmin(eta, 100))
  prob <- e / (1 + e)

  if (type == "response") {
    return(prob)
  }

  # type == "class"
  return(as.integer(prob >= threshold))
}


#' Print Method for FastLogisticLowRank
#'
#' @param x A \code{"FastLogisticLowRank"} object.
#' @param ... Additional arguments (currently unused).
#' @return The object invisibly.
#' @export
print.FastLogisticLowRank <- function(x, ...) {
  cat("Fast Logistic Regression via Low-Rank Approximation\n\n")
  cat("Call:\n")
  print(x$call)
  cat("\nCoefficients:\n")
  if (!is.null(x$intercept)) {
    coefs <- c("(Intercept)" = x$intercept, x$coefficients)
  } else {
    coefs <- x$coefficients
  }
  print(round(coefs, 6))
  cat("\nRank:", x$rank)
  cat("\nIterations:", x$n_iterations)
  cat("\nConverged:", x$converged, "\n")
  return(invisible(x))
}


#' Summary Method for FastLogisticLowRank
#'
#' @param object A \code{"FastLogisticLowRank"} object.
#' @param ... Additional arguments (currently unused).
#' @return The object invisibly.
#' @export
summary.FastLogisticLowRank <- function(object, ...) {
  cat("Fast Logistic Regression via Low-Rank Approximation\n")
  cat(paste(rep("=", 52), collapse = ""), "\n\n")
  cat("Call:\n")
  print(object$call)
  cat("\nCoefficients:\n")
  if (!is.null(object$intercept)) {
    coefs <- c("(Intercept)" = object$intercept,
               object$coefficients)
  } else {
    coefs <- object$coefficients
  }
  print(round(coefs, 6))
  cat("\n--- Model Details ---\n")
  cat("Rank of low-rank approximation:", object$rank, "\n")
  cat("Iterations:", object$n_iterations, "\n")
  cat("Converged:", object$converged, "\n")
  cat("Data reduction used:", object$data_reduction, "\n")
  cat("Feature reduction used:", object$feature_reduction, "\n")
  cat("\n--- Regularization ---\n")
  cat("lambda_ssr:", object$params$lambda_ssr, "\n")
  cat("gamma:", object$params$gamma, "\n")
  cat("f:", object$params$f, "\n")
  return(invisible(object))
}
