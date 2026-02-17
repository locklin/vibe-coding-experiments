# cv.R
#
# Cross-validation functionality using the CVST package, and the
# FastLogisticLowRankCV S3 class.


#' Cross-Validation for Optimal Lambda Selection
#'
#' Performs cross-validation to select the optimal \code{lambda_ssr}
#' regularization parameter for
#' \code{\link{FastLogisticRegressionLowRank}}.
#' Supports both standard k-fold cross-validation via
#' \code{\link[CVST]{CV}} and fast cross-validation via sequential
#' testing using \code{\link[CVST]{fastCV}}.
#'
#' @param X A numeric matrix of predictors (n x p).
#' @param y A binary response vector of length n with values in
#'   \{0, 1\}.
#' @param lambda_values A numeric vector of \code{lambda_ssr} values
#'   to evaluate. Default: \code{10^seq(-4, 2, length.out = 20)}.
#' @param method Character string: \code{"CV"} for standard k-fold
#'   cross-validation or \code{"fastCV"} for fast CV via sequential
#'   testing. Default: \code{"CV"}.
#' @param fold Number of folds for k-fold CV (used when
#'   \code{method = "CV"}). Default: 5.
#' @param verbose Logical; whether to print progress. Default: FALSE.
#' @param ... Additional arguments passed to
#'   \code{\link{FastLogisticRegressionLowRank}} (e.g., \code{gamma},
#'   \code{f}, \code{energy_percentile}).
#'
#' @return An object of class \code{"FastLogisticLowRankCV"}
#'   containing:
#'   \item{best_lambda}{The optimal \code{lambda_ssr} value.}
#'   \item{best_params}{The full optimal parameter set as returned
#'     by CVST.}
#'   \item{lambda_values}{The lambda values that were evaluated.}
#'   \item{method}{The CV method used.}
#'
#' @examples
#' \dontrun{
#' set.seed(42)
#' n <- 300
#' p <- 10
#' X <- matrix(rnorm(n * p), n, p)
#' beta_true <- c(1.5, -1, 0.8, rep(0, 7))
#' prob <- 1 / (1 + exp(-X %*% beta_true))
#' y <- rbinom(n, 1, prob)
#'
#' # Standard k-fold CV
#' cv_result <- CvFastLogisticLowRank(X, y,
#'   lambda_values = 10^seq(-3, 1, length.out = 10))
#' print(cv_result)
#'
#' # Fit final model with optimal lambda
#' fit <- FastLogisticRegressionLowRank(X, y,
#'   lambda_ssr = cv_result$best_lambda)
#'
#' # Fast CV via sequential testing
#' cv_fast <- CvFastLogisticLowRank(X, y, method = "fastCV",
#'   lambda_values = 10^seq(-3, 1, length.out = 10))
#' }
#'
#' @export
CvFastLogisticLowRank <- function(
    X, y,
    lambda_values = 10^seq(-4, 2, length.out = 20),
    method = c("CV", "fastCV"),
    fold = 5L,
    verbose = FALSE,
    ...) {

  if (!requireNamespace("CVST", quietly = TRUE)) {
    stop(
      "Package 'CVST' is required for cross-validation. ",
      "Install it with: install.packages('CVST')"
    )
  }

  method <- match.arg(method)
  if (!is.matrix(X)) {
    X <- as.matrix(X)
  }

  # Construct CVST data object (factor y for classification).
  cvst_data <- CVST::constructData(x = X, y = as.factor(y))

  # Build learner wrapper and parameter grid.
  learner <- .ConstructFblrLearner(...)
  params <- CVST::constructParams(lambda_ssr = lambda_values)

  # Run cross-validation.
  if (method == "CV") {
    best_params <- CVST::CV(
      cvst_data, learner, params,
      fold = fold, verbose = verbose
    )
  } else {
    setup <- CVST::constructCVSTModel()
    best_params <- CVST::fastCV(
      cvst_data, learner, params, setup,
      verbose = verbose
    )
  }

  # Extract the lambda value from the winning parameter set.
  winner_set <- best_params[[1]]
  best_lambda <- if (!is.null(winner_set$lambda_ssr)) {
    winner_set$lambda_ssr
  } else {
    winner_set[[1]]
  }

  result <- list(
    best_lambda = best_lambda,
    best_params = best_params,
    lambda_values = lambda_values,
    method = method
  )
  class(result) <- "FastLogisticLowRankCV"
  return(result)
}


#' Print Method for FastLogisticLowRankCV
#'
#' @param x A \code{"FastLogisticLowRankCV"} object.
#' @param ... Additional arguments (currently unused).
#' @return The object invisibly.
#' @export
print.FastLogisticLowRankCV <- function(x, ...) {
  cat("Cross-Validation Result for FastLogisticLowRank\n")
  cat(paste(rep("-", 48), collapse = ""), "\n")
  cat("Method:", x$method, "\n")
  cat("Lambda values evaluated:",
      paste(round(x$lambda_values, 6), collapse = ", "), "\n")
  cat("Best lambda:", x$best_lambda, "\n")
  return(invisible(x))
}


# ---- Internal CVST learner wrapper ---------------------------------------

#' Construct a CVST learner for FastLogisticRegressionLowRank (internal).
#'
#' @param ... Additional arguments forwarded to
#'   \code{FastLogisticRegressionLowRank}.
#' @return A \code{CVST.learner} object.
#' @keywords internal
.ConstructFblrLearner <- function(...) {
  extra_args <- list(...)

  learn_fn <- function(data, params) {
    X <- CVST::getX(data)
    y <- as.numeric(as.character(data$y))

    # constructParams with a single parameter produces unnamed inner
    # lists, so extract by name if available, else by position.
    lambda_val <- if (!is.null(params$lambda_ssr)) {
      params$lambda_ssr
    } else {
      params[[1]]
    }

    call_args <- extra_args
    call_args$X <- X
    call_args$y <- y
    call_args$lambda_ssr <- lambda_val

    return(do.call(FastLogisticRegressionLowRank, call_args))
  }

  predict_fn <- function(model, data) {
    X <- CVST::getX(data)
    preds <- predict.FastLogisticLowRank(model, X, type = "class")
    # CVST classification loss: test$y != pred.
    return(factor(preds, levels = levels(data$y)))
  }

  return(CVST::constructLearner(learn_fn, predict_fn))
}
