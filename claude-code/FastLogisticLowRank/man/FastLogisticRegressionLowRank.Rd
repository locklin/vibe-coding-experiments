\name{FastLogisticRegressionLowRank}
\alias{FastLogisticRegressionLowRank}
\title{Fast Binary Logistic Regression via Low-Rank Approximation}
\description{
Fits a binary logistic regression model using low-rank matrix approximation
via SVD. The algorithm projects the design matrix into a lower-dimensional
space determined by an energy-based rank selection criterion, reducing
computational cost on large datasets while maintaining accuracy.
}
\usage{
FastLogisticRegressionLowRank(X, y,
  epsilon = 1e-10,
  lambda_ssr = 0,
  f = 0,
  gamma = 0,
  energy_percentile = 99.9999,
  convergence_tolerance = 1e-3,
  minimum_iteration = 2L,
  maximum_iteration = 10L,
  fit_intercept = TRUE)
}
\arguments{
  \item{X}{A numeric matrix of predictors (n x p).}
  \item{y}{A binary response vector of length n with values in \{0, 1\}.}
  \item{epsilon}{Numerical stability constant. Default: 1e-10.}
  \item{lambda_ssr}{L2 regularization strength. Default: 0 (no regularization).}
  \item{f}{Regularization exponent for adaptive penalty. Used in the weight
    penalty term \eqn{|w|^{2-f}}. Default: 0.}
  \item{gamma}{Lp norm regularization strength. Default: 0 (no Lp regularization).}
  \item{energy_percentile}{Energy percentile threshold (0-100) for determining
    the rank of the low-rank approximation from the SVD singular values.
    Higher values retain more singular values for a more accurate approximation.
    Default: 99.9999.}
  \item{convergence_tolerance}{Convergence criterion for the iterative
    optimization. Iteration stops when the maximum absolute change in weights
    falls below this value. Default: 1e-3.}
  \item{minimum_iteration}{Minimum number of iterations before checking
    convergence. Must be >= 1. Default: 2.}
  \item{maximum_iteration}{Maximum number of iterations allowed. Must be >=
    \code{minimum_iteration}. Default: 10.}
  \item{fit_intercept}{Logical; whether to include an intercept (bias) term.
    When TRUE, a column of ones is prepended to X internally. Default: TRUE.}
}
\value{
An object of class \code{"FastLogisticLowRank"} containing:
  \item{coefficients}{Named numeric vector of fitted coefficients (excluding intercept).}
  \item{intercept}{The intercept value, or NULL if \code{fit_intercept = FALSE}.}
  \item{rank}{The rank used in the low-rank approximation.}
  \item{classes}{The unique class labels (0 and 1).}
  \item{data_reduction}{Logical indicating whether data reduction was applied.}
  \item{feature_reduction}{Logical indicating whether feature reduction (randomized SVD) was used.}
  \item{n_iterations}{Number of iterations performed.}
  \item{converged}{Logical indicating whether the algorithm converged.}
  \item{call}{The matched call.}
  \item{params}{A list of all fitting parameters.}
}
\details{
The algorithm works as follows:
\enumerate{
  \item Determines whether data reduction (subsampling rows for SVD) and/or
    feature reduction (randomized SVD) are beneficial based on the data
    dimensions.
  \item Computes a truncated SVD of the (possibly subsampled) design matrix.
  \item Selects the rank using an energy-based criterion on the log singular
    values.
  \item Iteratively updates the weight vector using a Newton-Raphson-style
    update in the reduced space, with element-wise weighting derived from
    the logistic loss.
  \item Optionally applies L2 (\code{lambda_ssr}) and adaptive Lp
    (\code{gamma}) regularization.
}

When \code{lambda_ssr > 0} or \code{gamma > 0}, the algorithm solves a
regularized system at each iteration. The intercept term is not penalized
when regularization is active.
}
\examples{
# Simple binary classification
set.seed(42)
n <- 200
p <- 5
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(1, -0.5, 0.3, 0, 0.8)
prob <- 1 / (1 + exp(-X \%*\% beta_true))
y <- rbinom(n, 1, prob)

# Basic fit
fit <- FastLogisticRegressionLowRank(X, y)
print(fit)
summary(fit)

# With L2 regularization
fit_l2 <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.1)

# With both L2 and Lp regularization
fit_reg <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.1, gamma = 0.05)

# Without intercept
fit_noint <- FastLogisticRegressionLowRank(X, y, fit_intercept = FALSE)
}
\seealso{
\code{\link{predict.FastLogisticLowRank}} for predictions.
}
