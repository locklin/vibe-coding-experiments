\name{CvFastLogisticLowRank}
\alias{CvFastLogisticLowRank}
\title{Cross-Validation for Optimal Lambda Selection}
\description{
Performs cross-validation to select the optimal \code{lambda_ssr}
regularization parameter for
\code{\link{FastLogisticRegressionLowRank}}.
Supports both standard k-fold cross-validation via
\code{\link[CVST]{CV}} and fast cross-validation via sequential
testing using \code{\link[CVST]{fastCV}} from the CVST package.
}
\usage{
CvFastLogisticLowRank(X, y,
  lambda_values = 10^seq(-4, 2, length.out = 20),
  method = c("CV", "fastCV"),
  fold = 5L,
  verbose = FALSE,
  ...)
}
\arguments{
  \item{X}{A numeric matrix of predictors (n x p).}
  \item{y}{A binary response vector of length n with values in
    \{0, 1\}.}
  \item{lambda_values}{A numeric vector of \code{lambda_ssr} values
    to evaluate.
    Default: \code{10^seq(-4, 2, length.out = 20)}.}
  \item{method}{Character string specifying the cross-validation
    method:
    \describe{
      \item{\code{"CV"}}{Standard k-fold cross-validation
        (default).}
      \item{\code{"fastCV"}}{Fast CV via sequential testing, which
        can be faster by eliminating underperforming candidates
        early.}
    }
  }
  \item{fold}{Number of folds for k-fold CV (used when
    \code{method = "CV"}). Default: 5.}
  \item{verbose}{Logical; whether to print progress during
    cross-validation. Default: FALSE.}
  \item{...}{Additional arguments passed to
    \code{\link{FastLogisticRegressionLowRank}} (e.g.,
    \code{gamma}, \code{f}, \code{energy_percentile},
    \code{fit_intercept}).}
}
\value{
An object of class \code{"FastLogisticLowRankCV"} with components:
  \item{best_lambda}{The optimal \code{lambda_ssr} value as
    determined by cross-validation.}
  \item{best_params}{The full optimal parameter set as returned by
    CVST.}
  \item{lambda_values}{The lambda values that were evaluated.}
  \item{method}{The CV method used (\code{"CV"} or
    \code{"fastCV"}).}
}
\details{
This function wraps the
\code{\link{FastLogisticRegressionLowRank}} fitting procedure as a
CVST learner and uses the CVST package's cross-validation
infrastructure to evaluate different values of the L2
regularization parameter \code{lambda_ssr}.

The \code{"CV"} method performs standard k-fold cross-validation.
The \code{"fastCV"} method uses non-parametric testing coupled with
sequential analysis to determine the best parameter, potentially
eliminating underperforming candidates early.

The classification error rate (misclassification rate) is used as
the loss function for parameter selection.

Any additional model parameters (e.g., \code{gamma}, \code{f},
\code{energy_percentile}) are held fixed during cross-validation;
only \code{lambda_ssr} is tuned.
}
\note{
Requires the \pkg{CVST} package to be installed. Install it with
\code{install.packages("CVST")}.
}
\examples{
\dontrun{
set.seed(42)
n <- 300
p <- 10
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(1.5, -1, 0.8, rep(0, 7))
prob <- 1 / (1 + exp(-X \%*\% beta_true))
y <- rbinom(n, 1, prob)

# Standard k-fold CV
cv_result <- CvFastLogisticLowRank(X, y,
  lambda_values = 10^seq(-3, 1, length.out = 10),
  method = "CV", fold = 5)
print(cv_result)

# Fit final model with optimal lambda
fit <- FastLogisticRegressionLowRank(X, y,
  lambda_ssr = cv_result$best_lambda)

# Fast CV via sequential testing
cv_fast <- CvFastLogisticLowRank(X, y,
  lambda_values = 10^seq(-3, 1, length.out = 10),
  method = "fastCV")

# With additional fixed parameters
cv_reg <- CvFastLogisticLowRank(X, y,
  lambda_values = c(0.01, 0.1, 0.5, 1, 5),
  gamma = 0.1, f = 0.5)
}
}
\seealso{
\code{\link{FastLogisticRegressionLowRank}} for model fitting,
\code{\link[CVST]{CV}} and \code{\link[CVST]{fastCV}} for the
underlying cross-validation procedures.
}
