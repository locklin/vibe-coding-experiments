\name{FastLogisticLowRank-package}
\alias{FastLogisticLowRank-package}
\docType{package}
\title{Fast Binary Logistic Regression via Low-Rank Approximation}
\description{
Implements a fast binary logistic regression classifier using
low-rank matrix approximation via SVD. The algorithm reduces
computational cost by projecting the design matrix into a
lower-dimensional space determined by an energy-based rank
selection criterion. Supports L2 and adaptive Lp regularization.

Based on the methodology described in the FBLR (Fast Binary
Logistic Regression) framework.
}
\details{
The main functions are:
\itemize{
  \item \code{\link{FastLogisticRegressionLowRank}}: Fit a fast
    logistic regression model with low-rank approximation.
  \item \code{\link{predict.FastLogisticLowRank}}: Generate
    predictions (probabilities, classes, or linear predictor) from
    a fitted model.
  \item \code{\link{CvFastLogisticLowRank}}: Cross-validation for
    selecting the optimal L2 regularization parameter using the
    CVST package.
}
}
\author{
FastLogisticLowRank Authors
}
\seealso{
\code{\link{FastLogisticRegressionLowRank}},
\code{\link{predict.FastLogisticLowRank}},
\code{\link{CvFastLogisticLowRank}}
}
\examples{
set.seed(42)
n <- 200
p <- 5
X <- matrix(rnorm(n * p), n, p)
beta_true <- c(1, -0.5, 0.3, 0, 0.8)
prob <- 1 / (1 + exp(-X \%*\% beta_true))
y <- rbinom(n, 1, prob)

fit <- FastLogisticRegressionLowRank(X, y)
probs <- predict(fit, X, type = "response")
classes <- predict(fit, X, type = "class")
cat("Accuracy:", mean(classes == y), "\n")
}
