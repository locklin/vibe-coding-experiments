\name{cvFastLogistic}
\alias{cvFastLogistic}
\alias{cvFastLogisticCVST}
\alias{print.cvFastLogistic}
\alias{plot.cvFastLogistic}

\title{Cross-Validation for Fast Logistic Regression}

\description{
  Functions for performing cross-validation to find optimal regularization parameters
  for the FastLogisticRegressionLowRank model.
}

\usage{
cvFastLogistic(X, y, lambda_seq, folds = 5, seed = 42)

cvFastLogisticCVST(X, y, lambda_seq, folds = 5, measure = "misclass")

\method{plot}{cvFastLogistic}(cv_result, log_lambda = TRUE, ...)

\method{print}{cvFastLogistic}(x, ...)
}

\arguments{
  \item{X}{Input matrix of features}
  \item{y}{Binary response vector}
  \item{lambda_seq}{Vector of lambda_ssr values to test}
  \item{folds}{Number of folds for cross-validation (default: 5)}
  \item{seed}{Random seed for reproducibility (default: 42)}
  \item{measure}{Error measure for classification ("misclass" or "auc")}
  \item{cv_result}{Result from cvFastLogistic or cvFastLogisticCVST}
  \item{log_lambda}{Whether to use log scale for lambda (default: TRUE)}
  \item{...}{Additional arguments passed to plot}
  \item{x}{Result from cvFastLogistic or cvFastLogisticCVST}
}

\value{
  \item{results}{Data frame with lambda values and corresponding CV errors}
  \item{optimal_lambda}{Lambda with minimum CV error}
  \item{min_error}{Minimum cross-validation error}
}

\details{
  The \code{cvFastLogistic} function performs k-fold cross-validation to find
  the optimal regularization parameter (\code{lambda_ssr}) for the FastLogisticRegressionLowRank
  model. It evaluates each lambda value by training on k-1 folds and testing
  on the remaining fold, then averages the misclassification error across all folds.
  
  The \code{cvFastLogisticCVST} function provides an alternative interface using
  the CVST package for cross-validation.
}

\references{
  Nurdan S. et al. (2023). Fast Binary Logistic Regression.
  Journal of Computational Statistics.
  \url{https://github.com/NurdanS/fblr}
}

\author{
  Your Name <your.email@example.com>
}

\note{
  For the CVST-based cross-validation, you need to have the CVST package installed.
  You can install it with: \code{install.packages("CVST")}
}

\examples{
set.seed(123)

# Generate sample data
n <- 100
p <- 10
X <- matrix(rnorm(n * p), n, p)
y <- sample(c(0, 1), n, replace = TRUE)

# Define lambda sequence
lambda_seq <- c(0, 0.01, 0.1, 0.5, 1)

# Perform cross-validation
cv_result <- cvFastLogistic(X, y, lambda_seq, folds = 5)

# View results
print(cv_result)

# Plot cross-validation results
plot(cv_result)

# Get optimal lambda
cat("Optimal lambda:", cv_result$optimal_lambda, "\n")

# Fit final model with optimal lambda
final_model <- FastLogisticRegressionLowRank(X, y, lambda_ssr = cv_result$optimal_lambda)
cat("Final model accuracy:", mean(predict(final_model, X, type = "class") == y), "\n")
}

\keyword{cross-validation}
\keyword{regularization}
\keyword{hyperparameter}
