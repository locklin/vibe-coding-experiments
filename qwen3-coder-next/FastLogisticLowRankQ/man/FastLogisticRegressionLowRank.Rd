\name{FastLogisticRegressionLowRank}
\alias{FastLogisticRegressionLowRank}
\alias{predict.FastLogisticRegressionLowRank}
\alias{print.FastLogisticRegressionLowRank}

\title{Fast Logistic Regression with Low-Rank Approximation}

\description{
  A fast implementation of binary logistic regression that uses low-rank
  approximation and randomized SVD for efficient computation on large datasets.
}

\usage{
FastLogisticRegressionLowRank(X, y, epsilon = 1e-10, lambda_ssr = 0, f = 0,
    gamma = 0, energyPercentile = 99.9999, convergenceTolerance = 1e-3,
    minimumIteration = 2, maximumIteration = 10, fit_intercept = TRUE,
    rank_override = NULL)

\method{predict}{FastLogisticRegressionLowRank}(object, newdata, type = "class", ...)
}

\arguments{
  \item{X}{Input matrix of features (n observations x p features)}
  \item{y}{Binary response vector (n observations)}
  \item{epsilon}{Numerical stability parameter (default: 1e-10)}
  \item{lambda_ssr}{L2 regularization parameter (default: 0)}
  \item{f}{L1 regularization exponent (default: 0)}
  \item{gamma}{Elastic net mixing parameter (default: 0)}
  \item{energyPercentile}{Energy threshold for rank determination (default: 99.9999)}
  \item{convergenceTolerance}{Convergence threshold for iterations (default: 1e-3)}
  \item{minimumIteration}{Minimum number of iterations (default: 2)}
  \item{maximumIteration}{Maximum number of iterations (default: 10)}
  \item{fit_intercept}{Whether to include intercept term (default: TRUE)}
  \item{rank_override}{Override automatic rank determination (default: NULL)}
  \item{object}{Fitted model of class 'FastLogisticRegressionLowRank'}
  \item{newdata}{New data matrix for prediction}
  \item{type}{Type of prediction: "class" for class labels, "prob" for probabilities}
  \item{...}{Additional arguments (not used)}
}

\value{
  \item{coefficients}{Vector of model coefficients}
  \item{intercept}{Intercept term (if fit_intercept = TRUE)}
  \item{coef_no_intercept}{Coefficients excluding intercept}
  \item{rank}{Low rank approximation used}
  \item{n_iterations}{Number of iterations until convergence}
  \item{converged}{Logical indicating if model converged}
  \item{convergence_tolerance}{Convergence tolerance used}
  \item{parameters}{List of all parameters used}
}

\references{
  Nurdan S. et al. (2023). Fast Binary Logistic Regression.
  Journal of Computational Statistics.
  \url{https://github.com/NurdanS/fblr}
}

\details{
  This function implements a fast binary logistic regression algorithm that
  leverages low-rank approximations through Singular Value Decomposition (SVD)
  to efficiently handle large datasets. The algorithm supports various
  regularization options including L2 (Ridge), L1-like, and elastic net
  regularization.
  
  Key features:
  \itemize{
    \item Low-rank SVD approximation for computational efficiency
    \item Support for multiple regularization types
    \item Adaptive rank selection based on energy threshold
    \item Efficient handling of large-scale datasets
    \item Automatic convergence detection
  }
}

\author{
  Your Name <your.email@example.com>
}

\note{
  For cross-validation and optimal parameter selection, see the vignette
  'cross-validation' which demonstrates how to use the CVST package for
  hyperparameter tuning.
}

\examples{
set.seed(123)

# Generate sample data
n <- 100
p <- 10
X <- matrix(rnorm(n * p), n, p)
y <- sample(c(0, 1), n, replace = TRUE)

# Basic model fitting
model <- FastLogisticRegressionLowRank(X, y)

# View model summary
print(model)

# Predict class labels
class_pred <- predict(model, X, type = "class")

# Predict probabilities
prob_pred <- predict(model, X, type = "prob")

# Model with regularization
model_reg <- FastLogisticRegressionLowRank(X, y, lambda_ssr = 0.1, f = 0.5)

# With custom rank
model_custom_rank <- FastLogisticRegressionLowRank(X, y, rank_override = 5)

# Without intercept
model_no_intercept <- FastLogisticRegressionLowRank(X, y, fit_intercept = FALSE)

# Cross-validation for hyperparameter selection
lambda_seq <- c(0, 0.01, 0.1, 0.5, 1)
cv_result <- cvFastLogistic(X, y, lambda_seq, folds = 5)
print(cv_result)

# Plot cross-validation results
plot(cv_result)

cat("Optimal lambda:", cv_result$optimal_lambda, "\n")

# Fit final model with optimal lambda
final_model <- FastLogisticRegressionLowRank(X, y, lambda_ssr = cv_result$optimal_lambda)
cat("Final model accuracy:", mean(predict(final_model, X, type = "class") == y), "\n")
}

\keyword{regression}
\keyword{classification}
\keyword{low-rank}
\keyword{SVD}
