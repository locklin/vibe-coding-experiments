\name{predict.FastLogisticLowRank}
\alias{predict.FastLogisticLowRank}
\title{Predict Method for FastLogisticLowRank}
\description{
Generate predictions from a fitted FastLogisticLowRank model. Supports
predicted probabilities, class labels, and linear predictor values.
}
\usage{
\method{predict}{FastLogisticLowRank}(object, newdata,
  type = c("response", "class", "link"),
  threshold = 0.5, ...)
}
\arguments{
  \item{object}{A fitted \code{"FastLogisticLowRank"} object returned by
    \code{\link{FastLogisticRegressionLowRank}}.}
  \item{newdata}{A numeric matrix of new predictor values with the same
    number of columns as the training data X.}
  \item{type}{Character string specifying the type of prediction:
    \describe{
      \item{\code{"response"}}{Returns predicted probabilities of class 1 (default).}
      \item{\code{"class"}}{Returns predicted class labels (0 or 1).}
      \item{\code{"link"}}{Returns the linear predictor values (log-odds).}
    }
  }
  \item{threshold}{Decision threshold for class predictions when
    \code{type = "class"}. Observations with predicted probability >= threshold
    are assigned to class 1. Default: 0.5.}
  \item{...}{Additional arguments (currently unused).}
}
\value{
Depending on \code{type}:
  \itemize{
    \item \code{"response"}: A numeric vector of predicted probabilities.
    \item \code{"class"}: An integer vector of predicted class labels (0 or 1).
    \item \code{"link"}: A numeric vector of linear predictor values.
  }
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

# Predicted probabilities
probs <- predict(fit, X, type = "response")
head(probs)

# Predicted classes
classes <- predict(fit, X, type = "class")
table(Predicted = classes, Actual = y)

# Linear predictor
eta <- predict(fit, X, type = "link")
# Should match: 1/(1+exp(-eta)) == probs
}
\seealso{
\code{\link{FastLogisticRegressionLowRank}} for model fitting.
}
