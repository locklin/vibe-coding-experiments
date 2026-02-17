#' Mean Squared Error
#' @param actual Numeric vector of actual values.
#' @param predicted Numeric vector of predicted values.
#' @return Scalar MSE value.
#' @export
mse <- function(actual, predicted) {
  mean((actual - predicted)^2)
}

#' Root Mean Squared Error
#' @param actual Numeric vector of actual values.
#' @param predicted Numeric vector of predicted values.
#' @return Scalar RMSE value.
#' @export
rmse <- function(actual, predicted) {
  sqrt(mean((actual - predicted)^2))
}

#' Mean Absolute Error
#' @param actual Numeric vector of actual values.
#' @param predicted Numeric vector of predicted values.
#' @return Scalar MAE value.
#' @export
mae <- function(actual, predicted) {
  mean(abs(actual - predicted))
}

#' R-squared (Coefficient of Determination)
#' @param actual Numeric vector of actual values.
#' @param predicted Numeric vector of predicted values.
#' @return Scalar R-squared value.
#' @export
r_squared <- function(actual, predicted) {
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  if (ss_tot == 0) return(1.0)
  1.0 - ss_res / ss_tot
}
