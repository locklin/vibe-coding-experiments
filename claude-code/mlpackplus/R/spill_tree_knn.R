#' SpillTree K-Nearest Neighbors
#'
#' Builds a spill tree on reference data and finds k-nearest neighbors.
#' When tau=0, this is equivalent to exact kd-tree search.
#'
#' @param reference Numeric matrix of reference points (n x d).
#' @param k Number of neighbors to find.
#' @param tau Overlap parameter in [0, 0.5). Controls the fraction of
#'   range overlap between children (default 0, exact kd-tree).
#' @param max_leaf_size Maximum leaf size (default 20).
#' @return An S3 object of class \code{SpillTreeKNN}.
#' @export
#' @examples
#' ref <- matrix(rnorm(200), ncol = 2)
#' model <- SpillTreeKNN(ref, k = 5)
#' # Query with new points
#' query <- matrix(rnorm(20), ncol = 2)
#' result <- predict(model, query)
SpillTreeKNN <- function(reference, k, tau = 0.0, max_leaf_size = 20L) {
  reference <- as.matrix(reference)
  k <- as.integer(k)

  if (k < 1L) stop("k must be >= 1")
  if (tau < 0 || tau >= 0.5) stop("tau must be in [0, 0.5)")
  if (max_leaf_size < 1L) stop("max_leaf_size must be >= 1")
  if (k > nrow(reference)) {
    stop("k must be <= number of reference points")
  }

  # Search reference against itself.
  result <- cpp_build_and_search_spill_tree(
    reference, reference, k, tau, as.integer(max_leaf_size))

  structure(
    list(
      reference = reference,
      k = k,
      tau = tau,
      max_leaf_size = max_leaf_size,
      neighbors = result$neighbors,
      distances = result$distances,
      n_nodes = result$n_nodes,
      n_reference = nrow(reference),
      n_features = ncol(reference)
    ),
    class = "SpillTreeKNN"
  )
}

#' @export
predict.SpillTreeKNN <- function(object, newdata, ...) {
  newdata <- as.matrix(newdata)
  if (ncol(newdata) != object$n_features) {
    stop("newdata must have ", object$n_features, " columns")
  }

  result <- cpp_build_and_search_spill_tree(
    object$reference, newdata, object$k, object$tau,
    as.integer(object$max_leaf_size))

  list(neighbors = result$neighbors, distances = result$distances)
}

#' @export
print.SpillTreeKNN <- function(x, ...) {
  cat("SpillTreeKNN\n")
  cat("  Reference points:", x$n_reference, "\n")
  cat("  Features:        ", x$n_features, "\n")
  cat("  k:               ", x$k, "\n")
  cat("  tau:             ", x$tau, "\n")
  cat("  Max leaf size:   ", x$max_leaf_size, "\n")
  cat("  Tree nodes:      ", x$n_nodes, "\n")
  invisible(x)
}
