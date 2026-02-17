# Synthetic data generators for tests.

# Step function: y = 0 if x < 0, y = 1 if x >= 0.
make_step_data <- function(n = 200, d = 1, noise_sd = 0.05, seed = 123) {
  set.seed(seed)
  x <- matrix(rnorm(n * d), ncol = d)
  y <- ifelse(x[, 1] > 0, 1, 0) + rnorm(n, sd = noise_sd)
  list(x = x, y = y)
}

# Sine function: y = sin(x1) + 0.5 * cos(x2) + noise.
make_sine_data <- function(n = 200, noise_sd = 0.1, seed = 123) {
  set.seed(seed)
  x <- matrix(runif(n * 2, -pi, pi), ncol = 2)
  y <- sin(x[, 1]) + 0.5 * cos(x[, 2]) + rnorm(n, sd = noise_sd)
  list(x = x, y = y)
}

# Low-rank matrix for NMF: V = W_true * H_true.
make_low_rank_matrix <- function(m = 20, n = 15, rank = 3, seed = 123) {
  set.seed(seed)
  W_true <- matrix(runif(m * rank), nrow = m, ncol = rank)
  H_true <- matrix(runif(rank * n), nrow = rank, ncol = n)
  V <- W_true %*% H_true
  list(V = V, W_true = W_true, H_true = H_true)
}

# Grid points for KNN.
make_grid_data <- function(n = 100, d = 2, seed = 123) {
  set.seed(seed)
  x <- matrix(rnorm(n * d), ncol = d)
  x
}
