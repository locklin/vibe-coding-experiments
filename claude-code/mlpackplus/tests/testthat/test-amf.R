test_that("AMF returns correct S3 class", {
  data <- make_low_rank_matrix()
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_s3_class(model, "AMF")
})

test_that("AMF factors have correct dimensions", {
  data <- make_low_rank_matrix(m = 20, n = 15, rank = 3)
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_equal(dim(model$W), c(20, 3))
  expect_equal(dim(model$H), c(3, 15))
})

test_that("AMF recovers low-rank matrix", {
  data <- make_low_rank_matrix(m = 20, n = 15, rank = 3)
  model <- AMF(data$V, rank = 3, max_iter = 200, seed = 42)
  recon <- predict(model)
  rel_error <- sqrt(sum((data$V - recon)^2)) / sqrt(sum(data$V^2))
  expect_lt(rel_error, 0.05)
})

test_that("AMF factors are non-negative", {
  data <- make_low_rank_matrix()
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_true(all(model$W >= 0))
  expect_true(all(model$H >= 0))
})

test_that("AMF final residual is less than initial", {
  data <- make_low_rank_matrix()
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_lt(model$final_residual, model$initial_residual)
})

test_that("AMF input validation works", {
  V <- matrix(1:20, nrow = 4, ncol = 5)
  expect_error(AMF(V, rank = 0), "rank must be >= 1")
  expect_error(AMF(V, rank = 5), "rank must be <= min")
  expect_error(AMF(V, rank = 2, max_iter = 0), "max_iter")
})

test_that("AMF predict returns reconstruction", {
  data <- make_low_rank_matrix()
  model <- AMF(data$V, rank = 3, seed = 42)
  recon <- predict(model)
  expect_equal(dim(recon), dim(data$V))
  # Should equal W %*% H
  expect_equal(recon, model$W %*% model$H)
})

test_that("print.AMF runs without error", {
  data <- make_low_rank_matrix()
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_output(print(model), "AMF")
})
