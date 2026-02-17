test_that("RandomForestRegressor returns correct S3 class", {
  data <- make_sine_data(n = 100)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 42)
  expect_s3_class(model, "RandomForestRegressor")
})

test_that("RandomForestRegressor predictions have correct dimensions", {
  data <- make_sine_data(n = 100)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 42)
  preds <- predict(model, data$x)
  expect_length(preds, nrow(data$x))
})

test_that("RandomForestRegressor fits sine function well", {
  data <- make_sine_data(n = 200, noise_sd = 0.1)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 50,
                                  max_depth = 8, seed = 42)
  preds <- predict(model, data$x)
  expect_lt(mse(data$y, preds), 0.1)
})

test_that("RandomForestRegressor outperforms single tree (or is close)", {
  data <- make_sine_data(n = 200)
  dt <- DecisionTreeRegressor(data$x, data$y, max_depth = 5)
  rf <- RandomForestRegressor(data$x, data$y, n_trees = 50,
                               max_depth = 5, seed = 42)
  mse_dt <- mse(data$y, predict(dt, data$x))
  mse_rf <- mse(data$y, predict(rf, data$x))
  # RF should not be dramatically worse than single tree on train data
  expect_lt(mse_rf, mse_dt * 3)
})

test_that("RandomForestRegressor returns OOB MSE", {
  data <- make_sine_data(n = 200)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 50, seed = 42)
  expect_true(is.numeric(model$oob_mse))
  expect_gt(model$oob_mse, 0)
})

test_that("RandomForestRegressor input validation works", {
  expect_error(RandomForestRegressor(matrix(1:10, ncol = 2), 1:3),
               "must match")
  expect_error(RandomForestRegressor(matrix(1:10, ncol = 2), 1:5,
                                      n_trees = 0),
               "n_trees")
  expect_error(RandomForestRegressor(matrix(1:10, ncol = 2), 1:5,
                                      sample_fraction = 0),
               "sample_fraction")
})

test_that("print.RandomForestRegressor runs without error", {
  data <- make_sine_data(n = 50)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 5, seed = 42)
  expect_output(print(model), "RandomForestRegressor")
})

test_that("RandomForestRegressor with seed is deterministic", {
  data <- make_sine_data(n = 100)
  m1 <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 99)
  m2 <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 99)
  p1 <- predict(m1, data$x)
  p2 <- predict(m2, data$x)
  expect_equal(p1, p2)
})
