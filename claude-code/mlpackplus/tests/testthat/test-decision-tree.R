test_that("DecisionTreeRegressor returns correct S3 class", {
  data <- make_step_data(n = 100)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 5)
  expect_s3_class(model, "DecisionTreeRegressor")
})

test_that("DecisionTreeRegressor predictions have correct dimensions", {
  data <- make_step_data(n = 100)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 5)
  preds <- predict(model, data$x)
  expect_length(preds, nrow(data$x))
})

test_that("DecisionTreeRegressor fits step function well", {
  data <- make_step_data(n = 200, noise_sd = 0.05)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 5)
  preds <- predict(model, data$x)
  expect_lt(mse(data$y, preds), 0.01)
})

test_that("DecisionTreeRegressor with depth 0 predicts mean", {
  data <- make_step_data(n = 100, d = 2)
  # max_depth = 1 (just root + 1 split), but min_leaf_size very large
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 1,
                                  min_leaf_size = 100)
  preds <- predict(model, data$x)
  # All predictions should be (approximately) the same: the mean
  expect_equal(length(unique(round(preds, 6))), 1)
  expect_equal(preds[1], mean(data$y), tolerance = 0.01)
})

test_that("MSE decreases with increasing depth", {
  data <- make_sine_data(n = 200)
  model_shallow <- DecisionTreeRegressor(data$x, data$y, max_depth = 2)
  model_deep <- DecisionTreeRegressor(data$x, data$y, max_depth = 10)
  mse_shallow <- mse(data$y, predict(model_shallow, data$x))
  mse_deep <- mse(data$y, predict(model_deep, data$x))
  expect_lt(mse_deep, mse_shallow)
})

test_that("DecisionTreeRegressor input validation works", {
  expect_error(DecisionTreeRegressor(matrix(1:10, ncol = 2), 1:3),
               "must match")
  expect_error(DecisionTreeRegressor(matrix(1:10, ncol = 2), 1:5,
                                      max_depth = 0),
               "max_depth")
  expect_error(DecisionTreeRegressor(matrix(1:10, ncol = 2), 1:5,
                                      min_leaf_size = 0),
               "min_leaf_size")
})

test_that("predict.DecisionTreeRegressor validates column count", {
  data <- make_step_data(n = 100, d = 2)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 3)
  expect_error(predict(model, matrix(1:10, ncol = 1)), "columns")
})

test_that("print.DecisionTreeRegressor runs without error", {
  data <- make_step_data(n = 50)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 3)
  expect_output(print(model), "DecisionTreeRegressor")
})
