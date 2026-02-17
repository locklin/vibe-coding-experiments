test_that("update.DecisionTreeRegressor errors informatively", {
  data <- make_step_data(n = 50)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 3)
  expect_error(update(model), "does not support incremental updates")
})

test_that("update.RandomForestRegressor adds new trees", {
  data <- make_sine_data(n = 100)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 42)
  expect_equal(model$n_trees, 10)

  new_data <- make_sine_data(n = 50, seed = 999)
  updated <- update(model, new_data$x, new_data$y, n_new_trees = 5)
  expect_equal(updated$n_trees, 15)
  expect_equal(length(updated$trees), 15)
  expect_s3_class(updated, "RandomForestRegressor")
})

test_that("update.RandomForestRegressor predictions work", {
  data <- make_sine_data(n = 100)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 42)

  new_data <- make_sine_data(n = 50, seed = 999)
  updated <- update(model, new_data$x, new_data$y, n_new_trees = 5)

  preds <- predict(updated, data$x)
  expect_length(preds, 100)
  expect_true(is.numeric(preds))
})

test_that("update.RandomForestRegressor validates column count", {
  data <- make_sine_data(n = 100)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 42)
  expect_error(update(model, matrix(1:30, ncol = 3), 1:10),
               "columns")
})

test_that("update.AMF extends with new rows", {
  data <- make_low_rank_matrix(m = 20, n = 15, rank = 3)
  model <- AMF(data$V, rank = 3, seed = 42)

  set.seed(99)
  new_rows <- matrix(runif(15 * 3), nrow = 3, ncol = 15)
  updated <- update(model, new_rows = new_rows)

  expect_equal(nrow(updated$W), 23)  # 20 + 3
  expect_equal(ncol(updated$H), 15)  # unchanged
  expect_s3_class(updated, "AMF")
})

test_that("update.AMF extends with new columns", {
  data <- make_low_rank_matrix(m = 20, n = 15, rank = 3)
  model <- AMF(data$V, rank = 3, seed = 42)

  set.seed(99)
  new_cols <- matrix(runif(20 * 4), nrow = 20, ncol = 4)
  updated <- update(model, new_cols = new_cols)

  expect_equal(nrow(updated$W), 20)  # unchanged
  expect_equal(ncol(updated$H), 19)  # 15 + 4
})

test_that("update.AMF validates dimensions", {
  data <- make_low_rank_matrix(m = 20, n = 15, rank = 3)
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_error(update(model, new_rows = matrix(1:10, ncol = 2)),
               "columns")
  expect_error(update(model, new_cols = matrix(1:10, nrow = 5)),
               "rows")
})

test_that("update.AMF errors without new data", {
  data <- make_low_rank_matrix()
  model <- AMF(data$V, rank = 3, seed = 42)
  expect_error(update(model), "Must provide")
})
