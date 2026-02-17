test_that("decision_tree() trains and predicts", {
  data <- make_step_data(n = 100)
  result <- decision_tree(training_data = data$x, training_labels = data$y,
                           test_data = data$x, max_depth = 5)
  expect_true("output_model" %in% names(result))
  expect_true("predictions" %in% names(result))
  expect_s3_class(result$output_model, "DecisionTreeRegressor")
  expect_length(result$predictions, 100)
})

test_that("decision_tree() trains only (no test_data)", {
  data <- make_step_data(n = 100)
  result <- decision_tree(training_data = data$x, training_labels = data$y,
                           max_depth = 5)
  expect_true("output_model" %in% names(result))
  expect_false("predictions" %in% names(result))
})

test_that("decision_tree() predicts with input_model", {
  data <- make_step_data(n = 100)
  model <- DecisionTreeRegressor(data$x, data$y, max_depth = 5)
  result <- decision_tree(input_model = model, test_data = data$x)
  expect_true("predictions" %in% names(result))
  expect_length(result$predictions, 100)
})

test_that("decision_tree() errors without model or training data", {
  expect_error(decision_tree(test_data = matrix(1:10, ncol = 2)),
               "Must provide")
})

test_that("random_forest() trains and predicts", {
  data <- make_sine_data(n = 100)
  result <- random_forest(training_data = data$x, training_labels = data$y,
                           test_data = data$x, n_trees = 10, seed = 42)
  expect_true("output_model" %in% names(result))
  expect_true("predictions" %in% names(result))
  expect_s3_class(result$output_model, "RandomForestRegressor")
  expect_length(result$predictions, 100)
})

test_that("random_forest() predicts with input_model", {
  data <- make_sine_data(n = 100)
  model <- RandomForestRegressor(data$x, data$y, n_trees = 10, seed = 42)
  result <- random_forest(input_model = model, test_data = data$x)
  expect_length(result$predictions, 100)
})
