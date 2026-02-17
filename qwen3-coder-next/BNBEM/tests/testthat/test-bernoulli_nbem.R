test_that("BernoulliNBEM class creation works", {
  model <- BernoulliNBEM()
  expect_s3_class(model, "BernoulliNBEM")
  expect_equal(model$n_classes, 0)
  expect_equal(model$n_words, 0)
})

test_that("fit_bernoulli_nbem works with labeled data only", {
  # Create simple test data
  labeled_docs <- list(
    c("word1", "word2", "word3"),
    c("word1", "word4", "word5"),
    c("word2", "word3", "word6"),
    c("word4", "word5", "word6")
  )
  labels <- c("class1", "class1", "class2", "class2")
  
  model <- fit_bernoulli_nbem(
    labeled_data = list(documents = labeled_docs, labels = labels),
    max_iterations = 10
  )
  
  expect_s3_class(model, "BernoulliNBEM")
  expect_equal(model$n_classes, 2)
  expect_equal(model$n_words, 6)
  expect_true(model$convergence)
})

test_that("fit_bernoulli_nbem works with unlabeled data", {
  # Create test data with both labeled and unlabeled
  test_data <- create_test_data(n_labeled = 20, n_unlabeled = 30, n_words = 20, n_classes = 2)
  
  model <- fit_bernoulli_nbem(
    labeled_data = test_data$labeled_data,
    unlabeled_data = test_data$unlabeled_data,
    max_iterations = 20
  )
  
  expect_s3_class(model, "BernoulliNBEM")
  expect_equal(model$n_classes, 2)
  expect_equal(model$n_words, 20)
  expect_equal(nrow(model$unlabeled_probs), 30)
})

test_that("predict.BernoulliNBEM works", {
  # Create test data
  test_data <- create_test_data(n_labeled = 30, n_unlabeled = 0, n_words = 15, n_classes = 2)
  
  # Fit model
  model <- fit_bernoulli_nbem(
    labeled_data = test_data$labeled_data,
    max_iterations = 10
  )
  
  # Create new documents for prediction
  new_docs <- list(
    c("word1", "word2", "word3"),
    c("word10", "word11", "word12")
  )
  
  # Predict
  predictions <- predict(model, new_docs)
  
  expect_length(predictions$predictions, 2)
  expect_true(all(predictions$predictions %in% model$class_labels))
})

test_that("preprocess_documents works", {
  texts <- c("Hello, World! This is a test.",
             "Another test document with numbers 123.")
  
  documents <- preprocess_documents(texts)
  
  expect_length(documents, 2)
  expect_type(documents[[1]], "character")
  expect_true(all(documents[[1]] == tolower(documents[[1]])))
})

test_that("create_test_data generates valid data", {
  test_data <- create_test_data(n_labeled = 10, n_unlabeled = 20, n_words = 10, n_classes = 2)
  
  expect_length(test_data$labeled_data$documents, 20)
  expect_length(test_data$labeled_data$labels, 20)
  expect_length(test_data$unlabeled_data, 20)
  expect_length(test_data$true_params$vocabulary, 10)
  expect_equal(nrow(test_data$true_params$class_probs), 2)
})

test_that("EM algorithm converges", {
  test_data <- create_test_data(n_labeled = 50, n_unlabeled = 100, n_words = 30, n_classes = 3)
  
  model <- fit_bernoulli_nbem(
    labeled_data = test_data$labeled_data,
    unlabeled_data = test_data$unlabeled_data,
    max_iterations = 50,
    tolerance = 1e-5
  )
  
  expect_true(model$convergence)
  expect_gt(model$em_iterations, 0)
  expect_is(model$log_likelihood, "numeric")
})

test_that("summary method works", {
  test_data <- create_test_data(n_labeled = 20, n_unlabeled = 0, n_words = 15, n_classes = 2)
  
  model <- fit_bernoulli_nbem(
    labeled_data = test_data$labeled_data,
    max_iterations = 10
  )
  
  summary_result <- summary(model)
  
  expect_is(summary_result, "list")
  expect_length(summary_result, 6)
  expect_equal(summary_result$n_classes, 2)
  expect_equal(summary_result$vocabulary_size, 15)
})
