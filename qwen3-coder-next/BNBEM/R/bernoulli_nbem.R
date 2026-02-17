#' Bernoulli Naive Bayes with EM Algorithm
#'
#' @description
#' This package implements a Bernoulli Naive Bayes classifier with Expectation-Maximization (EM) algorithm
#' for semi-supervised learning. It can handle both labeled and unlabeled documents.
#'
#' @details
#' The model represents documents by presence/absence of word stems (Bernoulli distribution).
#' For unlabeled documents, the EM algorithm iteratively estimates class probabilities and updates
#' model parameters to improve classification accuracy.
#'
#' @name bnbem-package
#' @keywords internal
#' "_PACKAGE"
#' @import methods
#' @import stats
#' @import tm
#' @import SnowballC
NULL

#' @importFrom utils globalVariables
utils::globalVariables(c("word", "class", "doc_id", "count", "prob"))

# S3 class definitions
#' BNBEM Model Object
#'
#' @description
#' S3 class representing a Bernoulli Naive Bayes model with EM algorithm
#'
#' @slot vocabulary Character vector of word stems
#' @slot class_priors Numeric vector of class prior probabilities
#' @slot word_probs List of matrices containing P(word|class) for each class
#' @slot n_classes Integer number of classes
#' @slot n_words Integer number of words in vocabulary
#' @slot em_iterations Integer number of EM iterations performed
#' @slot log_likelihood Numeric final log-likelihood
#' @slot class_labels Character vector of class names
#' @slot unlabeled_probs Matrix of class probabilities for unlabeled documents
#' @slot convergence Logical indicating if EM algorithm converged
#' @slot tolerance Numeric tolerance for convergence
#' @slot max_iterations Maximum number of EM iterations
#' @slot seed Random seed for reproducibility
#' @export
BNBEM <- function(vocabulary = character(0),
                  class_priors = numeric(0),
                  word_probs = list(),
                  n_classes = 0,
                  n_words = 0,
                  em_iterations = 0,
                  log_likelihood = NA_real_,
                  class_labels = character(0),
                  unlabeled_probs = matrix(0, 0, 0),
                  convergence = FALSE,
                  tolerance = 1e-6,
                  max_iterations = 100,
                  seed = 42) {
  structure(list(
    vocabulary = vocabulary,
    class_priors = class_priors,
    word_probs = word_probs,
    n_classes = n_classes,
    n_words = n_words,
    em_iterations = em_iterations,
    log_likelihood = log_likelihood,
    class_labels = class_labels,
    unlabeled_probs = unlabeled_probs,
    convergence = convergence,
    tolerance = tolerance,
    max_iterations = max_iterations,
    seed = seed
  ), class = "BNBEM")
}

#' Print method for BNBEM objects
#'
#' @param x BNBEM object
#' @param ... Additional arguments (unused)
#' @export
print.BNBEM <- function(x, ...) {
  cat("Bernoulli Naive Bayes Model with EM Algorithm\n")
  cat("============================================\n")
  cat("Number of classes:", x$n_classes, "\n")
  cat("Vocabulary size:", x$n_words, "\n")
  cat("EM iterations:", x$em_iterations, "\n")
  cat("Convergence:", x$convergence, "\n")
  cat("Class labels:", paste(x$class_labels, collapse = ", "), "\n")
  invisible(x)
}

#' Summary method for BNBEM objects
#'
#' @param object BNBEM object
#' @param ... Additional arguments (unused)
#' @export
summary.BNBEM <- function(object, ...) {
  list(
    n_classes = object$n_classes,
    vocabulary_size = object$n_words,
    em_iterations = object$em_iterations,
    convergence = object$convergence,
    log_likelihood = object$log_likelihood,
    class_priors = object$class_priors,
    class_labels = object$class_labels
  )
}

#' Fit Bernoulli Naive Bayes model with EM algorithm
#'
#' @description
#' Fit a Bernoulli Naive Bayes model using labeled data and optionally unlabeled data
#' with the EM algorithm for parameter estimation.
#'
#' @param labeled_data List containing:
#'   - documents: List of character vectors (word stems) for each document
#'   - labels: Character vector of class labels for each document
#' @param unlabeled_data Optional list of unlabeled documents (list of character vectors)
#' @param vocabulary Optional predefined vocabulary. If NULL, will be built from data
#' @param max_iterations Maximum number of EM iterations (default: 100)
#' @param tolerance Convergence tolerance (default: 1e-6)
#' @param seed Random seed for reproducibility (default: 42)
#' @param smoothing Numeric smoothing parameter (Laplace smoothing, default: 1)
#' @return BNBEM object
#' @export
fit_bnbem <- function(labeled_data,
                      unlabeled_data = NULL,
                      vocabulary = NULL,
                      max_iterations = 100,
                      tolerance = 1e-6,
                      seed = 42,
                      smoothing = 1) {
  
  # Extract labeled documents and labels
  labeled_docs <- labeled_data$documents
  labels <- labeled_data$labels
  
  # Build vocabulary if not provided
  if (is.null(vocabulary)) {
    vocabulary <- build_vocabulary(c(labeled_docs, if (!is.null(unlabeled_data)) unlabeled_data))
  }
  
  # Get unique classes
  classes <- unique(labels)
  n_classes <- length(classes)
  n_labeled <- length(labeled_docs)
  
  # Initialize class priors
  class_priors <- numeric(n_classes)
  names(class_priors) <- classes
  for (i in seq_along(classes)) {
    class_priors[classes[i]] <- sum(labels == classes[i]) / n_labeled
  }
  
  # Initialize word probabilities for each class
  word_probs <- vector("list", n_classes)
  names(word_probs) <- classes
  
  # Count word occurrences per class
  word_counts <- matrix(0, nrow = length(vocabulary), ncol = n_classes)
  colnames(word_counts) <- classes
  rownames(word_counts) <- vocabulary
  
  # Count word presence for labeled documents
  for (i in seq_along(labeled_docs)) {
    doc <- labeled_docs[[i]]
    class_label <- labels[i]
    for (word in vocabulary) {
      if (word %in% doc) {
        word_counts[word, class_label] <- word_counts[word, class_label] + 1
      }
    }
  }
  
  # Calculate word probabilities with smoothing
  for (j in seq_along(classes)) {
    class_name <- classes[j]
    total_words <- sum(word_counts[, class_name]) + smoothing * length(vocabulary)
    word_probs[[class_name]] <- (word_counts[, class_name] + smoothing) / total_words
  }
  
  # Initialize result object
  model <- BNBEM(
    vocabulary = vocabulary,
    class_priors = class_priors,
    word_probs = word_probs,
    n_classes = n_classes,
    n_words = length(vocabulary),
    em_iterations = 0,
    log_likelihood = NA_real_,
    class_labels = classes,
    unlabeled_probs = matrix(0, 0, 0),
    convergence = FALSE,
    tolerance = tolerance,
    max_iterations = max_iterations,
    seed = seed
  )
  
  # If no unlabeled data, return the model trained on labeled data only
  if (is.null(unlabeled_data) || length(unlabeled_data) == 0) {
    model$convergence <- TRUE
    model$em_iterations <- 0
    model$log_likelihood <- calculate_log_likelihood(model, labeled_docs, labels)
    return(model)
  }
  
  # EM Algorithm for unlabeled data
  set.seed(seed)
  model <- em_algorithm(model, labeled_docs, labels, unlabeled_data)
  
  return(model)
}

#' Build vocabulary from documents
#'
#' @param documents List of character vectors (word stems)
#' @return Character vector of unique words
#' @noRd
build_vocabulary <- function(documents) {
  all_words <- unlist(documents)
  unique_words <- unique(all_words)
  sort(unique_words)
}

#' EM algorithm for semi-supervised learning
#'
#' @param model Initial BNBEM model
#' @param labeled_docs List of labeled document word stems
#' @param labels Character vector of class labels
#' @param unlabeled_docs List of unlabeled document word stems
#' @return Updated BNBEM model
#' @noRd
em_algorithm <- function(model, labeled_docs, labels, unlabeled_docs) {
  tolerance <- model$tolerance
  max_iterations <- model$max_iterations
  vocab <- model$vocabulary
  n_labeled <- length(labeled_docs)
  n_unlabeled <- length(unlabeled_docs)
  n_docs <- n_labeled + n_unlabeled
  
  # Initialize class probabilities for unlabeled documents
  unlabeled_probs <- matrix(0, nrow = n_unlabeled, ncol = model$n_classes)
  colnames(unlabeled_probs) <- model$class_labels
  
  # Initialize with uniform distribution
  for (i in seq_len(n_unlabeled)) {
    unlabeled_probs[i, ] <- rep(1 / model$n_classes, model$n_classes)
  }
  
  prev_log_likelihood <- -Inf
  converged <- FALSE
  iterations <- 0
  
  for (iter in 1:max_iterations) {
    iterations <- iter
    
    # E-step: Calculate expected class assignments for unlabeled documents
    for (i in seq_len(n_unlabeled)) {
      doc <- unlabeled_docs[[i]]
      for (j in seq_len(model$n_classes)) {
        class_name <- model$class_labels[j]
        log_prob <- log(model$class_priors[class_name])
        
        for (word in vocab) {
          if (word %in% doc) {
            log_prob <- log_prob + log(model$word_probs[[class_name]][word])
          } else {
            log_prob <- log_prob + log(1 - model$word_probs[[class_name]][word])
          }
        }
        
        unlabeled_probs[i, j] <- exp(log_prob)
      }
      
      # Normalize
      total <- sum(unlabeled_probs[i, ])
      if (total > 0) {
        unlabeled_probs[i, ] <- unlabeled_probs[i, ] / total
      }
    }
    
    # M-step: Update parameters
    # Update class priors
    class_counts <- numeric(model$n_classes)
    names(class_counts) <- model$class_labels
    
    # From labeled data
    for (i in seq_len(n_labeled)) {
      class_counts[labels[i]] <- class_counts[labels[i]] + 1
    }
    
    # From unlabeled data (expected counts)
    for (i in seq_len(n_unlabeled)) {
      class_counts <- class_counts + unlabeled_probs[i, ]
    }
    
    model$class_priors <- class_counts / n_docs
    
    # Update word probabilities
    for (j in seq_len(model$n_classes)) {
      class_name <- model$class_labels[j]
      word_sum <- numeric(length(vocab))
      names(word_sum) <- vocab
      
      # From labeled data
      for (i in seq_len(n_labeled)) {
        if (labels[i] == class_name) {
          doc <- labeled_docs[[i]]
          for (word in vocab) {
            if (word %in% doc) {
              word_sum[word] <- word_sum[word] + 1
            }
          }
        }
      }
      
      # From unlabeled data (expected counts)
      for (i in seq_len(n_unlabeled)) {
        doc <- unlabeled_docs[[i]]
        weight <- unlabeled_probs[i, class_name]
        for (word in vocab) {
          if (word %in% doc) {
            word_sum[word] <- word_sum[word] + weight
          }
        }
      }
      
      # Calculate probability with smoothing
      total <- sum(word_sum) + model$n_words
      model$word_probs[[class_name]] <- (word_sum + 1) / total
    }
    
    # Calculate log-likelihood
    current_log_likelihood <- calculate_log_likelihood(model, c(labeled_docs, unlabeled_docs), 
                                                       c(labels, rep(NA, n_unlabeled)))
    
    # Check convergence
    if (abs(current_log_likelihood - prev_log_likelihood) < tolerance) {
      converged <- TRUE
      break
    }
    
    prev_log_likelihood <- current_log_likelihood
  }
  
  model$em_iterations <- iterations
  model$log_likelihood <- prev_log_likelihood
  model$convergence <- converged
  model$unlabeled_probs <- unlabeled_probs
  
  return(model)
}

#' Calculate log-likelihood of the model given data
#'
#' @param model BNBEM model
#' @param documents List of document word stems
#' @param labels Character vector of class labels (can contain NA for unlabeled)
#' @return Numeric log-likelihood
#' @noRd
calculate_log_likelihood <- function(model, documents, labels) {
  log_likelihood <- 0
  vocab <- model$vocabulary
  
  for (i in seq_along(documents)) {
    doc <- documents[[i]]
    
    if (!is.na(labels[i])) {
      # Labeled document
      class_name <- labels[i]
      log_prob <- log(model$class_priors[class_name])
      
      for (word in vocab) {
        if (word %in% doc) {
          log_prob <- log_prob + log(model$word_probs[[class_name]][word])
        } else {
          log_prob <- log_prob + log(1 - model$word_probs[[class_name]][word])
        }
      }
      
      log_likelihood <- log_likelihood + log_prob
    } else {
      # Unlabeled document - marginalize over classes
      class_probs <- numeric(model$n_classes)
      names(class_probs) <- model$class_labels
      
      for (j in seq_len(model$n_classes)) {
        class_name <- model$class_labels[j]
        log_prob <- log(model$class_priors[class_name])
        
        for (word in vocab) {
          if (word %in% doc) {
            log_prob <- log_prob + log(model$word_probs[[class_name]][word])
          } else {
            log_prob <- log_prob + log(1 - model$word_probs[[class_name]][word])
          }
        }
        
        class_probs[class_name] <- exp(log_prob)
      }
      
      # Sum probabilities (in log space to avoid underflow)
      max_prob <- max(class_probs)
      if (max_prob > 0) {
        log_sum <- log(sum(class_probs / max_prob)) + log(max_prob)
        log_likelihood <- log_likelihood + log_sum
      }
    }
  }
  
  return(log_likelihood)
}

#' Predict class for new documents
#'
#' @param object BNBEM model
#' @param new_data List of new document word stems
#' @param return_probs Logical indicating if class probabilities should be returned
#' @return List containing:
#'   - predictions: Character vector of predicted classes
#'   - probabilities: Matrix of class probabilities (if return_probs = TRUE)
#' @export
predict.BNBEM <- function(object, new_data, return_probs = FALSE) {
  vocab <- object$vocabulary
  n_docs <- length(new_data)
  predictions <- character(n_docs)
  
  if (return_probs) {
    probabilities <- matrix(0, nrow = n_docs, ncol = object$n_classes)
    colnames(probabilities) <- object$class_labels
  }
  
  for (i in seq_len(n_docs)) {
    doc <- new_data[[i]]
    class_scores <- numeric(object$n_classes)
    names(class_scores) <- object$class_labels
    
    for (j in seq_len(object$n_classes)) {
      class_name <- object$class_labels[j]
      log_prob <- log(object$class_priors[class_name])
      
      for (word in vocab) {
        if (word %in% doc) {
          log_prob <- log_prob + log(object$word_probs[[class_name]][word])
        } else {
          log_prob <- log_prob + log(1 - object$word_probs[[class_name]][word])
        }
      }
      
      class_scores[class_name] <- exp(log_prob)
    }
    
    # Normalize to get probabilities
    total <- sum(class_scores)
    if (total > 0) {
      class_scores <- class_scores / total
    }
    
    predictions[i] <- names(which.max(class_scores))
    
    if (return_probs) {
      probabilities[i, ] <- class_scores
    }
  }
  
  result <- list(predictions = predictions)
  if (return_probs) {
    result$probabilities <- probabilities
  }
  
  return(result)
}

#' Preprocess text documents
#'
#' @description
#' Convert raw text documents to word stems for use with BNBEM
#'
#' @param texts Character vector of raw text documents
#' @param language Language for stemming (default: "en")
#' @return List of character vectors (word stems) for each document
#' @export
preprocess_documents <- function(texts, language = "en") {
  documents <- vector("list", length(texts))
  
  for (i in seq_along(texts)) {
    # Convert to lowercase
    text <- tolower(texts[i])
    
    # Remove punctuation
    text <- gsub("[[:punct:]]", " ", text)
    
    # Remove numbers
    text <- gsub("[0-9]", " ", text)
    
    # Remove extra whitespace
    text <- gsub("\\s+", " ", text)
    
    # Split into words
    words <- strsplit(text, "\\s+")[[1]]
    
    # Remove empty strings
    words <- words[words != ""]
    
    # Stem words
    stemmed_words <- SnowballC::wordStem(words, language = language)
    
    documents[[i]] <- stemmed_words
  }
  
  return(documents)
}

#' Create test data for BNBEM
#'
#' @description
#' Generate artificial Bernoulli distribution vectors correlated with classes
#'
#' @param n_labeled Number of labeled documents per class
#' @param n_unlabeled Number of unlabeled documents
#' @param n_words Number of words in vocabulary
#' @param n_classes Number of classes
#' @param class_separation Numeric value controlling class separation (higher = more separable)
#' @param seed Random seed (default: 42)
#' @return List containing:
#'   - labeled_data: List with documents and labels
#'   - unlabeled_data: List of unlabeled documents
#'   - true_params: List of true parameters used to generate data
#' @export
create_test_data <- function(n_labeled = 50,
                             n_unlabeled = 100,
                             n_words = 100,
                             n_classes = 3,
                             class_separation = 2,
                             seed = 42) {
  set.seed(seed)
  
  # Generate vocabulary
  vocabulary <- paste0("word", 1:n_words)
  
  # Generate class-specific word probabilities
  class_probs <- matrix(0, nrow = n_classes, ncol = n_words)
  rownames(class_probs) <- paste0("class", 1:n_classes)
  colnames(class_probs) <- vocabulary
  
  for (i in 1:n_classes) {
    # Create class-specific word distributions with some overlap
    base_prob <- runif(n_words, 0.1, 0.9)
    
    # Make classes more distinct based on class_separation parameter
    for (j in 1:n_classes) {
      if (i != j) {
        base_prob <- base_prob * (1 - class_separation / (n_classes * 10))
      }
    }
    
    # Ensure probabilities are between 0 and 1
    class_probs[i, ] <- pmin(pmax(base_prob, 0.05), 0.95)
  }
  
  # Generate labeled documents
  labeled_docs <- vector("list", n_labeled * n_classes)
  labels <- character(n_labeled * n_classes)
  
  doc_idx <- 1
  for (i in 1:n_classes) {
    class_name <- paste0("class", i)
    
    for (j in 1:n_labeled) {
      # Generate document based on class probabilities
      doc_words <- character(0)
      
      for (word in vocabulary) {
        if (runif(1) < class_probs[i, word]) {
          doc_words <- c(doc_words, word)
        }
      }
      
      labeled_docs[[doc_idx]] <- doc_words
      labels[doc_idx] <- class_name
      doc_idx <- doc_idx + 1
    }
  }
  
  # Generate unlabeled documents
  unlabeled_docs <- vector("list", n_unlabeled)
  
  for (i in 1:n_unlabeled) {
    # Randomly select a class for this document
    class_idx <- sample(1:n_classes, 1)
    class_name <- paste0("class", class_idx)
    
    # Generate document based on class probabilities
    doc_words <- character(0)
    
    for (word in vocabulary) {
      if (runif(1) < class_probs[class_idx, word]) {
        doc_words <- c(doc_words, word)
      }
    }
    
    unlabeled_docs[[i]] <- doc_words
  }
  
  return(list(
    labeled_data = list(documents = labeled_docs, labels = labels),
    unlabeled_data = unlabeled_docs,
    true_params = list(
      vocabulary = vocabulary,
      class_probs = class_probs,
      class_priors = rep(1/n_classes, n_classes)
    )
  ))
}
