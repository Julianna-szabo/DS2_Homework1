# Clear memory
rm(list=ls())

# Libraries

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)
library(DiagrammeR)
library(plotROC)
theme_set(theme_minimal())

library(h2o)
h2o.init()
h2o.no_progress()
#h2o.shutdown()

my_seed <- (5678)

# Load the data

data <- read_csv("KaggleV2-May-2016.csv")

# some data cleaning
data <- select(data, -one_of(c("PatientId", "AppointmentID", "Neighbourhood"))) %>%
  janitor::clean_names()

# for binary prediction, the target variable must be a factor + generate new variables
data <- mutate(
  data,
  no_show = factor(no_show, levels = c("Yes", "No")),
  handcap = ifelse(handcap > 0, 1, 0),
  across(c(gender, scholarship, hipertension, alcoholism, handcap), factor),
  hours_since_scheduled = as.numeric(appointment_day - scheduled_day)
)

# clean up a little bit
data <- filter(data, between(age, 0, 95), hours_since_scheduled >= 0) %>%
  select(-one_of(c("scheduled_day", "appointment_day", "sms_received")))

h2o_data <- as.h2o(data)

# A, Create train / validation / test sets, cutting the data into 5% - 45% - 50% parts

splitted_data <- h2o.splitFrame(h2o_data, ratios = c(0.05, 0.45), seed = my_seed)
data_train <- splitted_data[[1]]
data_valid <- splitted_data[[2]]
data_test <- splitted_data[[3]]

# B, Train a benchmark model of your choice (such as random forest, gbm or glm)
# evaluate it on the validation set.

y <- "no_show"
X <- setdiff(names(h2o_data), y)

glm_model <- h2o.glm(
  X, y,
  training_frame = data_train,
  model_id = "lm",
  lambda = 0,
  nfolds = 5,
  seed = my_seed,
  keep_cross_validation_predictions = TRUE
)

glm_model

h2o.rmse(h2o.performance(glm_model))
h2o.rmse(h2o.performance(glm_model, data_valid))

# C, Build at least 3 models of different families using cross validation, keeping cross validated predictions.
# You might also try deeplearning.

## Random forest

rf_model <- h2o.randomForest(
  X, y,
  training_frame = data_train,
  model_id = "rf",
  ntrees = 200,
  max_depth = 10,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

## GBM

gbm_model <- h2o.gbm(
  X, y,
  training_frame = data_train,
  model_id = "gbm",
  ntrees = 200,
  max_depth = 5,
  learn_rate = 0.1,
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

## Deep learning

deeplearning_model <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  model_id = "deeplearning",
  hidden = c(32, 8),
  seed = my_seed,
  nfolds = 5,
  keep_cross_validation_predictions = TRUE
)

# D, Evaluate validation set performance of each model.

getPerformanceMetrics <- function(model, newdata = NULL, xval = FALSE) {
  h2o.performance(model, newdata = newdata, xval = xval)@metrics$thresholds_and_metric_scores %>%
    as_tibble() %>%
    mutate(model = model@model_id)
}

plotROC <- function(performance_df) {
  ggplot(performance_df, aes(fpr, tpr, color = model)) +
    geom_path() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    coord_fixed() +
    labs(x = "False Positive Rate", y = "True Positive Rate")
}

plotRP <- function(performance_df) {
  ggplot(performance_df, aes(precision, tpr, color = model)) +  # tpr = recall
    geom_line() +
    labs(x = "Precision", y = "Recall (TPR)")
}

my_models <- list(glm_model, rf_model, gbm_model, deeplearning_model)
all_performance <- map_df(c(my_models), getPerformanceMetrics, xval = TRUE)
plotROC(all_performance)
plotRP(all_performance)

# E, How large are the correlations of predicted scores of the validation set produced by the base learners?



# F, Create a stacked ensemble model from the base learners.

ensemble_model_glm <- h2o.stackedEnsemble(
  X, y,
  training_frame = data_train,
  metalearner_algorithm = "glm",
  model_id = "stacked_model",
  base_models = my_models
)

# G, Evaluate ensembles on validation set. Did it improve prediction?

map_df(
  c(my_models, ensemble_model_glm),
  ~{tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_valid)))}
)

# H, Evaluate the best performing model on the test set
# How does performance compare to that of the validation set?

map_df(
  c(ensemble_model_glm),
  ~{tibble(model = .@model_id, auc = h2o.auc(h2o.performance(., newdata = data_test)))}
)
