# Clear memory
rm(list=ls(Y))

# Libraries

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)
theme_set(theme_minimal())

library(h2o)
h2o.init()
h2o.no_progress()
#h2o.shutdown()

my_seed <- (5678)


# Load the data

data <- as_tibble(ISLR::Hitters) %>%
  drop_na(Salary) %>%
  mutate(log_salary = log(Salary), Salary = NULL)
h2o_data <- as.h2o(data)

# A, Train 2 Random Forest models

y <- "log_salary"
X <- setdiff(names(h2o_data), y)

## 1, Random forest with 2 variables

rf_2_var <- h2o.randomForest(
  X, y,
  training_frame = h2o_data,
  model_id = "rf_2_var",
  mtries = 2,
  seed = my_seed
)

## 2, Random forest with 10 variables

rf_2_var <- h2o.randomForest(
  X, y,
  training_frame = h2o_data,
  model_id = "rf_2_var",
  mtries = 10,
  seed = my_seed
)

## Comparison of variable importance


# B, Explanation of extreme difference in variable importance




# C, Two GMB models 

## GBM sample_rate = 0.1

gbm_srate_01 <- h2o.gbm(
  x = X, y = y,
  model_id = "gbm_srate_01",
  training_frame = h2o_data,
  sample_rate = 0.1,
  seed = my_seed
)

## GBM sample_rate = 1

gbm_srate_1 <- h2o.gbm(
  x = X, y = y,
  model_id = "gbm_srate_1",
  training_frame = h2o_data,
  sample_rate = 1,
  seed = my_seed
)

## Comparison of variable importance
