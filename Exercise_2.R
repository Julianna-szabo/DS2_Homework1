# Clear memory
rm(list=ls())

# Libraries

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)
library(caret)
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

rf_10_var <- h2o.randomForest(
  X, y,
  training_frame = h2o_data,
  model_id = "rf_2_var",
  mtries = 10,
  seed = my_seed
)

## Comparison of variable importance

h2o.varimp_plot(rf_2_var)
h2o.varimp_plot(rf_10_var)

# Looks like the first 2 are the same for the two models.
# With just two variables all variables (except CAtBat) are more important.
# CRuns and CRBI also show up high on the list on both models. 
# Variables like Years and Hits show up in the first model but not the second.

# B, Explanation of extreme difference in variable importance

# In the one with 10 variables CAtBat is shown to be extremely more important than the others with a jump from almost 1 to 0.6
# This could be explained that when 10 variables are picked randomly then CAtBat will be included in more of the trees.
# Therefore, its importance crystallizes out easier.

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

h2o.varimp_plot(gbm_srate_01)
h2o.varimp_plot(gbm_srate_1)

# The one where the sample_rate is 1 has the more extreme values
# This is because bootstrapping always completely resampled the data and therefore build every unrelated trees.
# With a sample_rate equal to 0.1 the generated datasets are closed to each other since only 10% of the data gets replaced.
# This means the data will be more unform and increase variables importance.