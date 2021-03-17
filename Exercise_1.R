# Clear memory
rm(list=ls(
  all.names = TRUE))

# Libraries

library(tidyverse)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(xgboost)
library(randomForest)
library(data.tree)
theme_set(theme_minimal())

library(h2o)
h2o.init()
h2o.no_progress()
#h2o.shutdown()

my_seed <- (5678)

# Import data

data <- as_tibble(ISLR::OJ)
h2o_data <- as.h2o(data)

# A, Train and test data and build a decision tree for reference

## Train and test data
splitted_data <- h2o.splitFrame(h2o_data, ratios = 0.75, seed = my_seed)
data_train <- splitted_data[[1]]
data_test <- splitted_data[[2]]

## Simple decision tree

y <- "Purchase"
X <- setdiff(names(h2o_data), y)

gbm_params_1tree = list(max_depth = seq(2, 10))

# Train and validate a cartesian grid of GBMs
gbm_grid = h2o.grid("gbm", x = X, y = y,
                    grid_id = "gbm_grid_1tree",
                    training_frame = data_train,
                    validation_frame = data_test,
                    ntrees = 1, min_rows = 1, 
                    sample_rate = 1, col_sample_rate = 1,
                    learn_rate = .01, seed = my_seed,
                    hyper_params = gbm_params_1tree)

gbm_gridperf = h2o.getGrid(grid_id = "gbm_grid_1tree",
                           sort_by = "auc",
                           decreasing = TRUE)

ggplot(as.data.frame(sapply(gbm_gridperf@summary_table, as.numeric))) +
  geom_point(aes(max_depth, auc)) +
  geom_line(aes(max_depth, auc, group=1)) +
  labs(x="max depth", y="AUC", title="Grid Search for Single Tree Models")

data_1tree = 
  h2o.gbm(x = X, y = y, 
          training_frame = data_train,
          model_id = "data_1tree",
          ntrees = 1, min_rows = 1, 
          sample_rate = 1, col_sample_rate = 1,
          max_depth = 4,
          stopping_rounds = 3, stopping_tolerance = 0.01, 
          stopping_metric = "AUC", 
          seed = 1)

dataH2oTree = h2o.getModelTree(model = data_1tree, tree_number = 1)

source("Plotting_h2o_tree.R")

DataTree = createDataTree(dataH2oTree)

GetEdgeLabel <- function(node) {return (node$edgeLabel)}
GetNodeShape <- function(node) {switch(node$type, 
                                       split = "diamond", leaf = "oval")}
GetFontName <- function(node) {switch(node$type, 
                                      split = 'Palatino-bold', 
                                      leaf = 'Palatino')}
SetEdgeStyle(DataTree, fontname = 'Palatino-italic', 
             label = GetEdgeLabel, labelfloat = TRUE,
             fontsize = "26", fontcolor='royalblue4')
SetNodeStyle(DataTree, fontname = GetFontName, shape = GetNodeShape, 
             fontsize = "26", fontcolor='royalblue4',
             height="0.75", width="1")

SetGraphStyle(DataTree, rankdir = "LR", dpi=70.)

plot(DataTree, output = "graph")


# B, Tree ensemble models

## Random forest

rf_params <- list(
  ntrees = c(10, 50, 100, 300),
  mtries = c(2, 4, 6, 8),
  sample_rate = c(0.2, 0.632, 1),
  max_depth = c(10, 20)
)

rf_grid <- h2o.grid(
  "randomForest", 
  x = X, y = y,
  training_frame = data_train,
  grid_id = "rf",
  nfolds = 5,
  seed = my_seed,
  hyper_params = rf_params
)

h2o.getGrid(rf_grid@grid_id, "mae")
best_rf <- h2o.getModel(
  h2o.getGrid(rf_grid@grid_id, "mae")@model_ids[[1]]
)
h2o.mae(h2o.performance(best_rf, data_train))

## GBM

gbm_params <- list(
  learn_rate = c(0.01, 0.05, 0.1, 0.3),
  ntrees = c(10, 50, 100, 300),
  max_depth = c(2, 5),
  sample_rate = c(0.2, 0.5, 0.8, 1)
)

gbm_grid <- h2o.grid(
  "gbm", x = X, y = y,
  grid_id = "gbm",
  training_frame = data_train,
  nfolds = 5,
  seed = my_seed,
  hyper_params = gbm_params
)

(h2o.getGrid(grid_id = gbm_grid@grid_id, sort_by = "rmse", decreasing = TRUE))
best_gbm <- h2o.getModel(
  h2o.getGrid(grid_id = gbm_grid@grid_id, sort_by = "rmse", decreasing = TRUE)@model_ids[[1]]
)

best_gbm
h2o.rmse(h2o.performance(best_gbm))

## XG Boost

xgboost_params <- list(
  learn_rate = c(0.1, 0.3), 
  ntrees = c(50, 100),
  max_depth = c(2, 5),
  gamma = c(0, 1, 2),
  sample_rate = c(0.5, 1)
)

xgboost_grid <- h2o.grid(
  "xgboost", x = X, y = y,
  grid_id = "xgboost",
  training_frame = data_train,
  nfolds = 5,
  seed = my_seed,
  hyper_params = xgboost_params
)


h2o.getGrid(grid_id = xgboost_grid@grid_id, sort_by = "rmse", decreasing = TRUE)
best_xgboost <- h2o.getModel(
  h2o.getGrid(grid_id = xgboost_grid@grid_id, sort_by = "rmse", decreasing = TRUE)@model_ids[[1]]
)

best_xgboost
h2o.rmse(h2o.performance(best_xgboost))

# C, Model comparison

my_models <- list(
  data_1tree, best_rf, best_gbm, best_xgboost
)

rmse_validation <- map_df(my_models, ~{
  tibble(model = .@model_id, RMSE_train = h2o.rmse(h2o.performance(., data_train)),
         RMSE_test = h2o.rmse(h2o.performance(., data_test)),
         AUC_train = h2o.auc(h2o.performance(., data_train)),
         AUC_test = h2o.auc(h2o.performance(., data_test)))}) %>% 
  arrange(RMSE_train)

rmse_validation

# D, Plot ROC curve for the best model

plotROC <- function(performance_df) {
  ggplot(performance_df, aes(fpr, tpr, color = model)) +
    geom_path() +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    coord_fixed() +
    labs(x = "False Positive Rate", y = "True Positive Rate")
}

plotROC(  )

# E, Show variable importance

h2o.varimp_plot(   )
