library(ggplot2)
library(ggmosaic)
library(tidyverse)
library(tidymodels)
library(vroom)
library(embed)

train_amazon <- vroom("train.csv") %>%
  mutate(ACTION = factor(ACTION))
test_amazon <- vroom("test.csv")


# EXPLANATORY PLOTS 
# ==================================================

# ggplot(data = train_amazon) +
#   geom_mosaic(aes(x = product(factor(ROLE_FAMILY)), fill = ROLE_ROLLUP_1)) +
#   labs(title = "Mosaic Plot of ROLE_ROLLUP_1 vs ACTION",
#        x = "ROLE_ROLLUP_1",
#        y = "Proportion",
#        fill = "ACTION")
# 
# 
# ggplot(train_amazon, aes(x = as.factor(ROLE_FAMILY))) +
#   geom_bar(fill = "steelblue") +
#   labs(title = "Counts of ACTION (1 = Granted, 0 = Denied)",
#        x = "ACTION",
#        y = "Count")




# LOGISTIC REGRESSION
# ============================================================================

# amazon_recipe <- recipe(ACTION ~ ., data = train_amazon) %>%
#   step_mutate_at(all_numeric_predictors(), fn=factor) %>%
#   step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
#   step_dummy(all_nominal_predictors())
# 
#   
# prep <- prep(amazon_recipe)
# baked <- bake(prep, new_data = train_amazon)
# 
# logistic_model <- logistic_reg() %>%
#   set_engine("glm")
# 
# amazon_workflow <- workflow() %>%
#     add_recipe(amazon_recipe) %>%
#     add_model(logistic_model) %>%
#     fit(data = train_amazon)
# 
# amazon_predictions <- amazon_workflow %>%
#   predict(new_data = test_amazon, type = "prob") %>%
#   bind_cols(test_amazon %>% select(id)) %>%
#   select(id, .pred_1) %>%
#   rename(action = .pred_1)

# we want the second column; probability of a 1
# amazon_predictions[2]

# vroom_write(amazon_predictions, "logistic_regression_predictions.csv", delim = ',') 



# PENALIZED LOGISTIC REGRESSION
# ============================================================================

amazon_recipe <- recipe(ACTION ~ ., data = train_amazon) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_other(all_nominal_predictors(), threshold = 0.001, other = "other") %>%
  #step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_nominal_predictors())

pen_log_mod <- logistic_reg(mixture = tune(), penalty = tune()) %>%
  set_engine("glmnet")

amazon_workflow <- workflow() %>%
  add_recipe(amazon_recipe) %>%
  add_model(pen_log_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) # L^2 total tuning possibilities

folds <- vfold_cv(train_amazon, v = 3, repeats = 1)

CV_results <- amazon_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))
# roc_auc, f_meas, sens, recall, spec, precision, accuracy

bestTune <- CV_results %>% select_best(metric = "roc_auc")

final_wf <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train_amazon)

amazon_predictions <- final_wf %>%
  predict(new_data = test_amazon, type = "prob") %>%
  bind_cols(test_amazon %>% select(id)) %>%
  select(id, .pred_1) %>%
  rename(action = .pred_1)

vroom_write(amazon_predictions, "pen_log_reg_predictions.csv", delim = ',') 



