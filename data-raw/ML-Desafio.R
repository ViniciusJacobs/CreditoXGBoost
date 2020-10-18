library(tidyverse)
library(tidymodels)
library(ggplot2)
library(skimr)
library(stats)
library(doParallel)
cores = 5


base_adult <- read_rds("data/adult.rds")

glimpse(base_adult)


base_adult$workclass <- as.factor(base_adult$workclass)
base_adult$education <- as.factor(base_adult$education)
base_adult$marital_status <- as.factor(base_adult$marital_status)
base_adult$occupation <- as.factor(base_adult$occupation)
base_adult$relationship <- as.factor(base_adult$relationship)
base_adult$race <- as.factor(base_adult$race)
base_adult$sex <- as.factor(base_adult$sex)


base_adult <- base_adult %>%
  tidyr::replace_na(replace = list(native_country = "United-States")) %>%
  tidyr::replace_na(replace = list(workclass = "Private")) %>%
  tidyr::replace_na(replace = list(occupation = "Other-service"))

skimr::skim(base_adult)

questionr::freq.na(base_adult)






#ML

set.seed(55)
base_adult_split <- initial_split(base_adult %>% select(-id), prop = 0.50)
base_adult_train <- training(base_adult_split)
base_adult_test <- testing(base_adult_split)


base_adult_recipe <- recipe(resposta ~ ., base_adult_train) %>%
  step_center(all_numeric()) %>%
  step_scale(all_numeric()) %>%
  step_dummy(all_nominal(), -all_outcomes())

base_adult_resamples <- vfold_cv(base_adult_train, v = 5)



base_adult_model <- boost_tree(
  mtry = 0.8,
  trees = tune(),
  min_n = 5,
  tree_depth = 4,
  loss_reduction = 0,
  learn_rate = tune(),
  sample_size = 0.8
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", nthread = cores)


base_adult_model


base_adult_wf <- workflow() %>%
  add_model(base_adult_model) %>%
  add_recipe(base_adult_recipe)




base_adult_grid <- expand.grid(
  learn_rate = c(0.05, 0.1, 0.2, 0.3),
  trees = c(100, 250, 500, 1000, 1500, 2000)
)
base_adult_grid



base_adult_grid <- base_adult_wf %>%
  tune_grid(
    resamples = base_adult_resamples,
    grid = base_adult_grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )



autoplot(base_adult_grid)
base_adult_grid %>% show_best(metric = "roc_auc", n = 5)
base_adult_select_best_passo1 <- base_adult_grid %>% select_best(metric = "roc_auc")
base_adult_select_best_passo1




base_adult_model <- boost_tree(
  mtry = 0.8,
  trees = base_adult_select_best_passo1$trees,
  min_n = tune(),
  tree_depth = tune(),
  loss_reduction = 0,
  learn_rate = base_adult_select_best_passo1$learn_rate,
  sample_size = 0.8
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", nthread = cores)

#### Workflow
base_adult_wf <- workflow() %>%
  add_model(base_adult_model) %>%
  add_recipe(base_adult_recipe)

#### Grid
base_adult_grid <- expand.grid(
  tree_depth = c(3, 4, 6, 8, 10),
  min_n = c(5, 15, 30, 60, 90)
)

base_adult_grid <- base_adult_wf %>%
  tune_grid(
    resamples = base_adult_resamples,
    grid = base_adult_grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

#### Melhores hiperparâmetros
autoplot(base_adult_grid)
base_adult_grid %>% show_best(metric = "roc_auc", n = 5)
base_adult_select_best_passo2 <- base_adult_grid %>% select_best(metric = "roc_auc")
base_adult_select_best_passo2


base_adult_model <- boost_tree(
  mtry = 0.8,
  trees = base_adult_select_best_passo1$trees,
  min_n = base_adult_select_best_passo2$min_n,
  tree_depth = base_adult_select_best_passo2$tree_depth,
  loss_reduction = tune(),
  learn_rate = base_adult_select_best_passo1$learn_rate,
  sample_size = 0.8
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", nthread = cores)

#### Workflow
base_adult_wf <- workflow() %>%
  add_model(base_adult_model) %>%
  add_recipe(base_adult_recipe)

#### Grid
base_adult_grid <- expand.grid(
  loss_reduction = c(0, 0.05, 0.1, 0.15, 0.25, 0.35, 0.45, 0.5, 1, 2)
)

base_adult_grid <- base_adult_wf %>%
  tune_grid(
    resamples = base_adult_resamples,
    grid = base_adult_grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

#### Melhores hiperparâmetros
autoplot(base_adult_grid)
base_adult_grid %>% show_best(metric = "roc_auc", n = 5)
base_adult_select_best_passo3 <- base_adult_grid %>% select_best(metric = "roc_auc")
base_adult_select_best_passo3





base_adult_model <- boost_tree(
  mtry = tune(),
  trees = base_adult_select_best_passo1$trees,
  min_n = base_adult_select_best_passo2$min_n,
  tree_depth = base_adult_select_best_passo2$tree_depth,
  loss_reduction = base_adult_select_best_passo3$loss_reduction,
  learn_rate = base_adult_select_best_passo1$learn_rate,
  sample_size = tune()
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", nthread = cores)

#### Workflow
base_adult_wf <- workflow() %>%
  add_model(base_adult_model) %>%
  add_recipe(base_adult_recipe)

#### Grid
base_adult_grid <- expand.grid(
  sample_size = seq(0.5, 1.0, length.out = 10),
  mtry = seq(0.1, 1.0, length.out = 10)
)

base_adult_grid <- base_adult_wf %>%
  tune_grid(
    resamples = base_adult_resamples,
    grid = base_adult_grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

#### Melhores hiperparâmetros
autoplot(base_adult_grid)
base_adult_grid %>% show_best(metric = "roc_auc", n = 5)
base_adult_select_best_passo4 <- base_adult_grid %>% select_best(metric = "roc_auc")
base_adult_select_best_passo4


base_adult_model <- boost_tree(
  mtry = base_adult_select_best_passo4$mtry,
  trees = tune(),
  min_n = base_adult_select_best_passo2$min_n,
  tree_depth = base_adult_select_best_passo2$tree_depth,
  loss_reduction = base_adult_select_best_passo3$loss_reduction,
  learn_rate = tune(),
  sample_size = base_adult_select_best_passo4$sample_size
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", nthread = cores)

#### Workflow
base_adult_wf <- workflow() %>%
  add_model(base_adult_model) %>%
  add_recipe(base_adult_recipe)

#### Grid
base_adult_grid <- expand.grid(
  learn_rate = c(0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3),
  trees = c(100, 250, 500, 1000, 1500, 2000, 3000)
)

base_adult_grid <- base_adult_wf %>%
  tune_grid(
    resamples = base_adult_resamples,
    grid = base_adult_grid,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc)
  )

#### Melhores hiperparâmetros
autoplot(base_adult_grid)
base_adult_grid %>% show_best(metric = "roc_auc", n = 5)
base_adult_select_best_passo5 <- base_adult_grid %>% select_best(metric = "roc_auc")
base_adult_select_best_passo5





#desempenho final


base_adult_model <- boost_tree(
  mtry = 0.3,
  trees = 3000,
  min_n = 5,
  tree_depth = 4,
  loss_reduction = 0.05,
  learn_rate = 0.01,
  sample_size = 0.9444444
) %>%
  set_mode("classification") %>%
  set_engine("xgboost", nthread = cores)

#### Workflow
base_adult_wf <- workflow() %>%
  add_model(base_adult_model) %>%
  add_recipe(base_adult_recipe)

base_adult_last_fit <- base_adult_wf %>%
  last_fit(
    split = base_adult_split,
    control = control_grid(save_pred = TRUE, verbose = FALSE, allow_par = TRUE),
    metrics = metric_set(roc_auc, f_meas)
  )

#### Métricas
collect_metrics(base_adult_last_fit)

#### Variáveis Importantes
base_adult_last_fit %>%
  pluck(".workflow", 1) %>%
  pull_workflow_fit() %>%
  vip::vip(num_features = 20)

#### Curva ROC
base_adult_last_fit %>%
  collect_predictions() %>%
  roc_curve(resposta, ".pred_>50K") %>%
  autoplot()


adult_test_preds <- collect_predictions(base_adult_last_fit)


adult_test_preds %>%
  mutate(
    resposta_class = factor(if_else(`.pred_<=50K` > 0.6, "<=50K", ">50K"))
  ) %>%
  conf_mat(resposta, resposta_class)



adult_modelo_final <- base_adult_wf %>% fit(base_adult)




adult_val <- read_rds("data/adult_val.rds")




adult_val_sumbissao <- adult_val %>%
  mutate(
    more_than_50k = predict(adult_modelo_final, new_data = adult_val, type = "prob")$`.pred_>50K`
  ) %>%
  select(id, more_than_50k)


questionr::freq.na(adult_val_sumbissao)

write_csv(adult_val_sumbissao, "data/adult_val_sumbissao.csv")
