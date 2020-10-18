
# Análise de clientes por perfil

``` r
library(tidyverse)
library(tidymodels)
library(ggplot2)
library(skimr)
library(stats)
library(doParallel)
```

``` r
cores = 5
base_adult <- read_rds("data/adult.rds")
```

``` r

glimpse(base_adult)
#> Rows: 32,561
#> Columns: 16
#> $ age            <dbl> 39, 50, 38, 53, 28, 37, 49, 52, 31, 42, 37, 30, 23, ...
#> $ workclass      <chr> "State-gov", "Self-emp-not-inc", "Private", "Private...
#> $ fnlwgt         <dbl> 77516, 83311, 215646, 234721, 338409, 284582, 160187...
#> $ education      <chr> "Bachelors", "Bachelors", "HS-grad", "11th", "Bachel...
#> $ education_num  <dbl> 13, 13, 9, 7, 13, 14, 5, 9, 14, 13, 10, 13, 13, 12, ...
#> $ marital_status <chr> "Never-married", "Married-civ-spouse", "Divorced", "...
#> $ occupation     <chr> "Adm-clerical", "Exec-managerial", "Handlers-cleaner...
#> $ relationship   <chr> "Not-in-family", "Husband", "Not-in-family", "Husban...
#> $ race           <chr> "White", "White", "White", "Black", "Black", "White"...
#> $ sex            <chr> "Male", "Male", "Male", "Male", "Female", "Female", ...
#> $ capital_gain   <dbl> 2174, 0, 0, 0, 0, 0, 0, 0, 14084, 5178, 0, 0, 0, 0, ...
#> $ capital_loss   <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
#> $ hours_per_week <dbl> 40, 13, 40, 40, 40, 40, 16, 45, 50, 40, 80, 40, 30, ...
#> $ native_country <chr> "United-States", "United-States", "United-States", "...
#> $ resposta       <chr> "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K"...
#> $ id             <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1...

base_adult$workclass <- as.factor(base_adult$workclass)
base_adult$education <- as.factor(base_adult$education)
base_adult$marital_status <- as.factor(base_adult$marital_status)
base_adult$occupation <- as.factor(base_adult$occupation)
base_adult$relationship <- as.factor(base_adult$relationship)
base_adult$race <- as.factor(base_adult$race)
base_adult$sex <- as.factor(base_adult$sex)
glimpse(base_adult)
#> Rows: 32,561
#> Columns: 16
#> $ age            <dbl> 39, 50, 38, 53, 28, 37, 49, 52, 31, 42, 37, 30, 23, ...
#> $ workclass      <fct> State-gov, Self-emp-not-inc, Private, Private, Priva...
#> $ fnlwgt         <dbl> 77516, 83311, 215646, 234721, 338409, 284582, 160187...
#> $ education      <fct> Bachelors, Bachelors, HS-grad, 11th, Bachelors, Mast...
#> $ education_num  <dbl> 13, 13, 9, 7, 13, 14, 5, 9, 14, 13, 10, 13, 13, 12, ...
#> $ marital_status <fct> Never-married, Married-civ-spouse, Divorced, Married...
#> $ occupation     <fct> Adm-clerical, Exec-managerial, Handlers-cleaners, Ha...
#> $ relationship   <fct> Not-in-family, Husband, Not-in-family, Husband, Wife...
#> $ race           <fct> White, White, White, Black, Black, White, Black, Whi...
#> $ sex            <fct> Male, Male, Male, Male, Female, Female, Female, Male...
#> $ capital_gain   <dbl> 2174, 0, 0, 0, 0, 0, 0, 0, 14084, 5178, 0, 0, 0, 0, ...
#> $ capital_loss   <dbl> 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0...
#> $ hours_per_week <dbl> 40, 13, 40, 40, 40, 40, 16, 45, 50, 40, 80, 40, 30, ...
#> $ native_country <chr> "United-States", "United-States", "United-States", "...
#> $ resposta       <chr> "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K"...
#> $ id             <int> 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 1...
```