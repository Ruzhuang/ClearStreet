## Data Exploration and Visualization (for details, see data.ipynb)

I read the data and split the data, used data after 2023-01-01 as the test set, which will be kept in a "vault" until the end to predict performance. 

After cleaning the data and gather the data for Y1 and Y2 according to instruction.
Observations:
1. Some variables have missing values continuously in a time interval, which can hinder performance and are difficult to impute accurately.
2. Removing rows with more than 50% or missing values did not solve this problem
3. Some variables are very stable as also suggested in the earlier histogram
4. There is a pattern where missing data are especially prevalent during openning and closing hours, which suggest time dependence of the model


## Model Chosen: Trees - XGBoost
I start with this approach because

1. Trees has a inherent way of dealing with missing data
2. I can consider time as one of the variable and decide whether making the model a time dependent is important in performance.
3. Tree model output feature importance according to which I can implement feature selection.

## Developping environment
I encounter a problem where the RAM of my computer is not sufficient for maneuver of the training dataset. It is a common solution to train data in batches; however, that will undermine the robustness of XGBoost model.

I decided to use cloud computing and chose Amazon sagemaker studio where I can deploy a large RAM instance with already available machine learning package installed.

## Hyperparameter tuning and feature selection (See train.ipynb)

Inspired by a popular github repo, I took the approach of iterative hyperparameter tuning and feature selection. 

For the first iteration: first tune hyper parameters using the entire dataset (of course excluding the test set), obtain the best model; then select the most important 10 features; for the next iteration, tune hyper parameters using the dataset of remaining features, and obtain the best model.

I experimented on smaller sections of the dataset, and discovered that SHAP feature importance is superior to feature importance obtained from the model because it produced a more stable ranking of importance between iterations. So I proceed to use SHAP as my criteria of feature selection. 

For hyperparameter tuning technique, since the hyperparameters space of xgboost is pretty large, I chose the bayesian approach where each trial of hyperparameter tuning is dependent on the result of the previous trial, which is more efficient than brute force grid search.

## Results

### Feature importance for Y1 (importance_1.csv)

importance_1.csv contains the features and their shap importance for model of Y1 ranked from most important to least. "time" is the most important feature, implying that the model Y1 = f(X1, ..., X375) is highly dependent on time of day. So the final model will take time as a independent variable.

Here are the top 10 most important features and are chosen to build a model of Y1 based on.
|   | feature | importance                 |
|---|---------|----------------------------|
| 1 | time    | 0.03949241169207236        |
| 2 | X230  | 0.03797985158154886        |
| 3 | X232  | 0.0322190426200736...      |
| 4 | X121  | 0.0161205641461319...      |
| 5 | X53   | 0.0137375092929162...      |
| 6 | X52   | 0.0132722270829545...      |
| 7 | X49   | 0.0125039382910733...      |
| 8 | X51   | 0.011931316442625208      |
| 9 | X316  | 0.0099653088843039...      |
| 10| X372  | 0.009741910847787748      |


### Feature importance for Y1 (importance_2.csv)
Similarly, importance_2.csv contains the features and their shap importance ranked from most important to least, for model of Y2.
Here are the top 10 most important features and are chosen to build a model of Y2 based on.
| Rank | Feature | Importance                |
|------|---------|---------------------------|
| 1    | X253    | 0.01387275115954655       |
| 2    | X313    | 0.0050255154615261...     |
| 3    | X250    | 0.00388089886975602       |
| 4    | X203    | 0.0022233188610549...     |
| 5    | X316    | 0.0005195188876147...     |
| 6    | X222    | 0.0002753967191006...     |
| 7    | X373    | 8.100518278344504e...     |
| 8    | time    | 0.0                       |
| 9    | X324    | 0.0                       |
| 10   | X329    | 0.0                       |



### model (model_Y1.json, model_Y2.json)
The trained models are saved using xgb.save_model in model_Y1.json, model_Y2.json respectively

### Test (see test.ipynb)
The trained models are tested on the test set left in the "vault".

Unfortunately, the result of the model train is not predictive in the sense that the sqrt mean square error between model prediction and actual y values are not significantly smaller than the rmse between an array of 0 and the y values.
