# XGBoost Hyperparameter Tuning with pima indians Dataset

## Setting

We have gone through `pima-indians` dataset in the sessions. 
We achieved 74.02% accuracy using baseline XGBoost.
```from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# load data
dataset = loadtxt('./data/pima-indians-diabetes.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

# split data into train and test sets
seed = 42
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier(seed=42)
model.fit(X_train, y_train)

# evaluate predictions
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```
## 1. Write a function called `myXGBoost()`

* Accepts the following parameters:
    - X_train, y_train, X_test (Numpy arrays for training, testing; any format acceptable by sklearn will work)
    - paramgrid (list of parameters (including those of the classfier) for GridSearchCV)
    - KFold (the number of k-folds to be used in cross-validation) (Optional) (Default 3)
    - early_stopping_rounds (Int) (Optional) (Default 10)
    - seed (a number; a subsequent call to the function with the same seed will reproduce the same results)(optional) (Default 42)
    - - **kwargs (To set parameters to the base classifier)
  
* Should return
    - predictions for X_test
    - trained GridSerchCV object

## 2. Figure out optimum number of estimators for pima-indians dataset

* write a function called `n_estimator` which 
    * Takes in `X_train`, `X_test`, `y_train`, `param_grid`
    * Returns `y_pred_test`, `Trained GridSearch Object`




* You will use `myXGBoost()` function
* set `learning_rate` to 0.1
* You need to keep all other hyperparameters to their default values

## 3. Figure out optimum combination of `max_depth` and `min_child_weight`

* write a function called `depth_child` which 
    * Takes in Takes in `X_train`, `X_test`, `y_train`, `param_grid`
    * Returns `y_pred_test`, `Trained GridSearch Object`



* You will use `myXGBoost()` function
* You can set the optimised parameters from previous round(s)
* You need to keep all other hyperparameters to their default values

## 4. Figure out optimum combination of `gamma`, `lambda` and `alpha`

* write a function called `gal` which 
    * Takes in `X_train`, `X_test`, `y_train`, `param_grid`
    * Returns `y_pred_test`, `Trained GridSearch Object`

* You will use `myXGBoost()` function
* You can set the optimised parameters from previous round(s)
* You need to keep all other hyperparameters to their default values

## 5. Figure out optimum combination of `subsample`, `colsample_bytree`

* write a function called `suco` which 
    * Takes in `X_train`, `X_test`, `y_train`, `param_grid`
    * Returns `y_pred_test`, `Trained GridSearch Object`

* You will use `myXGBoost()` function
* You can set the optimised parameters from previous round(s)
* You need to keep all other hyperparameters to their default values

## 6. Finetune the Model

* Write a function called `finetune` which 
    * Takes in `X_train`, `X_test`, `y_train`, `param_grid`
    * Returns `y_pred_test`, `Trained GridSearch Object`

* You will use `myXGBoost()` function
* Based on the stage-wise optimization values further fine tune the model
* You need to keep all other hyperparameters to their default values

Hint: You can make a smaller grid these 8 parameters to further fine tune the model.

Caution: Too many values in param_grid will take up a lot of time. So wisely choose the param_grid values. You can consider 3 values for each hyperparameter. That will need 3^8 * 3(CV) models to be trained.