from unittest import TestCase
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from build import myXGBoost, n_estimator, depth_child, gal, suco, finetune

# load data
dataset = loadtxt('../data/pima-indians-diabetes.csv', delimiter=",")

# split data into X and y
X = dataset[:, 0:8]
Y = dataset[:, 8]

# split data into train and test sets
seed = 42
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
param_grid1 = {"n_estimators": [20, 30, 40, 50, 60, 70, 80, 90]}
param_grid2 = {"max_depth": [2, 3, 4, 5, 6, 7, 9, 11], "min_child_weight": [6, 7, 8, 9]}

class TestMyXGBoost(TestCase):

    def test_n_estimator(self):
        y_pred, gs = n_estimator(X_train, X_test, y_train, param_grid1)
        self.assertEqual(gs.best_params_['n_estimators'], 60)

    def test_depth_child(self):
        y_pred, gs = depth_child(X_train, X_test, y_train, param_grid2)
        self.assertEqual(gs.best_params_['max_depth'], 4)