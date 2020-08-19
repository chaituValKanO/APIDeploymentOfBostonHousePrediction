import os

import numpy as np
import pandas as pd

from sklearn import datasets

from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

import pickle

from features import cat_features, num_features


os.chdir(os.getcwd())
print(f"The current working directory is {os.getcwd()}")

boston_data = datasets.load_boston()
print(f"The shape of the dataset is {boston_data.data.shape}")
print(f"The shape of the target attribute is {boston_data.target.shape}")

data = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)
print(data.head())

target = boston_data.target

data[cat_features] = data[cat_features].astype('category')
data[num_features] = data[num_features].astype('float64')

X_train, X_val, y_train, y_val = train_test_split(data, target, test_size=0.2, shuffle=True, random_state=123)
print(f"Shape of train data is {X_train.shape}")
print(f"shape of validation data is {X_val.shape}")
print(f"Shape of train target is {y_train.shape}")
print(f"Shape of validation target is {y_val.shape}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)])

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestRegressor())])

param_grid = {"classifier__n_estimators" : [10, 20, 30],
              "classifier__max_depth" : [2,4,6],
              "classifier__max_features" : [3, 5, 7]}

rf_grid = GridSearchCV(clf, param_grid= param_grid, cv=3)

print("Training in progress....")
rf_grid.fit(X_train,y_train)

print(f"The best parameters are {rf_grid.best_params_}")

yhat_train = rf_grid.best_estimator_.predict(X_train)
print(f"The MSE on train data is {mean_squared_error(y_train, yhat_train)}")

yhat_val = rf_grid.best_estimator_.predict(X_val)
print(f"The MSE on validation  data is {mean_squared_error(y_val, yhat_val)}")

print(f"Saving the model in {os.getcwd()}+'/saved_models")
with open(os.getcwd()+'/saved_models/model.pkl', 'wb') as model_file:
    pickle.dump(rf_grid.best_estimator_, model_file)