import pandas as pd
import numpy as np

data =pd.read_csv("../lab3/simulated_data_multiple_linear_regression_for_ML.csv")
df = pd.DataFrame(data)

print(df.info())
print(df.columns)

feature = ['age', 'BMI', 'BP', 'blood_sugar', 'Gender']
target = ['disease_score_fluct']

X = df[feature]
y = df[target]

folds = 10
m,n = df.shape

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


import math
split_index = np.linspace(0, m, 10 , dtype=int)
print(split_index)
r2_scores = []
for i in range(0, len(split_index) -1 ):
    #split into k-folds
    test_data = df.iloc[split_index[i]:split_index[i+1] ]
    train_data = df.drop(df.index[split_index[i]:split_index[i + 1]])
    #split train test data
    X_train = train_data[feature]
    y_train = train_data[target]
    X_test = test_data[feature]
    y_test = test_data[target]

#Apply linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_test = lin_reg.predict(X_test)

    r2 = r2_score(y_test, y_pred_test)
    r2_scores.append(r2)
    mean_r2 = np.mean(r2_scores)
    st_dev_r2 = np.std(r2_scores)

print("Mean coded: ", mean_r2)
print("Std dev coded: ", st_dev_r2)


#cross verify with sklearn

from sklearn.model_selection import cross_val_score, KFold

#Apply linear regression
lin_reg = LinearRegression()
cv_score = cross_val_score(lin_reg, X, y, cv=10)
print(np.mean(cv_score))

lin_reg.fit(X_train, y_train)
y_pred_test = lin_reg.predict(X_test)

r2 = r2_score(y_test, y_pred_test)
r2_scores.append(r2)
mean_r2 = np.mean(r2_scores)
st_dev_r2 = np.std(r2_scores)

print("Mean sklearn:", mean_r2)
print("Std dev sklearn:", st_dev_r2)

def mean_norm(X_train, X_test):
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)

    X_train = (X_train - X_train_mean) / X_train_std
    X_train_scaled = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term

    X_test = (X_test - X_train_mean) / X_train_std
    X_test_scaled = np.c_[np.ones(X_test.shape[0]), X_test]  # Add intercept term
    return X_train_scaled, X_test_scaled
