import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sonar data.csv", header = None)
df = pd.DataFrame(df)

# print(df.info())
print(df.columns)
X=df.iloc[:,0:-1]
# print(X.info)
y = df.iloc[:,-1]
# print(y)

y_enc = np.where(y == 'R', 1, 0)


#data normalization
from lab6.ex_helper import min_max

def min_max(X):
    m, n = X.shape
    X_norm = X.copy()
    for j in range(0,m):
        for i in range(0, n):
            min_x = X.iloc[:,i].min()
            max_x = X.iloc[:,i].max()
            newX = (X.iloc[:,i] - min_x) / (max_x - min_x)
            X_norm.iloc[:,i] = newX
    return X_norm

X_norm = min_max(X)
# print(X)
# print(X_norm)

print("-"*30)
print("Using self made min_max normalization\n")

X_train, X_test, y_train, y_test = train_test_split(X_norm,y_enc,test_size=0.33, random_state=50)

# from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler

#Apply Logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {acc_score:.4f}")

print("-"*30)
print("Using scikit learn Standard scaler\n")

X_train, X_test, y_train, y_test = train_test_split(X,y_enc,test_size=0.33, random_state=50)

scalar = StandardScaler()
scalar.fit(X_train)
scalar.fit(X_test)

X_train_scaled = scalar.transform(X_train)
X_test_scaled = scalar.transform(X_test)


#Apply Logistic regression
log_reg = LogisticRegression()

log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)

acc_score = accuracy_score(y_test, y_pred)
print(f"Accuracy score (Z-score norm): {acc_score:.4f}")
