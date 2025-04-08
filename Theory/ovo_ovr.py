import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.svm import LinearSVC

#OneVsOneClassifier
X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=12)

model = OneVsOneClassifier(estimator=LogisticRegression(random_state=0))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics    import   accuracy_score
print(accuracy_score(y_test,y_pred))

#Onevs One
model = OneVsRestClassifier(estimator=LogisticRegression(random_state=0))
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import   accuracy_score
print(accuracy_score(y_test,y_pred))
