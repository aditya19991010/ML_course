'''Implement Gradient Boost Regression and Classification using scikit-learn.
Use the Boston housing dataset from the ISLP package for the regression problem
and weekly dataset from the ISLP package and use Direction as the target variable
for the classification.'''
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

'''Source - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier'''

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from ISLP import load_data

df = load_data('Weekly')
print(df.info())
print(df.columns)
print(df.describe())


X = df.iloc[:,:-1]
y = df['Direction']

print("Label encoding")
y = LabelEncoder().fit_transform(y)

#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)


alpha_val = np.linspace(0,0.5,num=20)


print("--"*15)
print("Model Training\n")
metrics = []
for i in alpha_val:
    model = GradientBoostingClassifier(learning_rate=i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    f1_sc = f1_score(y_test,y_pred)
    ras = roc_auc_score(y_test,y_pred)
    acc_sc = accuracy_score(y_test,y_pred)
    metrics.append((f1_sc, ras ,acc_sc, i))
    print(f"f1 score : {f1_sc}, ROC AUC score: {ras} \naccuracy score: {acc_sc} at alpha: {i}")


metrics = pd.DataFrame(metrics)
metrics.columns = ["f1_sc", "ras" ,"acc_sc", "alpha"]
print(metrics)

print("--"*15)
print("\nFinal evaluation")
print(metrics.loc[max(metrics["f1_sc"]),:])

#print(f"f1 score : {f1_sco:.4f}, ROC AUC score {ras_} \naccuracy score : {acc_sc_:.4f} at alpha : {_alpha:.4f}")
