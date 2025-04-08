# from ex2_3_log_Reg_sigmoid_utils import gradient_descent, comp_gz
import pandas as pd
import numpy as np

df = pd.read_csv('breast_Cancer_data.csv')
df = pd.DataFrame(df)

print(df.shape)
print(df.columns)

print(df["diagnosis"].info())

#label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

encoded_labels = le.fit_transform(df["diagnosis"])
df["diagnosis"] = encoded_labels


y = df["diagnosis"]
features = ['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']

X = df[features]
X = np.c_[np.ones([X.shape[0], 1]), X]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=55)
# grad_logL_history, theta_history = gradient_descent(X_train, y_train, iteration= 10000, alpha=0.00001)
# print(theta_history[-1])

logisR = LogisticRegression
clf = logisR(max_iter=10000, random_state=42)
clf.fit(X_train,y_train)
y_pred_test = clf.predict(X_test)

from sklearn.metrics import r2_score,accuracy_score

r2 = r2_score(y_test, y_pred_test)
print("R2 score:", r2)

acc_sco = accuracy_score(y_test,y_pred_test)
print("Accuracy score: ",acc_sco)


# from ex2_3_sigmoid_logis_reg import gradient_descent, comp_gz
#
#
# print(type(X_train))
# print(np.array(X_train))
# X_train = np.c_[np.ones([X_train.shape[0], 1]), X_train]
#
# grad_logL_history, theta_history  = gradient_descent(np.array(X_train), np.array(y_train), iteration=10000, alpha=0.0001)
# theta = theta_history[-1]
#
# #y predict
# y_pred_test = comp_gz(X_test, theta)
#
# r2 = r2_score(y_test, y_pred_test)
# print('R2 score, with func created from scratch : ',r2)
#
# from sklearn.metrics import accuracy_score
# acc = accuracy_score(y_test, y_pred_test)
# print("Accuracy, with func created from scratch:", acc)