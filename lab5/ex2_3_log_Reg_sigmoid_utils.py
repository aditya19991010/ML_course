# Implement sigmoid function in python and visualize it

# equation
#1.  hx = (y - hx)xi
#2. y_pred = threshold 0-1
#3. r2 score

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from ex2_3_sigmoid_logis_reg import comp_gz, gradient_descent
#Generate random mat

np.random.seed(55)
X = np.random.rand(500,3)

# X1 = np.random.randint(0,10, 30)
# X2 = np.random.randint(90,100, 30)
# X3 = np.random.randint(90,800, 30)
X = np.c_[np.ones([X.shape[0], 1]), X]

y = np.random.randint(0,2, 500, dtype=int)
y = y.reshape(500, 1)


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=22)

theta = np.ones([1, X.shape[1]])

print(f"Dimention: y {y.shape}, X {X.shape}, theta {theta.shape} )")


grad_logL_history, theta_history = gradient_descent(X_train, y_train, \
                                                    iteration= 10000, alpha=0.0001)



print("Theta: ",theta_history[-1])
op_theta = theta_history[-1]

y_pred_test = comp_gz(X_test, op_theta)

#compute r2 score value
r2 = r2_score(y_test,y_pred_test)
print('R2 score, coded from scratch: ',r2)

# print(y_pred_test)
# print(y_test)

from sklearn.metrics import accuracy_score
y_pred_labels = (y_pred_test >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_labels)
print("Accuracy, coded from scratch:", acc)


#plot
import seaborn as sns
import matplotlib.pyplot as plt

# sns.regplot(x= y_pred_test, y=y_test, label="Testing Data")
# plt.plot(X_test[:,1], y_pred_test)
plt.scatter(X_test[:,1], y_pred_test)
plt.ylim((0,1))
plt.xlabel(' Feature')
plt.ylabel('Predicted value (y_pred_test)')
plt.title(f'Sigmoid : Feature vs. Predicted; Data')
plt.axhline(0.5, color='red')
# plt.legend(loc="upper left")
plt.show()


#Testing With sklearn
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression
clf = lr(max_iter=10000, random_state=42)
clf.fit(X_train, y_train)

y_pred_test = clf.predict(X_test)
#compute r2 score value
r2 = r2_score(y_test,y_pred_test)
print('R2 score, With sklearn : ',r2)

# print(y_pred_test)
# print(y_test)

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred_test)
print("Accuracy, With sklearn :", acc)


