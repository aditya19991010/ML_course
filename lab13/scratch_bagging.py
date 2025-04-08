## implement bagging from scratch
from sklearn.datasets import load_iris
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

## subset the data into several parts with replacement
# calculate average metric for  each model

X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
n,m = X_train.shape #get row and column numbers


#creating empty list for storing data
y_pred_list = []
acc_score_list = []

# select split size
_splits = 10
for i in range(_splits):
    size = int(n*0.8) # selecting 80% of the data
    indices = np.random.choice( n, size=size, replace=True) #choosing random samples with replacement
    ##modelling
    model = DecisionTreeClassifier(random_state=11)
    model.fit(X_train[indices,],y_train[indices,])
    y_pred_list.append(model.predict(X_test))

    #evaluation
y_pred_list = np.array(y_pred_list)


from scipy.stats import mode
# Perform majority voting (most frequent class label per test instance
y_final_pred = mode(y_pred_list, axis = 0)[0]

y_final_pred = y_final_pred.flatten()

acc_score = accuracy_score(y_test,y_final_pred)
print(f"\nAccuracy using Bagging : {acc_score}")
