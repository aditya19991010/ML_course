import numpy as np
import pandas as pd
from statsmodels.stats.rates import nonequivalence_poisson_2indep


#Calculating mean squared error - MSE
def mse(y):
    return np.var(y)*len(y) #variance * N = mse

#BEST SPLIT
def best_split(X,y,min_sample_split):
    # initalization
    best_mse = float("inf")
    best_split=None

    #Feature x Samples
    n,m = X.shape


    for feat_idx in range(m): #iterating through each feature
        feature = np.unique(X[:,feat_idx]) #get data of each feature for splitting data by threshold level
        for threshold in feature:

            #Create left and right index for separating data
            left_idx = X[:,feat_idx] <= threshold
            right_idx = X[:,feat_idx] > threshold

            #Stopping condition ; if each lead contains only 1 data point
            if sum(left_idx) < min_sample_split or sum(right_idx) < min_sample_split:
                continue

            #calculate mse for each node
            left_mse = mse(y[left_idx])
            right_mse = mse(y[right_idx])
            total_mse = left_mse + right_mse

            #change best mse ; it should be minimum
            if total_mse < best_mse:
                best_mse = total_mse
                best_split = (feat_idx, threshold,left_idx,right_idx) #set best split info in a tuple

    return best_split


#BUILDTREE
def build_tree(X,y,min_sample_split=2, max_depth=None, depth=0):

    #Stopping condition is y contains only 1 sample or depth >= max_depth
    if len(y) < min_sample_split or (max_depth is not None and depth >= max_depth):
        return np.mean(y)

    split  = best_split(X,y,min_sample_split)

    if split is None:
        return np.mean(y)

    feat_idx, threshold,left_idx,right_idx = split

    #recursively create trees
    left_tree = build_tree(X[left_idx,:],y[left_idx],min_sample_split, max_depth, depth+1)
    right_tree = build_tree(X[right_idx,:],y[right_idx], min_sample_split, max_depth, depth+1 )

    return (feat_idx, threshold, left_tree, right_tree)

#PRECISION_ONE

# for predicting value of 1 sample only
def precision_one(x,tree):
    if isinstance(tree, (float, int)):
        return tree

    feat_idx, threshold, left_tree, right_tree = tree
    if x[feat_idx] <= threshold:
        return precision_one(x, left_tree)
    else:
        return precision_one(x, right_tree)


#for generating an array of predicted value
def predict(X, tree):
    return np.array([precision_one(x, tree) for x in X])


###Dataset

# df = pd.read_csv("/home/ibab/learning/ML_Lab/datasets/Admission_Predict_Ver1.1.csv")
# df = pd.DataFrame(df)
# #
# X = df.iloc[:,:-1]
# y = df.iloc[:,-1]
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([5, 7, 9, 10, 15, 14, 17, 19, 21, 22])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=11)

# Train the decision tree regressor
tree = build_tree(X_train, y_train, min_sample_split=2, max_depth=3)

# Predict on new data
# X_test = np.array([[3], [6], [9]])
y_pred = predict(X_test, tree)

from sklearn.metrics import r2_score
print("R2 score: ",r2_score(y_test,y_pred))
print(f"Predictions: {y_pred}")
