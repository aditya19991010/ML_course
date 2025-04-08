import numpy as np
import pandas as pd

#calculating entropy
def entropy(y):
    classes, count = np.unique(y, return_counts=True)
    Pr = count/len(y)
    return np.sum([-p*np.log2(p) for p in Pr if p > 0])


#calculating information theory
def information_gain(X, feature, y, threshold):

    left_idx = X[:,feature] <= threshold
    right_idx = X[:,feature] > threshold

    # return condition
    if sum(left_idx) == 0 or sum(right_idx) == 0:
        return 0

    #calculate data entropy
    H_data = entropy(y)
    # calculate feature entropy
    H_feature= sum(left_idx)/len(y) * entropy(y[left_idx]) + sum(right_idx)/len(y)*entropy(y[right_idx])

    return H_data - H_feature



def best_split(X,y):

    #best IG and split
    best_ig = 0
    best_split=None #tuple of feature and threshold

    n,m = X.shape #shape

    #iterating through feature index
    for feat_idx in range(m):
        column= np.unique(X[:,feat_idx]) #select feature column
    #iterating though each value of the column to get the threshold for splitting
        for threshold in column:
            ig = information_gain(X, feat_idx, y, threshold) #calculate IG for each feature at several thresholds

        #replace values
            if ig > best_ig:
                best_ig = ig
                best_split = (feat_idx, threshold)

    return  best_split


def build_tree(X,y,max_depth=None, depth=0):
    #get classes and counts
    classes , counts = np.unique(y, return_counts=True)

    # stopping conditions
    if len(classes) == 0 or (max_depth is not None and depth >= max_depth):
        return classes[np.argmax(counts)]

    #get the split at feature and threshold
    split = best_split(X,y)
    if split is None:
        return classes[np.argmax(counts)]


    feature, threshold = split
    left_idx = X[:,feature] <= threshold
    right_idx = X[:,feature] > threshold

    #recursive tree construction
    left_subtree = build_tree(X[left_idx,:], y[left_idx], max_depth, depth=depth+1)
    right_subtree = build_tree(X[right_idx,:], y[right_idx], max_depth, depth=depth+1)

    return (feature, threshold, left_subtree, right_subtree)



def predict_one(x,tree):

    # stopping condition - if tree contains an integer value
    if isinstance(tree, (int, np.int64)):
        return tree

    # recursive iterate the test values and return the tree value
    feature, threshold, left, right = tree
    if x[feature] <= threshold:
        return predict_one(x, left)
    if x[feature] > threshold:
        return predict_one(x,right)

def predict(X, tree):
    return np.array([predict_one(x,tree) for x in X])


def main():
    np.random.seed(11)

    df = pd.read_csv("/home/ibab/learning/ML_Lab/datasets/sonar data.csv", header = None)
    df = pd.DataFrame(df)

    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    y_enc =  np.where(y == "R",1,0).flatten()


    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.33, random_state=11)

    #converting to numpy
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()


    tree = build_tree(X_train,y_train,max_depth=3)
    y_predict = predict(X_test,tree)

    print(y_predict)

    from sklearn.metrics import accuracy_score
    print(X_train.shape, X_test.shape,y_train.shape, y_test.shape,y_predict.shape)
    print(accuracy_score(y_true=y_test, y_pred=y_predict))

main()