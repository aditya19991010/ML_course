# Implement entropy measure using Python. The function should accept a set of data points and their class labels and return the entropy value.


#Entropy calculation using breast_cancer dataset
# Split the data
# calculate the entropy of the dataset
# calculate entropy for each the category distribution based on target value


import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

#Calculating Entropy of the dataset
def entropy(y):
    counts = np.bincount(y) #count entropy for each class
    Pr = counts / len(y)
    E = np.sum([-1 * x * np.log2(x) for x in Pr if x > 0]) #sum all the entropy
    return E


#calculate entropy for each cateogry of the feature
#select the feature
#count the frequency of each feature based in y value
#subset the data
#calculate the entropy

#Entropy of the feature
def entropy_feature(X_feature, y):
    values = np.unique(X_feature) #Create an array of unique categories
    feature_entropy = 0 #initial entropy
    for x in values: #iterations through all the values
        subset_y = y[X_feature == x]  # Subset target data | to calculate entropy of each category in the feature
        feature_entropy += (len(subset_y) / len(y)) * entropy(subset_y)  # Calculate Weighted entropy
    return feature_entropy


def main():
    df = pd.read_csv("/home/ibab/learning/ML_Lab/lab 8/breast-cancer.csv", header=None)
    df = pd.DataFrame(df)
    print(df.columns)

    # Seperating data
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Label encoding
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)  # recurrence/re =1 , no-recurrence/ no_re=0
    X_enc = X.apply(LabelEncoder().fit_transform)  # Label encoding will be applicable to whole data

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.33, random_state=4)

    #Entropy calculation
    #recurrence/re =1 , no-recurrence/ no_re=0
    E = entropy(y_train) #Dataset entropy
    print(f"\nDataset Entropy : {E}")
    print("-"*30)

    # Calculate entropy for all the features
    entropy_values = {}
    for col in X_train.columns:
        entropy_values[col] = float(entropy_feature(X_train[col], y_train))

    ig = {}
    for col,E_val in entropy_values.items():
        ig[col]= (float(E - E_val))

    print("Entropy for each feature:",entropy_values)
    print("-"*30)
    print("IG for each feature:",ig)
    # print(infor_gain(E, entropy_values))

if __name__=="__main__":
    main()
