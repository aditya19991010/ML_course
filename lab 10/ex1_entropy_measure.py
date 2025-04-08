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
def entropy_feature(df, feature, target):
    #Separating data
    y = df[target]
    X = df[feature]

    values = np.unique(X) #Create an array of unique categories
    feature_entropy = 0 #initial entropy
    for x in values: #iterations through all the values
        subset_y = y[X == x]  # Subset target data | to calculate entropy of each category in the feature
        feature_entropy += (len(subset_y) / len(y)) * entropy(subset_y)  # Calculate Weighted entropy
    return feature_entropy



def main():

    data = {
        "Patrons": ["circle", "rectangle", "circle", "rectangle", "rectangle",
                    "circle", "rectangle", "rectangle", "circle", "rectangle",
                    "rectangle", "circle", "rectangle"],
        "Target": [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    df = pd.DataFrame(data)  # Convert to DataFrame

    feat = "Patrons"
    target = "Target"
    #Entropy calculation
    #recurrence/re =1 , no-recurrence/ no_re=0
    E = entropy(df[target]) #Dataset entropy
    print(f"\nDataset Entropy : {E}")
    print("-"*30)

    # Calculate entropy for all the features

    entropy_values = float(entropy_feature(df,feat, target))

    print(f"Entropy for feature {feat}:",entropy_values)
    print("-"*30)
    # print(infor_gain(E, entropy_values))

if __name__=="__main__":
    main()
