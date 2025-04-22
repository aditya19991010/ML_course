'''Hierarchial clustering'''

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn
from ISLP import load_data
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage

#data preparation
X = load_data("NCI60")["data"]
y = pd.DataFrame(load_data("NCI60")["labels"])
y=y.values.reshape((len(y),))

#label encoding
print(y)
y = LabelEncoder().fit_transform(y)


def hierar_clust(X,y, method:str):
    model = AgglomerativeClustering(n_clusters=4, linkage="ward") #Number of clusters = len(set(y))-(int(len(set(y))/3))
    y_pred = model.fit_predict(X,y)
    return y_pred, model

method = "ward"
y_pred, model = hierar_clust(X,y,method)
print(set(y_pred))


#Scaling
X = StandardScaler().fit_transform(X)
linkage_mat = linkage(X, method="ward")

#Scatter plot
# plt.scatter(X[:,3],X[:,8], c=y_pred,cmap='rainbow')
# plt.show()


plt.figure(1,(6,6))
dendrogram(linkage_mat,labels=y_pred)
plt.title("Hierarchial clustering using NCI60 data")
plt.show()