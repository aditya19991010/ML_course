import numpy as np
from ISLP import load_data

#data preparation
df = load_data("Weekly")
y = df["Direction"]
X = df.drop("Direction", axis="columns")

X = X.iloc[:,0:2]

# Set the number of clusters

 #elements per cluster


def generate_cluster(data,max_clusters = 4,item_cluster = 10):
    np.random.seed(22)

    clusters = {}

    for i in range(0,max_clusters):
        rand_inx = np.random.choice(len(data), size=item_cluster, replace=False)
        clusters[i+1] = np.array(rand_inx)

    return clusters

clusters = generate_cluster(X)

#calculate centroid
def calc_centroid(clusters):
    centroids = [np.mean(clusters[x]) for x in range(1, len(clusters)+1)]
    return centroids

centroid = calc_centroid(clusters)
print(centroid)

# randomly select K data points as cluster center
# Euclidean distance between each data point and each cluster center
# Assign each data point to that cluster whose center is nearest to that data point.
# recursion until no change observed in the clusters or iterations


