import numpy as np
from ISLP import load_data

#Approach 1
#data preparation
df = load_data("Weekly")
y = df["Direction"]
X = df.drop("Direction", axis="columns")

X = np.array(X.iloc[:,0:2])
# print(len(X.columns))
# Set the number of clusters

#elements per cluster


'''Pseudo code KMeans
1. Create clusters
    set number of clusters
    assign random samples in the clusters
2. calculate centroid of each clusters
3. calculate distance of samples from centroid of each cluster
    euclidiean dist =  sqrt(sum(feature - centroid)^2)
4. Assign labels of the closest cluster
'''

def generate_rand_cluster(data,max_clusters = 4,item_per_cluster = 10):
    np.random.seed(22)
    clusters = {}
    for i in range(0,max_clusters):
        rand_inx = np.random.choice(len(data), size=item_per_cluster, replace=False)
        clusters[i+1] = np.array(rand_inx)
    return clusters

clusters = generate_rand_cluster(X)
# print(X.iloc[clusters[1],1])
# print(X.iloc[clusters[1],0])

print(clusters)

#calculate centroid
def calc_centroid(clusters,X):
    clust_centroid = {}
    for v in range(1,len(clusters)+1):
        index_clus = clusters[v]
        centroid = np.zeros(X.shape[1])
        for i in range(len(centroid)):
            centroid[i] = np.mean(X[index_clus,i])
            clust_centroid[v] = np.array(centroid)
    return clust_centroid


cluster_centr = calc_centroid(clusters,X)

print(cluster_centr)
# randomly select K data points as cluster center
# rand_clust = np.random.choice(centroid,2)

# Euclidean distance between each data point, and each cluster center
def calc_eucl_dist(cluster_centroid:dict, data):
    '''
    Parameters
    :param cluster_centroid: dictionary of cluster and centroid array
    :param data: feature information
    :return: Dictionary of cluster and samples assigned in the cluster
    '''

    cluster_assignments = {}
    m,n = data.shape
    for sample in range(m):
        min_dist = float("inf")
        assigned_cluster = None
        for clus_num in range(1,len(cluster_centroid)+1):
            dist_cluster = np.sqrt(np.sum((data[sample,:] - cluster_centroid[clus_num])**2))
            if dist_cluster < min_dist:
                min_dist = dist_cluster
                assigned_cluster =clus_num
        cluster_assignments[sample] = assigned_cluster
    return  cluster_assignments

cluster_assignments = calc_eucl_dist(cluster_centr, X)

def assign_cluster(cluster_assignments):
    #Create a dictionary containing all the clusters as key
    # add samples in each cluster
    clusters_data = {i : [] for i in range(1,len(set(cluster_assignments.values()))+1)}
    for sample, clus in cluster_assignments.items():
        clusters_data[clus].append(sample)
    return clusters_data

clusters_data=assign_cluster(cluster_assignments)
print(clusters_data[1])

new_data = X[clusters_data[1],:]

iterations=10
clusters = generate_rand_cluster(X)

for l in range(iterations):
    cluster_centr = calc_centroid(clusters, X)
    cluster_assignments=calc_eucl_dist(cluster_centr, X)

# def generate_rand_cluster(cluster_centr,X, max_clusters=4, item_cluster=10):
#
#     # iterate through each sample
#
#     # fetch details of column of sample and subtract with centroid of the cluster
#     # calculate euclidean distance
#     # save the distance of each sample
#     # find the nearest cluster ,  sample having minimum distance
#     # assign the index of sample in that cluster
#
#     dist_cluster = {}
#     for cluster_num,centroid_val in cluster_centr.items():
#         for sample in range(len(X)):
#             dist_cluster[cluster_num][sample] = np.sqrt(())

# Assign each data point to that cluster whose center is nearest to that data point.
# recursion until no change observed in the clusters or iterations


