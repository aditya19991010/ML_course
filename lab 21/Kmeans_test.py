import numpy as np
from ISLP import load_data


def compute_centroids(df, labels, k):
    '''
    Parameters
    :param df: add df in the numpy format
    :param labels: Encoded Label
    :param k: number of clusters
    :return: numpy array of centroid
    '''

    "Create an empty centroid list"
    centroids = []
    for i in range(k):
        cluster_points = df[labels == i]

        if len(cluster_points) == 0:
            centroid = np.zeros(df.shape[1])
        else:
            centroid = cluster_points.mean(axis=0)
        centroids.append(centroid)
    return np.array(centroids)


def assign_labels(df, centroids):
    '''
    :param df: Input dataframe
    :param centroids: array of centroid
    :return: closest cluster
    '''
    distances = np.linalg.norm(df[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)


def main():
    # Data loading and preparation
    df = load_data("Weekly")
    y = df["Direction"]
    X = df.drop("Direction", axis="columns")

    df = X.iloc[:, 0:2]
    df = np.array(df)

    # Initiate random labeling
    k = 2
    labels = np.random.choice(k, size=df.shape[0])
    print("Initial Random Cluster Labels:", labels)

    # Run K-Means

    iteration = 0
    while True:
        print(f"\nITERATION {iteration + 1}:")
        centroids = compute_centroids(df, labels, k)
        print("Centroids:\n", centroids)
        new_labels = assign_labels(df, centroids)
        print("New Labels:", new_labels)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        iteration += 1


if __name__ == "__main__":
    main()