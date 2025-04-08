import numpy as np
#
# def min_max(X):
#     m, n = X.shape
#     X_norm = np.copy(X)
#     for i in range(0, m):
#         for j in range(0,n):
#             min_x = np.min(X[:,j])
#             max_x = np.max(X[:,j])
#             newX = (X[i][j] - min_x) / (max_x - min_x)
#             X_norm[i][j] = np.array(newX)
#     return X_norm

def min_max(X):
    m, n = X.shape
    X_norm = X.copy()
    for j in range(0,m):
        for i in range(0, n):
            min_x = X[:,i].min()
            max_x = X[:,i].max()
            newX = (X[:,i] - min_x) / (max_x - min_x)
            X_norm[:,i] = newX
    return X_norm