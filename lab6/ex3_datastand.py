import numpy as np
import matplotlib.pyplot as plt

np.random.seed(50)
X = np.random.randn(30).reshape(5,6)


def standardization(X):
    m, n = X.shape
    X_norm = np.copy(X)
    u = np.mean(X)
    var = np.std(X) ** 2

    for i in range(0, m):
        for j in range(0,n):
            z = (X[i][j]  - u) / var
            X_norm[i][j] = np.array(z)
    return X_norm

X_std = standardization(X)

print("-"*25)
print(f"Old matrix: \n{X}\n")
print("-"*25)
print(f"Z-score normalized mat :\n{X_std}")