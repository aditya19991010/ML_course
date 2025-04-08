import numpy as np


# calculate MSE
def mse(y):
    return np.var(y) * len(y) if len(y) > 0 else 0

# find the best feature and threshold to split the dataset.
def best_split(X, y, min_samples_split):
    best_mse = float("inf")
    best_split = None
    n, m = X.shape

    for feature in range(m):
        thresholds = np.unique(X[:, feature])

        for threshold in thresholds:
            left_idx = X[:, feature] <= threshold
            right_idx = X[:, feature] > threshold

            if sum(left_idx) < min_samples_split or sum(right_idx) < min_samples_split:
                continue

            #calculating total MSE
            left_mse = mse(y[left_idx])
            right_mse = mse(y[right_idx])
            total_mse = left_mse + right_mse

            #find the best split
            if total_mse < best_mse:
                best_mse = total_mse
                best_split = (feature, threshold, left_idx, right_idx)

    return best_split


#Recursively build tree using MSE and split data
def build_tree(X, y, min_samples_split=2, max_depth=None, depth=0):
    if len(y) < min_samples_split or (max_depth is not None and depth >= max_depth):
        return np.mean(y)  # Return leaf node value

    split = best_split(X, y, min_samples_split)
    if split is None:
        return np.mean(y)

    feature, threshold, left_idx, right_idx = split

    left_tree = build_tree(X[left_idx], y[left_idx], min_samples_split, max_depth, depth + 1)
    right_tree = build_tree(X[right_idx], y[right_idx], min_samples_split, max_depth, depth + 1)

    return (feature, threshold, left_tree, right_tree)


# Predict the value for a single sample
def predict_one(x, tree):
    if isinstance(tree, (int, float)):  # If leaf node
        return tree

    feature, threshold, left, right = tree
    if x[feature] <= threshold:
        return predict_one(x, left)
    else:
        return predict_one(x, right)

#Predict values for multiple samples
def predict(X, tree):
    return np.array([predict_one(x, tree) for x in X])


# Example usage
if __name__ == "__main__":
    # Sample dataset
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
    y = np.array([5, 7, 9, 10, 15, 14, 17, 19, 21, 22])

    # Train the decision tree regressor
    tree = build_tree(X, y, min_samples_split=2, max_depth=3)

    # Predict on new data
    X_test = np.array([[3], [6], [9]])
    y_pred = predict(X_test, tree)
    print(f"Predictions: {y_pred}")
