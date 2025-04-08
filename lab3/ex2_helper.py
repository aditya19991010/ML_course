#Linear regression
# - Write a function to compute hypothesis
# - Write a function to compute the cost
# - Write a function to compute the derivative
# - Write update parameters logic in the main function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand

#mean normalization
def train_test_split(df, size, seed, features, target):
    #feature, target : add in the list

    # train test split
    np.random.seed(seed)
    random_index = np.random.permutation(df.index)

    split_index = int(len(df)*size)
    train_inx = random_index[:split_index]
    test_inx = random_index[split_index:]

    train_df = df.loc[train_inx]
    test_df = df.loc[test_inx]

    # print(X)
    X_train = train_df[features].values
    y_train = train_df[target].values

    X_test = test_df[features].values
    y_test = test_df[target].values
    return X_train, X_test, y_train, y_test

def mean_norm(X_train, X_test):
    X_train_mean = np.mean(X_train, axis=0)
    X_train_std = np.std(X_train, axis=0)

    X_train = (X_train - X_train_mean) / X_train_std
    X_train_scaled = np.c_[np.ones(X_train.shape[0]), X_train]  # Add intercept term

    X_test = (X_test - X_train_mean) / X_train_std
    X_test_scaled = np.c_[np.ones(X_test.shape[0]), X_test]  # Add intercept term
    return X_train_scaled, X_test_scaled

def comp_hx(X, theta):
    hx = 1 + np.dot(X, theta)
    return hx

def plot_param(features,theta_history,J_history,alpha,iterations):
    import matplotlib.pyplot as plt

    for i in range(len(features)):
        plt.scatter(theta_history[:,i],J_history, label=features[i], s=10)
        plt.plot(theta_history[:,i], J_history)
    plt.ylabel("Cost function")
    plt.xlabel("Parameter")
    plt.legend()
    plt.title(f"Plot J vs Param ; alpha :  {alpha}, iteration: {iterations}")
    return plt.show()

# Gradient Descent Function
def gradient_descent(X, y, iterations=1000, alpha=0.01):
    m, n = X.shape  # m: number of samples, n: number of features
    theta = np.zeros(n)
    hx = np.zeros(m)  # Initialize y_predict
    J_history = []  # history of cost
    theta_history =[]

    hx = comp_hx(X, theta)

    def compute_cost(hx, y):
        # Compute cost J = (1/2) * sum((hx - y)^2)
        # J = 1/2((hx - y)**2 + ...)
        TSE  = 0
        for p,q in zip(hx,y):
            TSE += (p - q)**2
        J = TSE/2
        return J

    def comp_update_theta(hx, X, y, theta, alpha=0.001):
        new_theta = theta.copy()
        for j in range(len(theta)):  # Loop over each parameter
            dj_dt = 0
            for i in range(m):  # Loop over each training sample
                dj_dt += (hx[i] - y[i]) * X[i][j]
            new_theta[j] -= alpha * dj_dt  # Update parameter theta[j]
        return new_theta


    for i in range(iterations):
        hx = comp_hx(X, theta)
        cost = compute_cost(hx, y)
        J_history.append(cost)
        theta = comp_update_theta(hx, X, y, theta, alpha)
        theta_history.append(theta)
        if i % 100 ==0:
            print(f"Iteration {i}, Cost: {cost}")
    return np.array(J_history), np.array(theta_history)

