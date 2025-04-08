import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(22)
#Generate random mat
X0 = np.ones(30)

X1 = np.linspace(0,10, 30)
X2 = np.linspace(90,100, 30)
X3 = np.linspace(90,800, 30)
y = np.linspace(0,10, 30, dtype=int)
y.shape = 30,1

X = np.array([X0, X1, X2, X3]).T

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=42)
# X_test = X_test.c_[np.ones(X_test.shape[0],1),X_test]


# '''1. calculate hypothesis
# 2. Calculate log likelihood
# 3. update theta'''


# Stochastic Gradient Descent
    # Shuffle the training data
    # Iterate over each training example
    # Compute prediction
    # Compute error
    # Update theta
    # Print cost every 10 epochs
    # Compute predictions for the entire training set
    # Compute cost (MSE)

m,n = X_train.shape
alpha = 0.0000001
iterations = 1000


def stochast_grad(X_train, y_train, iteration=100, alpha = 0.000001):
    #select random theta
    np.random.seed(1)
    theta_list = []
    theta = np.random.randn(X_train.shape[1])

    for i in range(0,iterations):  #iterate
        for sample in range(0, m):
            xi = X_train[sample]
            yi = y_train[sample]
            hx = np.dot(xi, theta)
            grad = (hx - yi) * xi

            # update theta
            theta -= alpha*grad

            if i%1 == 0:
                y_pred = X_train@theta
                TSE = np.sum((y_pred - y_train)**2)
                theta_list.append(theta)
                # print(f'TSE for sample {sample} at {i}th iteration: {TSE}')
    return np.array(theta_list)[-1]
            


op_theta = stochast_grad(X_train, y_train, iteration=100, alpha = 0.0000001)
print(f'Optimal parameters: {op_theta}')

y_train = X_train@op_theta
# X_test = X_test.c_[np.ones(X.shape[0],1), X_test]

y_pred_test = X_test @ op_theta

from sklearn.metrics import r2_score
r2_Score = r2_score(y_pred_test,y_test)
print(f'r2 score: {r2_Score}')

# print(f"Dimention: y {y.shape}, X {X.shape} ,")
# #gz formula
# def comp_gz(X,theta):
#     gz = 1/ (1+ np.exp(-X@theta)) #sigmoid func
#     return gz
#
# def gradient_descent(X, y, iteration= 1000, alpha=0.01):
#     m,n = X.shape  # number of samples
#
#     l_theta_history = []
#     theta_history = []
#     grad_logL_history = []
#
#     theta = np.ones((n,1))
#     gz = comp_gz(X, theta)
#
#     print(f"Dimention: gz {gz.shape}, X {X.shape} ,")
#
#     #compute J
#     def comp_log_likelihood(gz):
#         print(f"Dimention: gz {gz.shape}, X {X.shape} ,Y {y.shape}")
#
#         log_theta = y.T@np.log(gz) + (1-y).T@np.log(1 - gz)
#         return log_theta
#
#     def gradient_logL(X, y, gz):
#         print(f"Dimention: gz {gz.shape}, X {X.shape} ,Y {y.shape}")
#
#         grad_logL = X.T @ (y - gz)
#         return grad_logL
#
#     #update theta
#     def update_theta(theta, y, gz, X):
#         theta_new = theta.copy()  # Create a copy of theta
#         theta_new += alpha * (X.T @ (gz - y  ))  # Update theta
#         return theta_new
#
#     for i in range(iteration):
#         l_theta = comp_log_likelihood(gz)
#         theta = update_theta(y,gz,X, theta)
#         grad_logL = gradient_logL(X, y, gz)
#
#         #save values
#         l_theta_history.append(l_theta)
#         theta_history.append(theta)
#         grad_logL_history.append(grad_logL)
#
#         if iteration%2==0:
#             print(f'Cost at {i}:',l_theta)
#     return grad_logL_history,theta_history
#
#
# grad_logL_history, theta_history = gradient_descent(X, y, iteration= 1000, alpha=0.01)
#
# print("Theta: ",theta_history[-1,:])
