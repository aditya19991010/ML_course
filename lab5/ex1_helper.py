import numpy as np

def comp_hx(X, theta):
    hx = X@theta
    return hx



# Gradient Descent Function
def gradient_descent(X, y, iterations=1000, alpha=0.01):
    n = X.shape[0]  # m: number of samples, n: number of features
    theta = np.zeros(n)
    # hx = np.zeros(m)  # Initialize y_predict
    J_history = []  # history of cost
    theta_history =[]

    hx = comp_hx(X, theta)

    def compute_cost(hx, y):
        # Compute cost J = (1/2) * sum((hx - y)^2)
        # J = 1/2((hx - y)**2 + ...
        J = (1/2) * (y-hx)
        return J

    def comp_update_theta(hx, X, y, theta, alpha=0.001):
        new_theta = theta.copy()
        new_theta -= alpha* (hx -y)*X
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


