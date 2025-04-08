import numpy as np

#gz formula
def comp_gz(X,theta):
    gz = 1 / ( 1 + np.exp(-X @ theta)) #sigmoid func
    return gz

def gradient_descent(X, y, iteration= 1000, alpha=0.01):
    m,n = X.shape  # number of samples
    theta = np.ones((n,1))
    y = y.reshape(-1, 1)

    l_theta_history = []
    theta_history = []
    grad_logL_history = []

    # theta = np.ones([1, n])
    # gz = comp_gz(X, theta)

    # print(f"Dimention: gz {gz.shape}, X {X.shape} ,")

    #compute J
    def comp_log_likelihood(gz):
        # print(f"Dimention: gz {gz.shape}, X {X.shape} ,Y {y.shape}")
        log_theta = np.sum(y * np.log(gz) + (1-y) * np.log(1 - gz))
        return log_theta

    def gradient_logL(X, y, gz, theta, l2_lambda):
        # print(f"Dimention: gz {gz.shape}, X {X.shape} ,Y {y.shape}")
        grad_logL = X.T @ (y - gz)
        return grad_logL

    #update theta
    def update_theta(theta, y, gz, X):
        theta_new = theta.copy()  # Create a copy of theta
        theta_new = theta + alpha * (X.T @ ( y - gz ))  # Update theta
        return theta_new

    for i in range(iteration):
        gz = comp_gz(X,theta)
        l_theta = comp_log_likelihood(gz)
        theta = update_theta(theta, y, gz, X)
        grad_logL = gradient_logL(X, y, gz, theta)

        #save values
        l_theta_history.append(l_theta)
        theta_history.append(theta)
        grad_logL_history.append(grad_logL)

        if i%100==0:
            print(f'Cost at {i}:',l_theta)
    return grad_logL_history, np.array(theta_history)


