import time

import pandas as pd
import numpy as np

import numpy as np

#l1 norm
def l1_norm(feat,lambda_l1):
    return lambda_l1*sum(abs(x) for x in feat)

#l2 norm
def l2_norm(feat,lambda_l2):
    return lambda_l2*np.sqrt(sum(x**2 for x in feat))


#with gradient descent
#gz formula
def comp_gz(X,theta):
    gz = 1 / ( 1 + np.exp(-X @ theta)) #sigmoid func
    return gz

def lasso_regression(X, y, iteration= 1000, alpha=0.01, l1_lambda=0.1):
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

    def gradient_logL(X, y, gz, theta, l1_lambda):
        # print(f"Dimention: gz {gz.shape}, X {X.shape} ,Y {y.shape}")
        grad_logL = X.T @ (y - gz) + (l1_lambda * theta)
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
        grad_logL = gradient_logL(X, y, gz, theta, l1_lambda)

        #save values
        l_theta_history.append(l_theta)
        theta_history.append(theta)
        grad_logL_history.append(grad_logL)

        if i%100==0:
            print(f'Cost at {i}:',l_theta)
    return grad_logL_history, np.array(theta_history)




df = pd.read_csv("../lab7/sonar data.csv", header= None)
df = pd.DataFrame(df)

# print(df.info())
print(df.columns)

#Understand the effect of lambda
# with and without regularization - check accuracy with multiple dataset
#check the coef larger the value lower the sensitivity
#change the lamba value and find value of time complex, accuracy and coef.

from sklearn.datasets import make_regression

X, y, w = make_regression(n_samples=100, n_features=3, n_informative=1, coef=True, random_state=1, noise=20)

import matplotlib.pyplot as plt

#plt data
# plt.scatter(x=X[:,1],y = y)
# plt.show()


# Obtain the true coefficients
print(f"The true coefficient of this regression problem are:\n{w}")


from sklearn.linear_model import LinearRegression, LogisticRegression , Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score


## SONAR Data
df = pd.read_csv("../lab7/sonar data.csv", header = None)
df = pd.DataFrame(df)

print(df.columns)
X = df.iloc[:,0:-1]
y = df.iloc[:,-1]
print(y)

y_enc = np.where(y == 'R', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(X,y_enc, test_size=0.33, random_state=10)

## Scratch

J_history , theta_history = lasso_regression(np.array(X_train),np.array(y_train),iteration=1000, alpha=0.01, l1_lambda=0.8)

optimal_theta = theta_history[-1,:]

y_pred_test = np.array(X_test) @ optimal_theta

y_pred_labels = (y_pred_test >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred_labels)

# print(y_pred_test)

acc_score = accuracy_score(y_test, y_pred_labels)
print(f"Accuracy score with scratch Lasso regression: {acc_score:.4f}")

# quit()


##scikit learn
model = LogisticRegression(penalty='l1', solver="liblinear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"Accuracy score with {model} penalty = l1 : {acc:.4f}")


print("-"*30, "\nUsing Lasso regression with multiple lambda values")

penalty = np.linspace(1,11,10, dtype=int)
print(penalty)
coef = []
accscore_list = []
time_taken = []

for i in penalty:
    start = time.time()
    clf = LogisticRegression(penalty= 'l1', solver="liblinear",C=i)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accscore_list.append(accuracy_score(y_test, y_pred))
    coef.append(clf.coef_)
    end = time.time()
    time_taken.append(end-start)

print("Accuracy score: ",accscore_list)
print("Time taken",time_taken)
# print(coef)

# plt.subplots(())
plt.plot(penalty, accscore_list)
plt.ylabel("accuracy score")
plt.title("Accuracy score as a function of regularization")
plt.xlabel("lambda value")
# plt.labels()
plt.show()

plt.plot(penalty, time_taken)
plt.ylabel("Time taken")
plt.title("Time as a function of regularization")
plt.xlabel("lambda value")
plt.show()


#plot coef
ax = plt.gca()

# print(coef)
coef_df = np.squeeze(coef, axis=1)
coef_df = pd.DataFrame(np.array(coef_df))


for i in range(0,60):
    ax.plot(1/penalty, coef_df[i])
    ax.axhline(0, linewidth=0.5, color='r')
    ax.set_xscale("log")
    # ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
    plt.xlabel("lambdas")
    plt.ylabel("parameter")
    plt.title("Lasso initial coefficients (60) as a function of the regularization")
    plt.axis("tight")
    plt.ylim(-5, 5)
plt.show()

# for i in range(0, len(lasso_lambda)):
#     plt.bar(np.array(coef[i]), lasso_lambda)
# plt.show()
