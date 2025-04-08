#Compute hypothesis or y_predict
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import seaborn as sns
# from sklearn.model_selection import train_test_split

from lab3.ex2_helper import plot_param, train_test_split
import matplotlib.pyplot as plt

#Closed form equation
# 1. hx = X*theta
# 2. J = 1/2 sum(theta*X - y).T * (theta*X - y)

# calculate J
def comp_theta(X,y): #X,y=  X_train, y_train
    X = np.c_[np.ones((X.shape[0], 1)), X] # intercept
    Xt_X = np.dot(X.T,X)
    inv_Xt_X = np.linalg.inv(Xt_X)
    Xt_y = np.dot(X.T,y)
    theta = np.dot(inv_Xt_X, Xt_y)
    return theta

def comp_hx(theta, X_train):
    X = np.c_[np.ones((X_train.shape[0], 1)), X_train]
    hx = X@theta
    return hx

def comp_J(hx,y_train):
    J = 0.5* (hx-y_train).T * (hx-y_train)
    return J
