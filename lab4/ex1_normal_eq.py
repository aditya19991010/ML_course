import pandas as pd
import numpy as np
from lab3.ex2_helper import plot_param, train_test_split
from lab4.ex4_helper import comp_theta, comp_hx, comp_J
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
# Download latest version
# path = kagglehub.dataset_download("mohansacharya/graduate-admissions")
# print("Path to dataset files:", path)

#Closed form equation
# 1. hx = theta.T*X
# 2. J = 1/2 sum(theta*X - y).T * (theta*X - y)


## Simulated dataset
def simulate_data_train(path_data):

# Define features and target
    print(f"{'#'*50}\nProcessing {path_data}\n{'#'*50}")
    #pre processing
    df = pd.read_csv(path_data)
    df = pd.DataFrame(df)

    features = ['age', 'BMI', 'BP', 'Gender', 'blood_sugar']
    target = 'disease_score_fluct'

    # train - test split
    size = 0.7
    seed = 42
    X = np.array(df[features])
    # print("X:",X)
    y = np.array(df[target])
    # print("y:",y)

    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42 )

    X_train, X_test, y_train, y_test = train_test_split(df,size,seed,features, target)
    #  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Model fit
    #compute theta
    theta = comp_theta(X_train,y_train)
    print("Theta values: ",theta)

        #comp y_pred of train data
    y_pred_train = comp_hx(theta,X_train)

    #cost func matrix
    J = comp_J(y_pred_train,y_train)


    #Model predict
    r2_train = r2_score(y_pred_train,y_train)
    print('Train r2 score:',r2_train)

    y_pred_test = comp_hx(theta,X_test)
    r2_test = r2_score(y_test,y_pred_test)
    print('Test r2 score:', r2_test)

#plot
    # sns.regplot(x=y_train,y=y_pred_train, label="Training Data")
    sns.regplot(x=y_test,y=y_pred_test, label="Testing Data")

    plt.xlabel('Actual value (y_test)')
    plt.ylabel('Predicted value (y_pred_test)')
    plt.title(f'Actual vs. Predicted; Data : {path_data}')
    plt.legend(loc="upper left")
    plt.show()

#Admission data
def admission_data(pata_data):
    # def admission_data_train
    print()
    print(f"{'#'*50}\nProcessing {path_data}\n{'#'*50}")

    # pre processing
    df = pd.read_csv(path_data)
    df = pd.DataFrame(df)
    print(df.info())
    # print(df.columns)

    features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP',
        'LOR ', 'CGPA', 'Research']
    target = 'Chance of Admit '

    df = df.iloc[:,1:]
    def hist_df(df):
        df_num = df.select_dtypes(include=['float64', 'int64'])
        df_num.hist(figsize=(16, 20), bins=60, xlabelsize=8, ylabelsize=8)

    hist_df(df)
    plt.show()

    # train - test split
    size = 0.7
    seed = 42
    X = np.array(df[features])
    # print("X:",X)
    y = np.array(df[target])
    # print("y:",y)

    # X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30, random_state=42 )

    X_train, X_test, y_train, y_test = train_test_split(df, size, seed, features, target)
    #  print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # Model fit
    # compute theta
    theta = comp_theta(X_train, y_train)
    print("Theta values: ", theta)

    # comp y_pred of train data
    y_pred_train = comp_hx(theta, X_train)

    # cost func matrix
    J = comp_J(y_pred_train, y_train)

    # Model predict
    r2_train = r2_score(y_pred_train, y_train)
    print('Train r2 score:', r2_train)

    y_pred_test = comp_hx(theta, X_test)
    r2_test = r2_score(y_test, y_pred_test)
    print('Test r2 score:', r2_test)

    # plot
    # sns.regplot(x=y_train,y=y_pred_train, label="Training Data")
    sns.regplot(x=y_test, y=y_pred_test, label="Testing Data")

    plt.xlabel('Actual value (y_test)')
    plt.ylabel('Predicted value (y_pred_test)')
    plt.title(f'Actual vs. Predicted; Data : {path_data}')
    plt.legend(loc="upper left")
    plt.show()

if __name__=="__main__":

    path_data = 'Admission_Predict_Ver1.1.csv'
    admission_data(path_data)

    path_data = '../lab3/simulated_data_multiple_linear_regression_for_ML.csv'
    simulate_data_train(path_data)
