from lab3.ex2_helper import gradient_descent, comp_hx, plot_param, train_test_split, mean_norm
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
import pandas as pd

#Working with simulated data
def main():
    # Read data
    df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
    df = pd.DataFrame(df)
    # Define features and target
    features = ['age', 'BMI', 'BP', 'Gender', 'blood_sugar']
    target = 'disease_score_fluct'

    #train -test split
    size=0.7
    seed=42
    X_train, X_test, y_train, y_test =train_test_split(df, size, seed, features, target)

    # Scaling -- Mean normalization
    # X_train, X_test = mean_norm(X_train, X_test)

    # Run gradient descent
    alpha= 1e-10
    iterations=4000
    J_history, theta_history = gradient_descent(X_train, y_train, iterations, alpha)

    # Output results
    # print("Optimal theta:", theta_history)
    # print("Final cost:", J_history)
    op_theta = theta_history[-1,:] #Optimal theta for all
    print(op_theta)
    # plot J vs param
    fig = plot_param(features, theta_history, J_history, alpha, iterations)
    fig

    #compute y_pred
    #Calculate hx with optimal theta values == predict y
    y_predict = comp_hx(X_test,op_theta)

    print("Test r2",r2_score(y_test,y_predict))
    y_predict = comp_hx(X_train,op_theta)
    print("Train r2 score",r2_score(y_train,y_predict))

if __name__ == "__main__":
    main()
