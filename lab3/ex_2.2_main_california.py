from lab3.ex2_helper import gradient_descent, comp_hx, plot_param, train_test_split, mean_norm
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score


#Working with simulated data
def main():
    # Working with california data
    df = fetch_california_housing(as_frame=True).frame
    # print(df.columns)

    # Define features and target
    #removed features =
    features = ['MedInc',  'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
    # features = ['MedInc']
    target = 'MedHouseVal'

    #train -test split
    size=0.7
    seed=42
    X_train, X_test, y_train, y_test =train_test_split(df, size, seed, features, target)

    # Scaling -- Mean normalization
    # X_train, X_test = mean_norm(X_train, X_test)

    # Run gradient descent
    alpha= 1e-12
    iterations=200
    J_history, theta_history = gradient_descent(X_train, y_train, iterations, alpha)

    # Output results
    # print("Optimal theta:", theta_history)
    print("Final cost:", J_history)
    op_theta = theta_history[-1,:] #Optimal theta for all

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
