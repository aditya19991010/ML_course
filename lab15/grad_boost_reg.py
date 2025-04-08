'''Implement Gradient Boost Regression and Classification using scikit-learn.
Use the Boston housing dataset from the ISLP package for the regression problem
and weekly dataset from the ISLP package and use Direction as the target variable
for the classification.'''
import numpy as np
from sklearn.metrics import r2_score, f1_score, roc_auc_score, mean_squared_error

'''Source - https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier'''

#Libraries
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from ISLP import load_data
from sklearn.model_selection import train_test_split
OJ = load_data('Boston')
print(OJ.columns)
print(OJ.head())
print(OJ.info())

y = OJ.iloc[:,-1]
X = OJ.iloc[:,:-1]


#Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=40)
#Training Data

alpha_val = np.linspace(0,1,num=20)


print("--"*15)
print("Model Training\n")
metrics = []
for i in alpha_val:
    model = GradientBoostingRegressor(learning_rate=i)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test,y_pred)
    r2sco = r2_score(y_test,y_pred)
    metrics.append((mse, r2sco ,i))
    print(f"MSE score : {mse}, \nR2score: {r2sco} at alpha: {i}")


print("--"*15)
print("\nFinal evaluation")
mse_, max_r2,_alpha = min(metrics)
print(f"min MSE score : {mse_:.4f},\nR2score: {max_r2:.4f} at alpha : {_alpha:.4f}")
