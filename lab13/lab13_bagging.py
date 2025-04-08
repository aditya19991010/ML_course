from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

#Loading Iris dataset

print("Using Bagging and estimator = Decision tree Classification\n")
X,y = load_iris(return_X_y=True)
print("-"*30)
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#implenting model
model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10,n_jobs=5, random_state=42, verbose=3)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#metrics
acc= accuracy_score(y_test,y_pred)
print("-"*30)
print(f"\n\nAccuracy iris data: {acc}\n\n")


##################################
#Diabetes dataset
print("-"*30)
print("-"*30)

print("Using Bagging and estimator = Decision tree regressor")
#Loading Iris dataset
X,y = load_diabetes(return_X_y=True)

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#implenting model
model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=10,n_jobs=5, random_state=42)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#metrics
print("-"*30)

acc= r2_score(y_test,y_pred)
print(f"\n\nr2 score Diabetes data: {acc}\n\n")
print("-"*30)
print("-"*30)
