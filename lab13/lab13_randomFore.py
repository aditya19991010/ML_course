### Implement Random Forest algorithm for regression and classification using scikit-learn.
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

print("-"*30)
print("\nRandom forest Classifier : Iris dataset")
from sklearn.datasets import load_iris, load_diabetes
X,y = load_iris(return_X_y=True)

n,m= X.shape
print(f"dim:{n, m}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=3)
model.fit(X_train,y_train)
y_pred= model.predict(X_test)

acc_score = accuracy_score(y_test,y_pred)
print(f"\nAccuracy using Bagging : {acc_score}")
print("-"*30)
print("-"*30)

print("\nRandom forest Classifier : Iris dataset")
X,y = load_diabetes(return_X_y=True)

n,m= X.shape
print(f"dim:{n, m}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=3)
model.fit(X_train,y_train)
y_pred= model.predict(X_test)

acc_score = r2_score(y_test,y_pred)
print(f"\nr2 score using Bagging : {acc_score:.4f}")
print("-"*30)
