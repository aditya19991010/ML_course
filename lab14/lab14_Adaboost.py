#Implement Adaboost classifier using scikit-learn. Use the Iris dataset.

from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#model
print("-"*30)
print("\nAdaboost Classifier : Iris dataset")
X,y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), random_state=42)

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

acc_score = accuracy_score(y_test,y_pred)
print("Accuracy score using Adaboost Classifier: ",acc_score)