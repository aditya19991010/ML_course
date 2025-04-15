import pandas as pd
from matplotlib.pyplot import figure
from sklearn.datasets import load_iris
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
X = X[:,0:2]

df = pd.DataFrame(X, columns=iris.feature_names[0:2])
df['target'] = y

#subset class 0 and 1
df_two_classes = df[df['target'].isin([1, 2])]
y = df_two_classes['target']
X = df_two_classes.iloc[:,0:2]


from sklearn.svm import SVC
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.10, random_state=42)

model = SVC(kernel="rbf", random_state=11)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

report= classification_report(y_true=y_test,y_pred=y_pred)
print(report)

cnf = confusion_matrix(y_true=y_test,y_pred=y_pred)
print("\nConfusion matrix\n",cnf)

def plot_decision_boundary(model,title):
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s=50, edgecolors="black")
    plt.title("Iris dataset ; class 1 vs 2")
    DecisionBoundaryDisplay.from_estimator(model,X=X, response_method="predict",
                                           plot_method="pcolormesh", alpha=0.3 , ax=ax)

    DecisionBoundaryDisplay.from_estimator(
        model,X=X, response_method="decision_function",
        plot_method="contour",
        ax=ax)

title = "Iris dataset ; class 1 vs 2"
plot_decision_boundary(model=model,title=title)
plt.show()