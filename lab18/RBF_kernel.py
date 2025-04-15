#RBFkernel
import pandas as pd
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import svm

df = [
  { "x1": 6, "x2": 5, "Label": "Blue" },
  { "x1": 6, "x2": 9, "Label": "Blue" },
  { "x1": 8, "x2": 6, "Label": "Red" },
  { "x1": 8, "x2": 8, "Label": "Red" },
  { "x1": 8, "x2": 10, "Label": "Red" },
  { "x1": 9, "x2": 2, "Label": "Blue" },
  { "x1": 9, "x2": 5, "Label": "Red" },
  { "x1": 10, "x2": 10, "Label": "Red" },
  { "x1": 10, "x2": 13, "Label": "Blue" },
  { "x1": 11, "x2": 5, "Label": "Red" },
  { "x1": 11, "x2": 8, "Label": "Red" },
  { "x1": 12, "x2": 6, "Label": "Red" },
  { "x1": 12, "x2": 11, "Label": "Blue" },
  { "x1": 13, "x2": 4, "Label": "Blue" },
  { "x1": 14, "x2": 8, "Label": "Blue" }
]


df = pd.DataFrame(df)
print(df)
X = df.iloc[:,0:2]
y=df.iloc[:,-1]

y = LabelEncoder().fit_transform(y)

# Consider the following dataset. Implement the RBF kernel.
# Check if RBF kernel separates the data well and compare it with the Polynomial Kernel.

from sklearn.svm import SVC

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#setting model to RBF
model = SVC(kernel="rbf",random_state=42)
model = model.fit(X_train,y_train)
y_pred=model.predict(X_test)


#metrics
cnf_mat = confusion_matrix(y_test,y_pred)
print("\nConfusion matrix:\n",cnf_mat)

from Code_repo.eval_metrics_reg_class import clas_metrics
metrics = clas_metrics(y_true=y_test, y_pred=y_pred)
print("\n",metrics.metric_report())



import matplotlib.pyplot as plt

#Plotting
def plot_decision_boundary(model):
  fig, ax = plt.subplots(1,1,figsize=(6, 5))

  # Plot decision regions
  DecisionBoundaryDisplay.from_estimator(
      model,
      X,
      response_method="predict",
      plot_method="pcolormesh",
      alpha=0.3,
      ax=ax
  )

  DecisionBoundaryDisplay.from_estimator(
      model,
      X,
      response_method="decision_function",
      plot_method="contour",
      levels=[-1, 0, 1],
      colors=["k", "k", "k"],
      linestyles=["--", "-", "--"],
      ax=ax
  )

  # Plot SVs
  plt.scatter(model.support_vectors_[:, 0],
             model.support_vectors_[:, 1],
             s=150,
             facecolors="none",
             edgecolors="k")

  # Plot original data
  scatter = plt.scatter(X['x1'], X['x2'], c=y, cmap='coolwarm', edgecolors='k')
  plt.title("SVM plot")


plot_decision_boundary(model=model)
plt.show()