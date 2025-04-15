import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from Code_repo.eval_metrics_reg_class import clas_metrics

df = [
    {"x1": 6, "x2": 5, "Label": "Blue"},
    {"x1": 6, "x2": 9, "Label": "Blue"},
    {"x1": 8, "x2": 6, "Label": "Red"},
    {"x1": 8, "x2": 8, "Label": "Red"},
    {"x1": 8, "x2": 10, "Label": "Red"},
    {"x1": 9, "x2": 2, "Label": "Blue"},
    {"x1": 9, "x2": 5, "Label": "Red"},
    {"x1": 10, "x2": 10, "Label": "Red"},
    {"x1": 10, "x2": 13, "Label": "Blue"},
    {"x1": 11, "x2": 5, "Label": "Red"},
    {"x1": 11, "x2": 8, "Label": "Red"},
    {"x1": 12, "x2": 6, "Label": "Red"},
    {"x1": 12, "x2": 11, "Label": "Blue"},
    {"x1": 13, "x2": 4, "Label": "Blue"},
    {"x1": 14, "x2": 8, "Label": "Blue"}
]
df = pd.DataFrame(df)
X = df[['x1', 'x2']]
y = LabelEncoder().fit_transform(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#gamma - Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’.
models = {
    "Linear": SVC(kernel="linear", random_state=42),
    "RBF": SVC(kernel="rbf", gamma='scale', random_state=42),
    "Polynomial": SVC(kernel="poly", degree=3, gamma='auto', random_state=42)
}

# Store results
results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n--- {name} Kernel ---")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    #print("Classification Report:\n", clas_metrics.metric_report(y_test, y_pred))
    results[name] = model


#Plotting
def plot_decision_boundary(model, title, ax):
    x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
    y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

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

    # Plot support vectors
    ax.scatter(model.support_vectors_[:, 0],
               model.support_vectors_[:, 1],
               s=150,
               facecolors="none",
               edgecolors="k")

    # Plot original data
    scatter = ax.scatter(X['x1'], X['x2'], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(title)


# ----------------------------
# Step 5: Plot All Kernels Side by Side
# ----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, model) in zip(axes, results.items()):
    plot_decision_boundary(model, f"{name} Kernel", ax)

plt.tight_layout()
plt.show()
