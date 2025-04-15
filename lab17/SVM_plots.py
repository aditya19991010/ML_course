#Kernel Methods
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpmath import sqrtm
from sklearn.preprocessing import LabelEncoder

data = {
    'x1': [1, 1, 2, 3, 6, 9, 13, 18, 3, 6, 6, 9, 10, 11, 12, 16],
    'x2': [13, 18, 9, 6, 3, 2, 1, 1, 15, 6, 11, 5, 10, 5, 6, 3],
    'Label': ['Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue', 'Blue',
              'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red', 'Red']
}

df = pd.DataFrame(data)
X = df.iloc[:,0:2]
y = LabelEncoder().fit_transform(df['Label'])
print(X,y)

# Plotting settings
fig, ax = plt.subplots(figsize=(8, 12))

scatter = ax.scatter(X.iloc[:, 0], X.iloc[:, 1], s=100, c=y, label=y, edgecolors="k")
ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
ax.set_title("Samples in two-dimensional feature space")
_ = plt.show()


from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


def plot_training_data_with_decision_boundary( model, kernel,levels_arr):

    # Settings for plotting
    _, ax = plt.subplots(figsize=(8, 8))
    x_min, x_max, y_min, y_max =  0, 20, 0, 20
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    common_params = {"estimator": model, "X": X, "ax": ax}
    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.2,
    )

    DecisionBoundaryDisplay.from_estimator(
        **common_params,
        response_method="decision_function",
        plot_method="contour",
        levels=levels_arr,
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    ax.scatter(
        model.support_vectors_[:, 0],
        model.support_vectors_[:, 1],
        s=150,
        facecolors="none",
        edgecolors="k")

    # Plot samples by color and add legend
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Classes")

    ax.set_title(f"Decision boundaries of {kernel} kernel in SVC")


kernel="linear"
clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)
levels=[-1, 0, 1]

plot_training_data_with_decision_boundary(model=clf,kernel="linear", levels_arr=levels)
plt.show()
quit()
########################################333


def Transform(x1, x2):
    # Polynomial transformation to higher dimension
    return np.column_stack((x1**2, x2**2, np.sqrt(x1*x2)))

# Apply transformation
transformed = Transform(df['x1'], df['x2'])
df['x3'] = transformed[:, 2]

# Step 4: Plot transformed 3D points
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Color by label
colors = df['Label'].map({'Blue': 'blue', 'Red': 'red'})

ax.scatter(df['x1'], df['x2'], df['x3'], c=colors)

# Labels
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('x3')
ax.set_title('Transformed 3D Points')

# Optional: Add a separating plane
# Plane equation: z = c (because x3 = sqrt(x1*x2))
xx, yy = np.meshgrid(np.linspace(df['x1'].min()-1, df['x1'].max(), 10),
                     np.linspace(df['x2'].min()-1, df['x2'].max(), 10))
zz_surface =  np.sqrt(xx**2 * yy**2)
ax.plot_surface(xx, yy, zz_surface, color='lightgreen', alpha=0.3, rstride=1, cstride=1)

# Plot the decision boundary plane
boundary_z = 100
zz_boundary = np.full_like(xx, boundary_z)
ax.plot_surface(xx, yy, zz_boundary, color='orange', alpha=0.5)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Blue Class', markerfacecolor='blue', markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Red Class', markerfacecolor='red', markersize=10),
    Line2D([0], [0], color='orange', lw=4, label='Decision Boundary (Plane)')
]
ax.legend(handles=legend_elements)

plt.show()