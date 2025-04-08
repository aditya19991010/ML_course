import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


np.random.seed(88)
X = np.linspace(-11,11,100).reshape(-1,1)
y = np.sin(X).flatten() + np.random.normal(0, 0.15, X.shape[0])

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.33, random_state=3)


degrees = [1,3,20 ]
train_errors, val_errors = [], []

plt.figure(figsize=(12,4))
for i, d in enumerate(degrees):
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Compute Errors
    train_error = mean_squared_error(y_train, y_train_pred)
    val_error = mean_squared_error(y_val, y_val_pred)

    train_errors.append(train_error)
    val_errors.append(val_error)

    # Plot results
    plt.subplot(1, 3, i + 1)
    plt.scatter(X_train, y_train, label="Train Data", color="blue", alpha=0.5)
    plt.scatter(X_val, y_val, label="Validation Data", color="red", alpha=0.5)
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    plt.plot(X_plot, y_plot, label=f"Degree {d}", color="black")
    plt.title(f"Polynomial Degree {d}")
    plt.legend()

plt.show()

# Step 4: Plot training vs validation error
plt.figure(figsize=(6, 4))
plt.plot(degrees, train_errors, marker="o", label="Training Error")
plt.plot(degrees, val_errors, marker="o", label="Validation Error", linestyle="dashed")
plt.xlabel("Model Complexity (Polynomial Degree)")
plt.ylabel("Mean Squared Error")
plt.title("Train vs Validation Error")
plt.legend()
plt.show()


