import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.preprocessing import PolynomialFeatures

# Define the input vectors
x1 = np.array([3, 6])
x2 = np.array([10, 10])

print("Original vectors:")
print(f"x1 = {x1}")
print(f"x2 = {x2}")

# Method 1: Manual transformation and dot product
def Transform(x):
    # For a 2D vector [a, b], transform to [1, a, b, a^2, a*b, b^2]
    return np.array([x[0]**2, x[0]*x[1],x[1]*x[0], x[1]**2])

# Transform the vectors
print("Transformed vectors:")

x1_transformed = Transform(x1)
x2_transformed = Transform(x2)
print(x1_transformed)
print(x2_transformed)

# Compute dot product in the higher dimension
dot_product_manual = np.dot(x1_transformed, x2_transformed)
print(f"\nDot product of transformed matrix\n{dot_product_manual}")

#Method 2: Using kernel trick
#K(a, b) = a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2

def Kernel_transform(a,b):
    return a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2

x1 = np.array([3,6])
x2 = np.array([10,10])

k_val = Kernel_transform(x1,x2)
print("\nUsing Kernel trick:")
print(k_val)