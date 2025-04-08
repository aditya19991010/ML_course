# Data normalization - scale the values between 0 and 1. Implement code from scratch.
import numpy as np

np.random.seed(50)
X = np.random.randint(1,15, size=50).reshape(5,10)
# min_max_normalization
from ex_helper import min_max

X_min_max = min_max(X)

print("-"*25)
print(f"Old matrix: \n{X}\n")
print("-"*25)
print(f"min max normalized mat :\n{X_min_max}")