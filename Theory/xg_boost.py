from sklearn.datasets import make_classification

X,y = make_classification(n_samples=1000, n_features=10,n_informative=5, random_state=1)
print(X.shape, y.shape)

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

model = GradientBoostingClassifier()

#model evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1, error_score='raise')
print(n_scores)

print(f"CV accuracy score: {np.mean(n_scores)} +- {np.std(n_scores)}")

#fit the model on the whole dataset
model.fit(X,y)
row = [[2.56999479, -0.13019997, 3.16075093, -4.35936352, -1.61271951, -1.39352057, -2.48924933, -1.93094078, 3.26130366, 2.05692145]]
y_hat =model.predict(y)
print(y_hat[0])