# Adaboost
## Source *https://towardsdatascience.com/wp-content/uploads/2021/04/1TcrnpBGsi5MsFog-fSdRMQ-1024x648.png*

import numpy as np4
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

X,y = make_classification(n_samples=100,n_clusters_per_class=1,n_features=5,n_classes=2, random_state=42)

#select the data
# X,y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

n = len(X_train)

w = np.ones(n)/n


##Initialization
M = 10

#Creating list to save models and lambda
models = []
lambda_b_list = []

#iterations
for m in range(M):
    # make decision tree for M times
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train,y_train, sample_weight=w)
    models.append(model)

    #pred values of training dataset
    y_pred = model.predict(X_train)

    #error calculation
    incorrect = y_pred != y_train
    e = np.sum(w* incorrect)/ np.sum(w)

    #Weight manipulation
    lambda_b = 0.5*np.log((1-e)/(e + 1e-10))
    lambda_b_list.append(lambda_b)
    w =w *np.exp(lambda_b* (2*incorrect -1))

    #final weight distribution
    w = w /np.sum(w)


#Prediction on test dataset
test_pred = np.zeros(len(X_test))

#enumeration
for i , (model, lambda_b) in enumerate(zip(models, lambda_b_list)):
    test_pred += lambda_b * model.predict(X_test)

#print(test_pred)
final_predictions = np.sign(test_pred)
#print(final_predictions)

#metrics
acc = np.mean(final_predictions==y_test)
print(f"AdaBoost Accuracy: {acc:.4f}")
