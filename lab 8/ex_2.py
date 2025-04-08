import pandas as pd
import numpy as np

dataset = pd.read_csv("breast-cancer.csv")
dataset= pd.DataFrame(dataset)

## Datasets split
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

##Encoding
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

###One hot encoding
enc = OneHotEncoder(sparse_output=False)
X_new = enc.fit_transform(X)

###label encoding
lab_enc = LabelEncoder()
y_new = lab_enc.fit_transform(y)

# #train-test split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
X_train, X_test, y_train, y_test =  train_test_split(X_new,y_new, random_state=11, test_size=0.33)


#Training the model with l1
model_l1 = Lasso(alpha=1)
model_l1.fit(X_train, y_train)
y_predict = model_l1.predict(X_test)

#Model performance metrics
from sklearn.metrics import r2_score
r2_sco = r2_score(y_test,y_predict)
print(f"r2 score L1 classifier: {r2_sco}")

#Training the model with l2
model_l2 = Ridge(alpha=1)
model_l2.fit(X_train, y_train)
y_predict = model_l2.predict(X_test)


#Model performance metrics
from sklearn.metrics import r2_score
r2_sco = r2_score(y_test,y_predict)
print(f"r2 score L2 classifier: {r2_sco}")
