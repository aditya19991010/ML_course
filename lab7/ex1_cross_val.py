import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("sonar data.csv", header = None)
df = pd.DataFrame(df)

# print(df.info())
print(df.columns)
X=df.iloc[:,0:-1]
# print(X.info)
y = df.iloc[:,-1]
# print(y)

y_enc = np.where(y == 'R', 1, 0)

# for i in range(0,len(y)):
#     if y[i] == 'R':
#         y_enc[i] = 1
#     else:
#         y_enc[i] = 0

print(df.iloc[:,-1].value_counts())
# print(df.iloc[:,-1].value_counts()[1])


X_train, X_test, y_train, y_test = train_test_split(X,y_enc,test_size=0.33, random_state=50)


from sklearn.model_selection import cross_val_score, KFold

#Apply Logistic regression
log_reg = LogisticRegression()
cv_score = cross_val_score(log_reg, X, y, cv=10)
print(f'Cross validaiton score: {np.mean(cv_score)}')

# print(y.convert_dtypes("int64"))

# print(y_enc)
log_reg.fit(X_train,y_train)
y_pred = log_reg.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print(f"\n\nAccuracy score: {acc_score:.2f}")
