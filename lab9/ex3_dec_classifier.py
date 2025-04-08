import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing  import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, plot_tree

df = pd.read_csv("../lab7/sonar data.csv", header=None)
df = pd.DataFrame(df)

print(df.columns)

y = df.iloc[:,60]

y_df = pd.DataFrame(y)
X = df.iloc[:,:59]

#Ordinal encoding
enc = OrdinalEncoder()
y_enc =enc.fit_transform(y_df)


#Train-test split
X_train, X_test,y_train, y_test = train_test_split(X,y_enc,test_size=0.33,random_state=4)

#Model training
model = DecisionTreeClassifier(max_depth=5, random_state=77)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#Metric calculations
mse = mean_squared_error(y_test, y_pred)
r2score=r2_score(y_pred,y_test)
print("R2 score :",r2score)
print("Mean square error: ",mse)


# from sklearn.tree import plot_tree
# from matplotlib import pyplot as plt
#
# plt.figure(figsize=(20, 10))
# plot_tree(
#     model,
#     feature_names=['age', 'BMI', 'BP', 'blood_sugar', 'Gender'],
#     filled=True,
#     proportion=True,
#     rounded=True,
#     fontsize=10
# )
# plt.title("Decision Tree Structure")
# plt.show()