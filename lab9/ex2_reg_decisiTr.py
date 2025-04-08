import pandas as pd
from sklearn.preprocessing  import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/home/ibab/learning/ML_Lab/lab3/simulated_data_multiple_linear_regression_for_ML.csv")
df = pd.DataFrame(df)

print(df.columns)
target = ["disease_score_fluct"]

X = df[['age', 'BMI', 'BP', 'blood_sugar', 'Gender']]
y = df[target]

X_train, X_test,y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=7)
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2score=r2_score(y_pred,y_test)
print("R2 score :",r2score)
print("Mean square error: ",mse)


from sklearn.tree import plot_tree
from matplotlib import pyplot as plt

# Visualizing decision tree
plt.figure(figsize=(20, 10))
plot_tree(
    model,
    feature_names=['age', 'BMI', 'BP', 'blood_sugar', 'Gender'],
    filled=True,
    proportion=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Structure")
plt.show()

