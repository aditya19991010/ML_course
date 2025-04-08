import pandas as pd
from sklearn.preprocessing  import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("/home/ibab/learning/ML_Lab/lab3/simulated_data_multiple_linear_regression_for_ML.csv")
df = pd.DataFrame(df)
# print(df.head(10))

# def partition(df,feature)

def partition_data(df, feat, t):
    df_1 = df[df[feat] < t]
    df_2 = df[df[feat] > t]
    return df_1 , df_2

print('-'*30)
feat = "BP"
t = 80
print(f"Create two dataset; feature: {feat} , threshold: {t}")
data1, data2 = partition_data(df,feat="BP",t=t)
print(data1.head(2))
print(data2.head(2))

print('-'*30)
t = 78
print(f"Create two dataset; feature: {feat} , threshold: {t}")

data3, data4 = partition_data(df,feat="BP",t=t)
print(data1.head(2))
print(data2.head(2))


print('-'*30)
t = 82
print(f"Create two dataset; feature: {feat} , threshold: {t}")
data5, data6 = partition_data(df,feat="BP",t=t)
print(data1.head(2))
print(data2.head(2))

print('-'*30)


