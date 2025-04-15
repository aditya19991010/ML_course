import pandas as pd

df = pd.read_csv("/home/ibab/learning/ML_Lab/datasets/Tweets.csv", sep=",")
df = pd.DataFrame(df)
print(df)
print(df.columns)

print(df.head())