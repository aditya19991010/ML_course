import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import random
import seaborn as sns

#
df = pd.read_csv("simulated_data_multiple_linear_regression_for_ML.csv")
print(df.columns)

features = ['age', 'BMI', 'BP','Gender', 'blood_sugar']

X = df[features]
print(X.head())

# target = ['disease_score']
target = ['disease_score_fluct']
y = df[target]

def reg_plot(a,b,data):
    plt.figure(figsize=(9, 9))
    sns.regplot(x=a, y=b, color='#FF6600', data=data)
    plt.xlabel(f'{a}')
    plt.ylabel(f'{b}')

def main():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=786)
    print(f"Dimention: Train Feature table = {X_train.shape}, Test Feature table = {X_test.shape}")

    scalar = StandardScaler()
    scalar = scalar.fit(X_train)

    X_train_scaled = scalar.transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    model = LinearRegression()
    model.fit(X_train, y_train)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    plt.plot(model.coef_)
    # plt.show()
    plt.scatter(y_test, y_pred)
    plt.show()
    print("Model Coef: ",model.coef_)
    reg_plot('age', 'disease_score_fluct', df)
    r2 = r2_score(y_test, y_pred)
    print(r2)
    plt.show()
    print(f"r2 score: {r2}")
    # cdf = pd.DataFrame(model.coef_, X.columns, columns=['coef'])
    # print(cdf)

if __name__=='__main__':
    main()