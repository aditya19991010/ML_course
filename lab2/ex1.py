import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
import random
import seaborn as sns


def load_data():
    [X,y] = fetch_california_housing(return_X_y=True)
    return (X,y)

california_housing = fetch_california_housing(as_frame=True)
print(california_housing.DESCR)

# print(california_housing.frame.head())
# print(california_housing.target.head())
print(california_housing.frame.info())

def hist_plot(dataset):
    dataset.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
    plt.subplots_adjust(hspace=0.7, wspace=0.4)
    plt.show()

dataset = california_housing
# hist_plot(dataset)

features_of_interest = ["AveRooms", "AveBedrms", "AveOccup", "Population"]
california_housing.frame[features_of_interest].describe()


def scatter_plt(dataset,x,y,size):
    sns.scatterplot(
        data=dataset.frame,
        x=x,
        y=y,
        size=size,
        hue=size,
        palette="viridis",
        alpha=0.5)

    plt.legend(title="MedHouseVal", bbox_to_anchor=(1.05, 0.95), loc="upper left")
    _ = plt.title("Median house value depending of\n their spatial location")
    plt.show()


x = "Longitude"
y = "Latitude"
size="MedHouseVal"

# scatter_plt(x,y,dataset,size)

rng = np.random.RandomState(0)
indices = rng.choice(np.arange(california_housing.frame.shape[0]), size=500, replace=False)
print(california_housing.frame.shape[0])

# Drop the unwanted columns
columns_drop = ["Longitude", "Latitude"]

subset = california_housing.frame.iloc[indices].drop(columns=columns_drop)

# Quantize the target and keep the midpoint for each interval
#subset the data based on quantile and saving only data which is in near median of house value.
# This is to generate pairplot for reducing the complexity
subset["MedHouseVal"] = pd.qcut(subset["MedHouseVal"], 6, retbins=False)
subset["MedHouseVal"] = subset["MedHouseVal"].apply(lambda x: x.mid)

_ = sns.pairplot(data=subset, hue="MedHouseVal", palette="viridis")

#ridgeCV
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate

alphas = np.logspace(-3, 1, num=30)
print("alphas:", alphas)
model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

cv_results = cross_validate(
    model,
    california_housing.data,
    california_housing.target,
    return_estimator=True,
    n_jobs=2,
)

print(cv_results)

# for each feature, test scores are generated
score = cv_results["test_score"]
print(f"R2 score: {score.mean():.3f} ± {score.std():.3f}")

# for each value in estimator of cv_results, then add this information in teh coef
coefs = pd.DataFrame(
    [est[-1].coef_ for est in cv_results["estimator"]],
    columns=california_housing.feature_names,
)

color = {"whiskers": "black", "medians": "black", "caps": "black"}
coefs.plot.box(vert=False, color=color)
plt.axvline(x=0, ymin=-1, ymax=1, color="black", linestyle="--")
_ = plt.title("Coefficients of Ridge models\n via cross-validation")
# plt.show()

def main():
    [X,y] = load_data()
    print(f"Dimention: Feature table = {X.shape}, Target = {y.shape}")


    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=786)
    print(f"Dimention: Train Feature table = {X_train.shape}, Test Feature table = {X_test.shape}")

    scalar  = StandardScaler()
    scalar = scalar.fit(X_train)

    X_train_scaled = scalar.transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    model  = LinearRegression()
    model.fit(X_train, y_train)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    plt.plot(model.coef_)
    # plt.show()
    plt.scatter(y_test,y_pred )
    # plt.show()
    r2 = r2_score(y_test,y_pred)
    print(f"r2 score: {r2}")

if __name__=="__main__":
    main()