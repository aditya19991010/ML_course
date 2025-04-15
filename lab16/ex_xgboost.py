#XGBoost classifier and regressor using scikit-learn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from ISLP import load_data
import numpy as np
import xgboost as xgb
from Code_repo.eval_metrics_reg_class import Regression_eval

df = load_data("Boston")
print(df.head())
print(df.info())
print(df.describe())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=22)



#matrix conversion
# dtrain_mat = xgb.DMatrix(X_train, y_train, enable_categorical=True)
# dtest_mat = xgb.DMatrix(X_test, y_test, enable_categorical=True)


#Raw modeling
print("Raw modelling ; with default params")
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test,y_pred)

#evaluation metrics
evaluate = Regression_eval(y_true=y_test,y_pred=y_pred)
metrics = evaluate.reg_metrics()
print(metrics)


print("\nHyperparameter tuning; Using Grid CV")

#Parameters
# n_estimators: Number of boosting rounds (trees)
# max_depth: Maximum depth of trees
# learning_rate: Step size shrinkage used to prevent overfitting
# subsample: Fraction of samples used for fitting trees
# colsample_bytree: Fraction of features used for fitting trees

learn_rate = np.linspace(start=0,stop=1,num=20)

params_grid = { "max_depth":[3,5,7],
           "learning_rate":learn_rate,
           "n_estimators":[10,50,100],
           "subsample":[0.5,0.70,1],
           "colsample_bytree":[0.6,0.8,1.0]}


# grid_model = GridSearchCV(estimator=XGBRegressor(random_state=42), param_grid=params_grid, cv=5, verbose=1)
#
# grid_model.fit(X_train,y_train)
# best_param_val =pd.DataFrame(grid_model.best_params_.items(), columns=["Param", "Value"])
# print(best_param_val)

##Saving params
# Fitting 5 folds for each of 1620 candidates, totalling 8100 fits
#               Param      Value
# 0  colsample_bytree   1.000000
# 1     learning_rate   0.473684
# 2         max_depth   3.000000
# 3      n_estimators  10.000000
# 4         subsample   0.500000

print("\nHyperparameter tuning; Done using Grid CV")



##Optimized params
model = xgb.XGBRegressor(random_state=42, learning_rate=0.473684, colsample_bytree=1, max_depth=3, n_estimators=10, subsample=0.5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate = Regression_eval(y_true=y_test,y_pred=y_pred)
metrics = evaluate.reg_metrics()
print(metrics)