#XGBoost classifier
from ISLP import load_data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.model_selection import train_test_split
df = load_data("Weekly")
print(df.head())
print(df.info())
print(df.describe())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(X,y)
#Label encoding
y = LabelEncoder().fit_transform(y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33, random_state=22)



#matrix conversion
# dtrain_mat = xgb.DMatrix(X_train, y_train, enable_categorical=True)
# dtest_mat = xgb.DMatrix(X_test, y_test, enable_categorical=True)


#Raw modeling
print("Raw modelling ; with default params")
model = xgb.XGBClassifier(random_state=42)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
acc_sco = accuracy_score(y_test,y_pred)

from Code_repo.eval_metrics_reg_class import clas_metrics
#evaluation metrics
evaluate = clas_metrics(y_true=y_test,y_pred=y_pred)
metrics = evaluate.metric_report()
print(metrics)
