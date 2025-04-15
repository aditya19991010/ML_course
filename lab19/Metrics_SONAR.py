import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, LabelEncoder

#Using "Sonar" ; Classification data
df = pd.read_csv("/home/ibab/learning/ML_Lab/datasets/sonar data.csv")

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
y = LabelEncoder().fit_transform(y)

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train classifier with probability output
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)
y_pred= model.predict(X_test)

#Metrics
from eval_metrics_reg_class_ROC import clas_metrics, ROC_AUC
evaluate = clas_metrics(y_true=y_test,y_pred=y_pred)
metrics = evaluate.metric_report()
print(metrics)

#ROC - AUC info
curve = ROC_AUC(model, y_true=y_test,X_test=X_test, title="Sonar data", class_index=1)
curve.plot_roc()
plt.show()