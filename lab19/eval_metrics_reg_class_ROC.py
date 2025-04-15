## Evaluation metrics
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

from sklearn.metrics._regression import (mean_squared_error,mean_absolute_error,
    r2_score,explained_variance_score)

from sklearn.metrics._classification import (accuracy_score, precision_score, recall_score, f1_score)

import pandas as pd

class Regression_eval:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)

    def reg_metrics(self):
        metrics = {
            'MSE': mean_squared_error(self.y_true, self.y_pred),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'R2' : r2_score(self.y_true, self.y_pred),
            'Explained variance': explained_variance_score(self.y_true, self.y_pred)
        }
        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "value"])
        return metrics_df




class clas_metrics:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def metric_report(self):
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        print("\n--Metric Report--\n")
        metrics = {"Accuracy" : accuracy_score(self.y_true, self.y_pred),
                   "Precision" : precision_score(self.y_true, self.y_pred),
                   "Recall":recall_score(self.y_true, self.y_pred),
                   "Specificity": tn / (tn + fp),
                   "f1":f1_score(self.y_true, self.y_pred)}

        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "value"])
        return metrics_df


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

class ROC_AUC:
    def __init__(self, model, y_true, X_test, title, class_index):
        self.y_true = y_true
        self.X_test = X_test
        self.model = model
        self.title = title
        self.class_index =class_index

    def plot_roc(self):
        y_score = self.model.predict_proba(self.X_test)[:, self.class_index]  # class 1 probabilities

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(self.y_true, y_score)
        print("\nThreshold used ",thresholds)
        roc_auc = auc(fpr, tpr)

        #Plotting
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='red', lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')  # random chance line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Receiver Operating Characteristic (ROC) ; {self.title}")
        plt.legend(loc="lower right")
        plt.grid()