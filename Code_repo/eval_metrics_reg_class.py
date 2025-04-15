## Evaluation metrics
import numpy as np
from sklearn.metrics import roc_auc_score

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
        metrics = {"Accuracy" : accuracy_score(self.y_true, self.y_pred),
                   "Preicision" : precision_score(self.y_true, self.y_pred),
                   "Recall":recall_score(self.y_true, self.y_pred),
                   "f1":f1_score(self.y_true, self.y_pred)}

        metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "value"])
        return metrics_df