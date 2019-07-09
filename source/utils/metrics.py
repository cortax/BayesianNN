import os
import sklearn.metrics as met
import numpy as np
from source.utils.utils import make_dir

class Metrics:
    def __init__(self, true, pred, target_type_string, list_labels=None):
        self.true = true
        self.pred = pred
        self.labels = list_labels
        self.target_type_string = target_type_string

        if target_type_string == 'Classification':
            self.accuracy = met.accuracy_score(true, pred) * 100
            self.recall = met.recall_score(true, pred, average='micro') * 100
            self.precision = met.precision_score(true, pred, average='micro') * 100
            self.f1 = met.f1_score(true, pred, average='weighted', labels=list_labels) * 100
            self.confusion_matrix = met.confusion_matrix(true, pred, labels=list_labels)
        elif target_type_string == 'Regression':
            self.expl_var = met.explained_variance_score(true, pred)
            #self.max_error = met.max_error(true, pred)
            self.mean_abs_error = met.mean_absolute_error(true, pred)
            self.mean_square_error = met.mean_squared_error(true, pred)
            #self.mean_square_log_error = met.mean_squared_log_error(true, pred)
            self.median_abs_error = met.median_absolute_error(true, pred)
            self.r2_score = met.r2_score(true, pred)

