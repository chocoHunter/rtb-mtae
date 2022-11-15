import sys
import numpy as np
import tensorflow as tf
from sklearn import metrics


epsilon = sys.float_info.epsilon


def auc(y_true, y_pred):
    def fallback_auc(y_true, y_pred):
        try:
            return metrics.roc_auc_score(y_true, y_pred)
        except:
            return 0.5
    return tf.py_function(fallback_auc, (y_true, y_pred), tf.double)


def anlp(z_true, z_pdf):
    z_size = len(z_true)
    nlp = np.zeros(shape=(z_size, ))
    for i in range(z_size):
        nlp[i] = -np.log(z_pdf[i][z_true[i]] + epsilon)
    return np.sum(nlp) / z_size


def anlp_fixed(z_true, z_pdf):
    total_num = len(z_true)
    return sum([-np.log(z_pdf[z] + epsilon) for z in z_true]) / total_num


def anlp_z_prob(z_prob):
    return sum(-np.log(z_prob + epsilon)) / len(z_prob)