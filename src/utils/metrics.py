import numpy as np
from sklearn import metrics


def auc_pauc(scores, labels, pauc_fpr=0.1):
    auc = metrics.roc_auc_score(labels, scores)
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    # pAUC from 0 to pauc_fpr
    idx = np.searchsorted(fpr, pauc_fpr, side="right")
    pauc = metrics.auc(fpr[:idx], tpr[:idx]) / pauc_fpr
    return auc, pauc
