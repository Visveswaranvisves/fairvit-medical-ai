import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

def group_metrics(y_true, y_pred, groups):
    """
    Compute accuracy and AUC per demographic group
    """
    results = {}

    unique_groups = np.unique(groups)

    for g in unique_groups:
        idx = groups == g
        if idx.sum() < 5:
            continue

        acc = accuracy_score(y_true[idx], y_pred[idx] > 0.5)
        auc = roc_auc_score(y_true[idx], y_pred[idx])

        results[g] = {
            "accuracy": acc,
            "auc": auc,
            "samples": idx.sum()
        }

    return results
