import numpy as np

def calc_accuracy(pred, labels):
    """
    Compute accuracy as (number of correct predictions) / (total predictions).
    """
    print(pred)
    print(labels)
    # threshold_pred = np.where(pred < 0.5, -1, 1)
    return np.count_nonzero(pred == labels) / len(labels)