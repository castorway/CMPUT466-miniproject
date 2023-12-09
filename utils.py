import numpy as np

def calc_accuracy(pred, labels):
    """
    Compute accuracy as (number of correct predictions) / (total predictions).
    """
    # print(pred)
    # print(labels)
    return np.count_nonzero(pred == labels) / len(labels)