import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from time import time
import pandas as pd
import utils

from data import Dataset

"""
Predict class from image data using a linear regression model with thresholding.

Hyperparameters:
- threshold [-1, 1]
- regularization {"none", "L1", "L2"}
"""

def threshold(pred, thresh):
    return np.where(pred < thresh, -1, 1)


class Model():
    def __init__(self, threshold=0, regularization="none", **kwargs):
        self.threshold = threshold
        self.regularization = regularization

        # check hparams
        assert -1 <= self.threshold <= 1
        assert self.regularization in ["none", "L1", "L2"]

        self.model = None

    def train(self, dataset):
        # create model
        if self.regularization == "none":
            self.model = LinearRegression()
        elif self.regularization == "L1":
            self.model = Lasso()
        elif self.regularization == "L2":
            self.model = Ridge()
        
        print(f"Fitting model...")

        t_start = time()
        self.model.fit(dataset.train_data['data'], dataset.train_data['labels'])
        t = time() - t_start

        print(f"> Model fit in {t:.2f} s.")

        train_pred = self.model.predict(dataset.train_data['data'])
        train_pred = threshold(train_pred, self.threshold)
        train_acc = utils.calc_accuracy(train_pred, dataset.train_data['labels'])

        val_pred = self.model.predict(dataset.val_data['data'])
        val_pred = threshold(val_pred, self.threshold)
        val_acc = utils.calc_accuracy(val_pred, dataset.val_data['labels'])

        print(f"> Train accuracy: {train_acc}, Val accuracy: {val_acc}")

        return train_acc, val_acc


    def test(self, dataset):
        # create model
        if self.regularization == "none":
            self.model = LinearRegression()
        elif self.regularization == "L1":
            self.model = Lasso()
        elif self.regularization == "L2":
            self.model = Ridge()

        print(f"Fitting model...")

        # for linear regression we reproduce run by just retraining the model because it's easy
        t_start = time()
        self.model.fit(dataset.train_data['data'], dataset.train_data['labels'])
        t = time() - t_start
        print(f"> Model fit in {t:.2f} s.")

        train_pred = self.model.predict(dataset.train_data['data'])
        train_pred = threshold(train_pred, self.threshold)
        train_acc = utils.calc_accuracy(train_pred, dataset.train_data['labels'])

        val_pred = self.model.predict(dataset.val_data['data'])
        val_pred = threshold(val_pred, self.threshold)
        val_acc = utils.calc_accuracy(val_pred, dataset.val_data['labels'])

        print(f"> Train accuracy: {train_acc}, Val accuracy: {val_acc}")

        print("Testing model...")

        test_pred = self.model.predict(dataset.test_data['data'])
        test_pred = threshold(test_pred, self.threshold)
        test_acc = utils.calc_accuracy(test_pred, dataset.test_data['labels'])

        print(f"> Test accuracy: {test_acc}")
        return test_acc

    def save(self):
        pass

    def load(self):
        pass


grid = {
    "threshold": [-0.8, -0.5, -0.2, -0.1, 0, 0.1, 0.2, 0.5, 0.8],
    "regularization": ["none", "L1", "L2"]
}
name = "linear_regression"


