import numpy as np
from sklearn.linear_model import LogisticRegression
from time import time
import pandas as pd
import utils
import json
import os
import joblib

from data import Dataset

"""
Predict class from image data using a logistic regression model.
"""


class Model():
    def __init__(self, results_dir, fit_intercept=False, regularization="none", reg_c=1.0, **kwargs):
        self.results_dir = results_dir

        # check hparams
        # assert fit_intercept in [True, False]
        assert regularization in ["none", "L1", "L2", "elastic"]
        # assert reg_c >= 0.0

        self.fit_intercept = fit_intercept
        self.reg_c = reg_c

        # convert from my format to sklearn's
        if regularization == "none":
            self.regularization = None
        elif regularization == "L1":
            self.regularization = "l1"
        elif regularization == "L2":
            self.regularization = "l2"

        self.model = None

        self.filename = self.results_dir / f"logistic_regression/model_fi{self.fit_intercept}_reg{self.regularization}_c{self.reg_c}.sav"


    def train(self, dataset):
        # create model
        self.model = LogisticRegression(
            solver="saga",
            verbose=True,
            fit_intercept=self.fit_intercept,
            C=self.reg_c,
            max_iter=100)

        print(f"Fitting model...")

        t_start = time()
        self.model.fit(dataset.train_data['data'], dataset.train_data['labels'])
        t = time() - t_start

        print(f"> Model fit in {t:.2f} s.")

        train_pred = self.model.predict(dataset.train_data['data'])
        train_acc = utils.calc_accuracy(train_pred, dataset.train_data['labels'])

        val_pred = self.model.predict(dataset.val_data['data'])
        val_acc = utils.calc_accuracy(val_pred, dataset.val_data['labels'])

        print(f"> Train accuracy: {train_acc}, Val accuracy: {val_acc}")

        return train_acc, val_acc


    def test(self, dataset):

        print("Testing model...")

        test_pred = self.model.predict(dataset.test_data['data'])
        test_acc = utils.calc_accuracy(test_pred, dataset.test_data['labels'])

        print(f"> Test accuracy: {test_acc}")
        return test_acc


    def save(self):        
        # save model in logistic_regression dir
        os.makedirs(self.filename.parent, exist_ok=True)
        joblib.dump(self.model, self.filename)


    def load(self):
        # load model from file
        self.model = joblib.load(self.filename)


grid = {
    "fit_intercept": [False, True],
    "regularization": ["none", "L1", "L2"],
    # "reg_c": [0.3, 1.0, 3]
}
name = "logistic_regression"


if __name__ == "__main__":

    # get data
    dataset = Dataset(
        "./Miniproject/cifar-10-batches-py",
        # select_classes=[3, 5],
        seed=0
    )

    model = Model()
    model.train(dataset)
    model.save()
    
    model.load()
    model.test(dataset)