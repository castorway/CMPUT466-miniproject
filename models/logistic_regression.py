from sklearn.linear_model import LogisticRegression
from time import time
import utils
import os
import joblib

"""
Predict class from image data using a logistic regression model.
"""


class Model():
    def __init__(self, results_dir, fit_intercept=False, alpha=1.0, **kwargs):
        self.results_dir = results_dir

        # check hparams
        assert fit_intercept in [True, False]
        # assert regularization in ["none", "L1", "L2"]
        assert alpha >= 0.0
        self.alpha = alpha

        self.fit_intercept = fit_intercept
        self.reg_c = 1.0 / (2*self.alpha)

        self.model = None

        self.filename = self.results_dir / f"logistic_regression/model_fi{self.fit_intercept}_a{self.alpha}.sav"


    def train(self, dataset):
        # create model
        self.model = LogisticRegression(
            verbose=True,
            penalty="l2",
            solver="sag",
            fit_intercept=self.fit_intercept,
            C=self.reg_c,
        )

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
    "alpha": [0.3, 0.7, 1.0, 3.0, 7.0],
    "fit_intercept": [False, True],
}
name = "logistic_regression"
