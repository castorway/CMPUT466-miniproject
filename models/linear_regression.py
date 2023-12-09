from sklearn.linear_model import RidgeClassifier
from time import time
import utils
import os
import joblib

"""
Predict class from image data using a linear regression model.
"""


class Model():
    def __init__(self, results_dir, alpha=1.0, fit_intercept=False, **kwargs):
        self.results_dir = results_dir
        
        self.alpha = alpha
        self.fit_intercept = fit_intercept

        # check hparams
        assert 0.0 < self.alpha
        assert self.fit_intercept in [True, False]

        self.model = None

        self.filename = self.results_dir / f"linear_regression/model_fi{self.fit_intercept}_a{self.alpha}.sav"


    def train(self, dataset):
        # create model
        self.model = RidgeClassifier(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept
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
        # save model in linear_regression dir
        os.makedirs(self.filename.parent, exist_ok=True)
        joblib.dump(self.model, self.filename)

    def load(self):
        # load model from file
        self.model = joblib.load(self.filename)


grid = {
    "alpha": [0.3, 0.7, 1.0, 3.0, 7.0],
    "fit_intercept": [False, True],
}
name = "linear_regression"
