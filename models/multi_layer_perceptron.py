import numpy as np
from sklearn.neural_network import MLPClassifier
from time import time
import pandas as pd
import utils
import json
import os
import joblib

from data import Dataset

"""
Predict class from image data using a multi-layer perceptron.

Hyperparameters:
- threshold [-1, 1]
- regularization {"none", "L1", "L2", "elastic"}
- reg_c [0.0, inf]
"""


class Model():
    def __init__(self, activation="identity", batch_size=8, lr=0.001, n_layers=1, n_layer_neurons=10, **kwargs):
        # check hparams
        assert activation in ["identity", "logistic", "tanh", "relu"]
        assert batch_size > 1
        assert lr > 0.0
        assert n_layers >= 1
        assert n_layer_neurons >= 1

        self.activation = activation
        self.batch_size = batch_size
        self.lr = lr
        self.hidden_layer_sizes = (n_layer_neurons,) * n_layers
        self.n_layers = n_layers
        self.n_layer_neurons = n_layer_neurons

        self.model = None

        self.filename_root = f"multi_layer_perceptron/model_act{self.activation}_b{self.batch_size}_lr{self.lr}_nl{self.n_layers}_nn{self.n_layer_neurons}"
        self.filename = self.filename_root + ".sav"


    def train(self, dataset):
        # create model
        self.model = MLPClassifier(
            activation=self.activation,
            batch_size=self.batch_size,
            learning_rate_init=self.lr,
            max_iter=100, # this is slow enough already :(
            verbose=True
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
        os.makedirs("k_nearest_neighbors", exist_ok=True)
        joblib.dump(self.model, self.filename)


    def load(self):
        # load model from file
        self.model = joblib.load(self.filename)


grid = {
    "activation": ["logistic", "tanh", "relu"],
    "batch_size": [16, 64, 256],
    "lr": [0.0001, 0.001, 0.01],
    # "n_layers": [1, 2, 3],
    # "n_hidden_neurons": [5, 10, 20] # keep this small
}

if __name__ == "__main__":

    # get data
    dataset = Dataset(
        "./Miniproject/cifar-10-batches-py",
        # select_classes=[3, 5],
        seed=0
    )

    model = Model()
    model.train(dataset)