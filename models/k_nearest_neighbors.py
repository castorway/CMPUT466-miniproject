from sklearn.neighbors import KNeighborsClassifier
from time import time
import utils
import os
import joblib

from data import Dataset

"""
Predict class from image data using a K-nearest-neighbours model.
"""


class Model():
    def __init__(self, results_dir, k=3, weights="uniform", **kwargs):
        self.results_dir = results_dir
        
        # check hparams
        assert k >= 1
        assert weights in ["uniform", "distance"]

        self.k = k
        self.weights = weights

        self.model = None

        self.filename = self.results_dir / f"k_nearest_neighbors/model_k{self.k}_w{self.weights}.sav"


    def train(self, dataset):
        # create model
        self.model = KNeighborsClassifier(
            n_neighbors=self.k,
            weights=self.weights
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
    "k": [1, 3, 5, 7],
    "weights": ["uniform", "distance"],
}
name = "k_nearest_neighbors"


if __name__ == "__main__":

    # get data
    dataset = Dataset(
        "./Miniproject/cifar-10-batches-py",
        # select_classes=[3, 5],
        seed=0
    )

    model = Model(k=5, weights="uniform")
    model.train(dataset)