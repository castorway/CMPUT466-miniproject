import torch
from torch import nn
import torch.nn.functional as F
from data import Dataset
from time import time
import numpy as np
import utils
import os

"""
Predict class from image data using a (very simple) CNN.

Hyperparameters:
- regularization {"none", "L1", "L2", "elastic"}
- reg_c [0.0, inf]
"""

MAX_EPOCHS = 50

class SimpleCNN(nn.Module):
    def __init__(self, n_layers=1, pooling_type=None):
        super(SimpleCNN, self).__init__()

        self.n_layers = n_layers

        if self.n_layers == 1:
            self.body = nn.Conv2d(1, 1, 3)
        elif self.n_layers == 2:
            self.body = nn.Sequential(
                nn.Conv2d(1, 1, 3),
                nn.Conv2d(1, 1, 3)
            )
        elif self.n_layers == 3:
            self.body = nn.Sequential(
                nn.Conv2d(1, 1, 3),
                nn.Conv2d(1, 1, 3),
                nn.Conv2d(1, 1, 3)
            )
        else:
            raise ValueError()

        self.pool = nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        x = self.body(x)
        x = self.pool(x)
        return x


class Model():
    def __init__(self, n_layers=1, lr=10, batch_size=8, **kwargs):
        # check hparams
        assert n_layers in [1, 2, 3]
        assert 0.0 < lr
        assert 1 <= batch_size

        self.n_layers = n_layers
        self.lr = lr
        self.batch_size = batch_size

        self.filename = f"cnn/model_n{self.n_layers}_lr{self.lr}_b{self.batch_size}.pt"

        self.model = SimpleCNN(n_layers=self.n_layers)
        
        self.loss_func = torch.nn.L1Loss()


    def train(self, dataset):
        # create model
        self.model = SimpleCNN(n_layers=self.n_layers)

        print(f"Fitting model...")

        t_start = time()

        m_train = dataset.train_data["data"].shape[0]
        m_val = dataset.val_data["data"].shape[0]

        train_data = torch.tensor(dataset.train_data["data"].astype(np.float32)).reshape(m_train, 1, 32, 32)
        train_labels = torch.tensor(dataset.train_data["labels"].astype(np.float32)).reshape(m_train, 1, 1, 1)
        val_data = torch.tensor(dataset.val_data["data"].astype(np.float32)).reshape(m_val, 1, 32, 32)
        val_labels = torch.tensor(dataset.val_data["labels"].astype(np.float32)).reshape(m_val, 1, 1, 1)

        for epoch in range(MAX_EPOCHS):
            b = 0
            epoch_loss = 0

            # for each batch in dataset...
            while b < m_train:

                # format data
                data = train_data[b:b+self.batch_size, :]
                labels = train_labels[b:b+self.batch_size, :]

                # get prediction
                pred = self.model(data)

                # calculate loss
                loss = self.loss_func(pred, labels)
                epoch_loss += loss.item()

                # backprop
                self.model.zero_grad()
                loss.backward()
                
                # backprop without an optimizer, closer to what we did in course
                # https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
                with torch.no_grad():
                    for param in self.model.parameters():
                        param -= self.lr * param.grad

                b += self.batch_size

            print(f"Epoch {epoch}, loss={epoch_loss}")

        t = time() - t_start

        print(f"> Model trained in {t:.2f} s.")

        train_pred = self.model(train_data)
        train_acc = utils.calc_accuracy(train_pred.detach().numpy(), train_labels.detach().numpy())

        val_pred = self.model(val_data)
        val_acc = utils.calc_accuracy(val_pred.detach().numpy(), val_labels.detach().numpy())

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
        os.makedirs("logistic_regression", exist_ok=True)

        torch.save(self.model)

    
    # def make_data(self, data):
    #     # convert M x HW to M x H x W
    #     m = data.shape[0]
    #     s = np.sqrt(hw)
    #     assert s == int(s)
    #     h = w = int(s)

    #     data = np.reshape(data, (m, 1, h, w))
        
    #     # convert to tensor
    #     data = torch.tensor(data)
    #     return data


    def load(self):
        # load model from file
        weights = np.load(self.filename + "_weights")
        bias = np.load(self.filename + "_bias")

        self.model = LogisticRegression()
        self.model._coef = weights
        self.model._intercept = bias


grid = {
    "n_layers": [1, 2, 3],
    "batch_size": [8, 16, 32],
    "lr": [0.01, 0.03, 0.1, 0.3]
}
