import pandas as pd
import itertools
from data import Dataset
import os
import numpy as np
from pathlib import Path

import models.linear_regression as linear_regression
import models.k_nearest_neighbors as k_nearest_neighbors
import models.logistic_regression as logistic_regression
import models.multi_layer_perceptron as multi_layer_perceptron
import models.naive_bayes as naive_bayes

# get directories
main_dir = Path(__file__).parent
results_dir = main_dir / "results"
data_dir = main_dir / "cifar-10-batches-py"
os.makedirs(results_dir, exist_ok=True)


def get_by_hparams(hparams, df):
    found = df.copy()
    for k, v in hparams.items():
        found = found[ found[k] == v ]
    
    return found


def tune(module):
    """
    Run model defined in module for each hyperparameter combination defined in module.grid.
    Save results in a CSV.
    """
    
    cols = list(module.grid.keys()) + ["train_acc", "val_acc", "test_acc"]

    # load csv
    csv_name = results_dir / f"{module.name}_tuning.csv"
    if os.path.exists(csv_name):
        print(f"Reading existing results from {csv_name}.")
        df = pd.read_csv(csv_name, index_col=None)
    else:
        print(f"Couldn't find {csv_name}.")
        df = pd.DataFrame(columns=cols)

    # get list of hyperparam combinations to try
    hparam_combs = itertools.product(*[v for v in module.grid.values()])
    hparam_combs = [dict(zip(module.grid.keys(), values)) for values in hparam_combs]

    for hparams in hparam_combs:
        print("Training with hyperparameters:", hparams)

        if len(get_by_hparams(hparams, df).index) > 0:
            info = dict(df.loc[0])
            print(f"> already done: train_acc={info['train_acc']}, val_acc={info['val_acc']}, test_acc={info['test_acc']}")
            continue
        
        # create model from hparams
        model = module.Model(results_dir, **hparams)
        
        # run the model with these hyperparams
        train_acc, val_acc = model.train(dataset)

        # record info (hparams with accuracy results) in dataframe
        info = dict(hparams, **{"train_acc": train_acc, "val_acc": val_acc})
        df.loc[len(df.index)] = info

        # save model weights
        model.save()

        # save csv back
        df.to_csv(csv_name, index=False)

    return df


def select_and_test(module):
    """
    Choose the best model (by validation accuracy) and test it on the test set.
    """
    
    # load csv
    csv_name = results_dir / f"{module.name}_tuning.csv"
    df = pd.read_csv(csv_name, index_col=None)

    # pick row with best val_acc
    best_loc = df['val_acc'] == max(list(df['val_acc']))
    best = df.loc[best_loc]

    print("Best model(s) from tuning:")
    print(best)
    print()

    for idx in best.index:
        print(f"--- Selected best model at index {idx}. ---")

        # test model
        hparams_dict = df[df.index == idx].to_dict(orient="records")[0]
        model = module.Model(results_dir, **hparams_dict)
        model.load() # load any data if exists
        test_acc = model.test(dataset)

        # write result to df
        df.loc[df.index == idx, "test_acc"] = test_acc

        print()

    # save csv back
    df.to_csv(csv_name, index=False)


def title_print(s):
    print(f"{s:{'='}^100}")


if __name__ == "__main__":

    # get data
    dataset = Dataset(
        data_dir,
        seed=0
    )

    # logistic regression...
    title_print("TUNING LINEAR REGRESSION")
    tune(linear_regression)

    title_print("TESTING LINEAR REGRESSION")
    select_and_test(linear_regression)

    # logistic regression...
    title_print("TUNING LOGISTIC REGRESSION")
    tune(logistic_regression)

    title_print("TESTING LOGISTIC REGRESSION")
    select_and_test(logistic_regression)

    # k nearest neighbours...
    title_print("TUNING K NEAREST NEIGHBOURS")
    tune(k_nearest_neighbors)

    title_print("TESTING K NEAREST NEIGHBOURS")
    select_and_test(k_nearest_neighbors)

    # naive bayes...
    title_print("TUNING NAIVE BAYES")
    tune(naive_bayes)

    title_print("TESTING NAIVE BAYES")
    select_and_test(naive_bayes)

    # MLP...
    title_print("TUNING MULTI LAYER PERCEPTRON")
    tune(multi_layer_perceptron)

    title_print("TESTING MULTI LAYER PERCEPTRON")
    select_and_test(multi_layer_perceptron)

