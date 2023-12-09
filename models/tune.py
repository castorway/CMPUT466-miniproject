import pandas as pd
import itertools
import linear_regression
import logistic_regression
import k_nearest_neighbors
import multi_layer_perceptron
from data import Dataset
import os
import numpy as np

# get data
dataset = Dataset(
    "./Miniproject/cifar-10-batches-py",
    select_classes=[3, 5],
    seed=0
)

def get_by_hparams(hparams, df):
    mask =  False

    found = df.copy()
    for k, v in hparams.items():
        found = found[ found[k] == v ]
    
    return found


def tune(module):
    cols = list(module.grid.keys()) + ["train_acc", "val_acc", "test_acc"]

    # load csv
    csv_name = f"{module.__name__}_tuning.csv"
    if os.path.exists(csv_name):
        df = pd.read_csv(csv_name, index_col=None)
    else:
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
        model = module.Model(**hparams)
        
        # run the model with these hyperparams
        train_acc, val_acc = model.train(dataset)

        # record info (hparams with accuracy results) in dataframe
        info = dict(hparams, **{"train_acc": train_acc, "val_acc": val_acc})
        df.loc[len(df.index)] = info

        # save model weights
        model.save()

    # save csv back
    df.to_csv(f"{module.__name__}_tuning.csv", index=False)

    return df


def select_and_test(module):
    # load csv
    csv_name = f"{module.__name__}_tuning.csv"
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
        model = module.Model(**hparams_dict)
        model.load() # load any data if exists
        test_acc = model.test(dataset)

        # write result to df
        df.loc[df.index == idx, "test_acc"] = test_acc

        print()

    # save csv back
    df.to_csv(f"{module.__name__}_tuning.csv", index=False)


def title_print(s):
    print(f"{s:{'='}^100}")


if __name__ == "__main__":

    # linear regression...
    # title_print("TUNING LINEAR REGRESSION")
    # df = tune(linear_regression)
    # print(df)

    # title_print("TESTING LINEAR REGRESSION")
    # select_and_test(linear_regression)

    # logistic regression...
    # title_print("TUNING LOGISTIC REGRESSION")
    # tune(logistic_regression)

    # title_print("TESTING LOGISTIC REGRESSION")
    # select_and_test(logistic_regression)

    # CNN...
    # title_print("TUNING K NEAREST NEIGHBOURS")
    # tune(k_nearest_neighbors)

    # title_print("TESTING K NEAREST NEIGHBOURS")
    # select_and_test(k_nearest_neighbors)

    # CNN...
    title_print("TUNING MULTI LAYER PERCEPTRON")
    tune(multi_layer_perceptron)

    title_print("TESTING MULTI LAYER PERCEPTRON")
    select_and_test(multi_layer_perceptron)

