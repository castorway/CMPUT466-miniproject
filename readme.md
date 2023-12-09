# CMPUT 466 Miniproject

Code to load and preprocess the CIFAR 10 dataset, and run several models from scikit-learn for classification.

## Dataset

I used the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) for this assignment. To reproduce my results, download the dataset ("CIFAR-10 python version" on the webpage), untar it, and put the `cifar-10-batches-py` directory inside this directory (i.e. `CMPUT 466 Miniproject/cifar-10-batches-py`).

## Dependencies

To load the dependencies with Anaconda, run `conda env create -f env.yml`. (There may be some extraneous dependencies in the environment; it is what I used for most of this course, and I provide it for convenience/reproducibility.)

## Program

To run the program, run `python main.py`.

This script creates the required directories and runs all the models with all the hyperparameter combinations I considered for tuning, creates CSV files and saved model files as it goes, and then chooses the best model (by validation accuracy) to test. The script may take a long time to run, especially when it gets to the logistic regression and multi-layer perceptrons. As it runs, results are cached in the CSV files; if you kill the program and rerun it, it will pick up where it left off without re-running anything that already has results in the CSV files.

## Results

Results are stored in the `CMPUT 466 Miniproject/results` directory. I have uploaded the CSV files generated to this repository, but not the saved model files.