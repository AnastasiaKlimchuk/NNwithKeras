import pandas as pd
import numpy as np

EPOSHS = 30 # Turn epochs to 30 to get 0.9967 accuracy
BATCH_SIZE = 86


def load_data(train_url, test_url):
    # Load the data
    train = pd.read_csv(train_url)
    test = pd.read_csv(test_url)
    Y_train = train["label"]
    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1)

    return X_train, Y_train, test


def normalize_data(train, test):
    return train/255.0, test/255.0


def reshape_data(train, test):
    X_train = train.values.reshape(-1, 28, 28, 1)
    test = test.values.reshape(-1, 28, 28, 1)
    return X_train, test
