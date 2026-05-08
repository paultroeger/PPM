# Imports and useful functions
import numpy as np
import pandas as pd
import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pytest
sns.set_style("whitegrid")

# This is the gradient of the logistic loss
def gradient(weights, xi, yi):
    exponent = yi * (xi.dot(weights))
    return - (yi*xi) / (1+np.exp(exponent))


def rdpgd(iterations, alpha, epsilon_bar):
    # TODO: your code here
    raise NotImplementedError()


def dpsgd(iterations, epsilon, delta, learning_rate, batch_size, C):
    # YOUR CODE HERE
    raise NotImplementedError()


if __name__ == "__main__":
    # TODO: your code here
    raise NotImplementedError()
