# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pytest
plt.style.use('seaborn-whitegrid')


def laplace_mech(v, sensitivity, epsilon):
    # TODO: your code here
    raise NotImplementedError()


def avg_wages(data, epsilon):
    # TODO: your code here
    raise NotImplementedError()


def hrs_cdf(lfs):
  a = lfs['ATOTHRS']
  return [len(a[a < i]) for i in range(990)]


def hrs_cdf_dp_laplace(lfs, epsilon):
    # TODO: your code here
    raise NotImplementedError()


def hrs_cdf_dp_gauss(lfs, epsilon, delta):
    # TODO: your code here
    raise NotImplementedError()


def hrs_cdf_v2(lfs, epsilon):
    # TODO: your code here
    raise NotImplementedError()


def rdp_mech(alpha):
    # TODO: your code here
    raise NotImplementedError()


def convert_RDP_ED(alpha, epsilon_bar, delta):
    # TODO: your code here
    raise NotImplementedError()


def encode_response_sales(response, alpha):
    # TODO: your code here
    raise NotImplementedError()


def decode_responses_sales(responses, alpha):
    # TODO: your code here
    raise NotImplementedError()


if __name__ == "__main__":
    # TODO: your code here
    raise NotImplementedError()
