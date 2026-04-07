# %%
# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pytest
try:
    plt.style.use('seaborn-whitegrid')
except:
    plt.style.use('seaborn-v0_8-whitegrid')

# our functions

CSV_PATH = 'pub0225.csv'

def load_data():
    return pd.read_csv(CSV_PATH)

# prints info about wages
def avg_wages_helper(wages, cap, epsilon):
    value = laplace_mech(wages[wages > cap].count(), 1, epsilon)
    print(f"How many people earn above {cap}? {value}")
        

# predefined functions

def laplace_mech(v, sensitivity, epsilon):
    scale = sensitivity / epsilon
    return v + np.random.laplace(0, scale)


def avg_wages(data, epsilon):
    # 4 parts for finding sensitivity, 1 part for query
    split_epsilon = epsilon / 5
    # use only wages and remove implied decimal
    wages = data['HRLYEARN'] / 100
    
    cap = 50
    avg_wages_helper(wages, cap, split_epsilon)
    cap = 100
    avg_wages_helper(wages, cap, split_epsilon)
    cap = 200
    avg_wages_helper(wages, cap, split_epsilon)
    cap = 250
    avg_wages_helper(wages, cap, split_epsilon)

    cap = 230
    return laplace_mech(wages.mean(), cap, split_epsilon)


def hrs_cdf(lfs):
  a = lfs['ATOTHRS']
  return [len(a[a < i]) for i in range(990)]


def hrs_cdf_dp_laplace(lfs, epsilon):
    cdf = np.array(hrs_cdf(lfs))
    sensitivity = 990
    scale = sensitivity / epsilon
    return cdf + np.random.laplace(0, scale, len(cdf))


def hrs_cdf_dp_gauss(lfs, epsilon, delta):
    cdf = np.array(hrs_cdf(lfs))
    sensitivity = np.sqrt(990)
    scale = 2 * np.pow(sensitivity, 2) * np.log(1.25/delta)
    scale /= np.pow(epsilon, 2)
    return cdf + np.random.normal(0, scale, len(cdf))


def hrs_cdf_v2(lfs):
    a = lfs['ATOTHRS'] // 10
    return [len(a[a < i]) for i in range(99)]


def rdp_mech(alpha):
    df = hrs_cdf(load_data())
    epsilon_bar = 0.001
    sensitivity = 112
    scale = np.pow(sensitivity, 2) * alpha
    scale /= 2 * epsilon_bar
    return df + np.random.normal(0, scale, len(df))


def convert_RDP_ED(alpha, epsilon_bar, delta):
    epsilon = epsilon_bar 
    epsilon += np.log(1/delta) / (alpha - 1)
    return (epsilon, delta)


def encode_response_sales(response, alpha):
    one_hot = [0] * 44
    if(np.isnan(response)):
        one_hot[0] = 1
    else:
        one_hot[int(response)] = 1
    
    one_hot_changed = []
    for b in one_hot:
        if b == 0:
            one_hot_changed.append(np.random.choice([1, 0], p=[alpha, 1 - alpha]))
        else:
            one_hot_changed.append(np.random.choice([0, 1], p=[alpha, 1 - alpha]))
    return pd.Series(one_hot_changed)


def decode_responses_sales(responses, alpha):
    sum_res = np.sum(responses, axis=0)
    p, q = 1 - alpha, alpha
    return (sum_res - len(responses) * q) / (p - q)


#%% Tasks
def task1():
    print('-' * 6, 'Task 1', '-' * 6)
    df = load_data()
    avg = avg_wages(df, 20)
    print(f"The differential privacy avg wage is {avg}.")
    print()

def task2():
    print('-' * 6, 'Task 2', '-' * 6)
    df = load_data()
    cdf = hrs_cdf(df)
    cdf_dp_laplace = hrs_cdf_dp_laplace(df, 1)
    
    delta = 0.9 * (1/np.pow(len(df), 2))
    cdf_dp_gauss = hrs_cdf_dp_gauss(df, 1, delta)
    
    total_diff_laplace = np.sum(np.abs(cdf-cdf_dp_laplace))
    print(f"Total diff laplace : {total_diff_laplace}")
    total_diff_gauss = np.sum(np.abs(cdf-cdf_dp_gauss))
    print(f"Total diff gauss : {total_diff_gauss}")

    print("cdf_v2:")
    print(hrs_cdf_v2(df), "\n")
    print()

def task3():
    print('-' * 6, 'Task 3', '-' * 6)
    df = load_data()['ATOTHRS']
    rdp_mech(5)
    print(f"ED: {convert_RDP_ED(5, 0.001, 10**(-5))}")
    print()

def task4():
    print('-' * 6, 'Task 4', '-' * 6)
    occ_data = load_data()['NOC_43']
    alpha = 0.05
    encoded = occ_data.apply(encode_response_sales, alpha=alpha)
    decoded = decode_responses_sales(encoded, alpha)
    print(decoded)
    print(f"Decoded Value: {decoded.iloc[12]}")
    print(f"Actual Value: {len(occ_data[occ_data == 12])}")
    print()


# %%
if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()

# %%
