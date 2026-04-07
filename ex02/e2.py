# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pytest


# plt.style.use('seaborn-whitegrid')


def laplace_mech(v, sensitivity, epsilon):
    if epsilon <= 0: return v
    b = sensitivity / epsilon
    noise = np.random.laplace(loc=0, scale=b)
    return v + noise


def test_laplace_mech_adds_noise():
    # With non-zero sensitivity, output should not equal input exactly
    result = laplace_mech(10.0, 1.0, 1.0)
    assert result != 10.0


def test_laplace_mech_zero_sensitivity_zero_epsilon():
    # sensitivity=0 → b=0 → no noise, output equals input exactly
    # epsilon=0 → b=0 → no noise, output equals input exactly
    result = laplace_mech(7.0, 0.0, 1.0)
    result2 = laplace_mech(7.0, 1.0, 0.0)
    assert result == 7.0
    assert result2 == 7.0


# prints info about wages
def avg_wages_helper(wages, cap, epsilon):
    value = laplace_mech(wages[wages > cap].count(), 1, epsilon)
    print(f"How many people earn above {cap}? {value}")


def avg_wages(data, epsilon):
    # 4 parts for finding sensitivity, 1 part for query
    split_epsilon = epsilon / 5
    # use only wages and remove implied decimal
    wages = data['HRLYEARN'] / 100
    if wages.empty: return 0

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


def test_avg_wages_correct_mean():
    # The mean should reflect the raw values
    df = pd.DataFrame({'HRLYEARN': [99999] * 100})
    result = avg_wages(df, 1000)  # High epsilon to remove noise
    assert abs(result - 999.99) < 1


def test_avg_wages_empty_data():
    # Empty data should return 0
    df = pd.DataFrame({'HRLYEARN': []})
    result = avg_wages(df, 1000)
    assert result == 0


def hrs_cdf(lfs):
    a = lfs['ATOTHRS']
    return [len(a[a < i]) for i in range(990)]


def hrs_cdf_dp_laplace(lfs, epsilon):
    a = lfs['ATOTHRS']
    # For every value in 'a' we add noise
    # The noise will be very high, as sensitivity divided by epsilon is very large
    # We introduce a lower bound of 0, since having a negative amount does not make sense
    return [max(0, round(laplace_mech(len(a[a < i]), 1, epsilon / 990), 0)) for i in range(990)]


def hrs_cdf_dp_gauss(lfs, epsilon, delta):
    a = lfs['ATOTHRS']
    sensitivity = np.sqrt(990)
    sigma = (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon
    return [max(0, round(len(a[a < i]) + np.random.normal(0, sigma), 0)) for i in range(990)]


def hrs_cdf_v2(lfs, epsilon):
    a = lfs['ATOTHRS'] / 10  # convert to actual hours: 0.0–99.0
    return [max(0, round(laplace_mech(len(a[a < i]), 1, epsilon / 100), 0)) for i in range(100)]


# Here I assume that we should use actual hours, not each hour in decimal.
def rdp_mech(alpha):
    l2_sensitivity = np.sqrt(112)
    epsilon_bar = 0.001
    sigma = np.sqrt(alpha * l2_sensitivity ** 2 / (2 * epsilon_bar))

    df = pd.read_csv('2025-02-CSV/pub0225.csv')
    a = df['ATOTHRS'] / 10
    return [max(0, round(len(a[a < i]) + np.random.normal(0, sigma), 0)) for i in range(100)]


def convert_RDP_ED(alpha, epsilon_bar, delta):
    epsilon = epsilon_bar + (np.log(1 / delta) / (alpha - 1))
    return epsilon


def encode_response_sales(response, alpha):
    if np.random.random() < 0.5 + alpha:
        return response
    else:
        return not response


def decode_responses_sales(responses, alpha):
    n = len(responses)
    noisy_count = sum(responses)
    true_count = (noisy_count - n * (0.5 - alpha)) / (2 * alpha)
    return true_count


# Currently the clipping parameter is 200 CAD. We chose it since it is reasonably above the avg of 35 CAD, but not too
# unreasonably high.
# The sensitivity we used is calculated by dividing the upper bound with the amount of entries in the dataset. The
# upper bound is equal to the clipping parameter.
# We apply the laplace mechanism exactly one with the given epsilon and since we only do one computation on the data
# with this epsilon, the total privacy cost is epsilon.
def task1():
    df = pd.read_csv('2025-02-CSV/pub0225.csv')
    print("AVG Wage with epsilon 0.01:", avg_wages(df, 0.01).round(2), "CAD")
    print("AVG Wage with epsilon 0.1:", avg_wages(df, 0.1).round(2), "CAD")
    print("AVG Wage with epsilon 1:", avg_wages(df, 1).round(2), "CAD \n")


# The L1 and L2 global sensitivity measure how much the output of hrs_cdf can change when one person is added or
# removed from the dataset. One person can appear in at most 990 buckets, changing each by +-1, so L1 sensitivity = 990
# (sum of all changes) and L2 sensitivity = sqrt(990) ≈ 31.5 (Euclidean norm of the change vector).
# According to the lectures a typical choice for delta is 1/n^2, which in this case would be 1/133780^2
# This leads to a delta of around 1e-11
# In theory, the privacy of the laplace implementation will be higher than the privacy of the gauss implementation,
# since laplace applies more noise per entry. However, the utility is severely harmed, since we cannot properly infer
# information from the output, while in the gauss implementation we can. As example, the mean error for laplace in this
# exercise is 1003, while for gauss it is just 175.
# We still have the same privacy cost, but by correctly interpreting the values and reducing the output vector from
# 999 to 100 buckets, the sensitivity drops around 10x. This results in around 10x less noise and therefore
# significantly better utility. This is cause by the smaller output vector, as fewer buckets mean one person affects
# fewer counts.
def task2():
    df = pd.read_csv('2025-02-CSV/pub0225.csv')
    # print("Laplace: \n")
    # print(hrs_cdf_dp_laplace(df, 1), "\n")
    # print("Gauss: \n")
    # print(hrs_cdf_dp_gauss(df, 1, 1e-11), "\n")
    print("Laplace v2: \n")
    print(hrs_cdf_v2(df, 1), "\n")


# We apply the RDP Gaussian mechanism exactly once, so the total privacy cost is (5, 0.001)-RDP.
# The total privacy cost in (ε, δ)-DP for an alpha of 5, epsilon_bar of 0.001 and delta of 1e-4 is ε = 2.3.
def task3():
    print("Renyi Differential Privacy: \n", rdp_mech(5), "\n")
    print("Converted Epsilon: ", convert_RDP_ED(5, 0.001, 1e-4), "\n")


# The answer we compute is not very accurate. This is because we have a very low alpha, which leads to very high noise.
# In addition, we also have over 100k people in the dataset, leading to possibly thousands of "wrong" answers.
def task4():
    df = pd.read_csv('2025-02-CSV/pub0225.csv')

    responses = [encode_response_sales(row == 12, 0.05) for row in df['NOC_43']]
    estimated_count = decode_responses_sales(responses, 0.05)
    true_count = len(df[df['NOC_43'] == 12])

    print("True count: ", true_count)
    print("Estimated count:", round(estimated_count))


if __name__ == "__main__":
    task1()
    task2()
    task3()
    task4()
