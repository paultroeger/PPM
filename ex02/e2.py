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

# prints count of wages above the cap
def avg_wages_helper(wages, cap, epsilon):
    value = laplace_mech(wages[wages > cap].count(), 1, epsilon)
    print(f"How many people earn above {cap}? {value}")
        

# returns noisy query result according to laplace mechanism result = true_answer + Lap(b) with b = Δf/eps
def laplace_mech(v, sensitivity, epsilon):
    if epsilon <= 0: return v
    scale = sensitivity / epsilon
    return v + np.random.laplace(0, scale)

# returns the noisy average wage with a privacy budget of epsilon. We successively probed for the most sensible clipping parameter to decrease sensitivity
# and thus improve utility
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

# returns a vector with 990 entries (buckets). The i-th bucket contains the number of people who worked <i ATOTHRS (actual total hours)
def hrs_cdf(lfs):
    a = lfs['ATOTHRS']
    return [len(a[a < i]) for i in range(990)]

# returns a eps-DP result of the hrs_cdf function using the laplace mechanism
def hrs_cdf_dp_laplace(lfs, epsilon):
    cdf = np.array(hrs_cdf(lfs))
    sensitivity = 990
    scale = sensitivity / epsilon
    return cdf + np.random.laplace(0, scale, len(cdf))

# returns a eps-delta-DP result of the hrs_cdf function using the gaussian mechanism
def hrs_cdf_dp_gauss(lfs, epsilon, delta):
    cdf = np.array(hrs_cdf(lfs))
    sensitivity = np.sqrt(990)
    scale = 2 * np.pow(sensitivity, 2) * np.log(1.25/delta)
    scale /= np.pow(epsilon, 2)
    return cdf + np.random.normal(0, np.sqrt(scale), len(cdf))


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

# %% Tasks
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
    print(f"Total diff gauss   : {total_diff_gauss}")

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

#%% Tests

# Test T1
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

# Tests T2
# Mock data for testing
@pytest.fixture
def mock_lfs_data():
    return pd.DataFrame({
        'ATOTHRS': [10, 20, 30, 40, 50],
        'HRLYEARN': [2000, 3000, 4000, 5000, 6000],
        'NAICS_21': [1, 2, 12, 4, 12] # 12 is the target occupation
    })

def test_laplace_mech_noise():
    """Verify that laplace_mech adds noise and centers around the value."""
    v = 100
    sensitivity = 1
    epsilon = 0.1
    # Run multiple times to check if mean is roughly correct (law of large numbers)
    results = [laplace_mech(v, sensitivity, epsilon) for _ in range(1000)]
    assert np.mean(results) == pytest.approx(v, abs=5)
    assert np.any(np.array(results) != v) # Ensure noise is actually added



def test_hrs_cdf_length(mock_lfs_data):
    """CDF must return exactly 990 elements as per the spec."""
    res = hrs_cdf(mock_lfs_data)
    print(res)
    assert len(res) == 990
    # Since max value in mock ATOTHRS': [10, 20, 30, 40, 50] is 50, index 51 should be the total count (5)
    assert res[51] == 5 #res[51] = len(a[a < 51]) = the number of people who worked less than 51 hours 
    assert res[52] == res[51] # 51 and 52 should both yield 5

def test_hrs_cdf_dp_laplace_shape_and_noise(mock_lfs_data):
    """Verify DP version returns same shape and is noisy."""
    epsilon = 1.0
    res_raw = np.array(hrs_cdf(mock_lfs_data))
    res_dp = hrs_cdf_dp_laplace(mock_lfs_data, epsilon)
    
    assert len(res_dp) == 990
    assert not np.array_equal(res_raw, res_dp)

def test_hrs_cdf_gauss_length(mock_lfs_data):
    """CDF must return exactly 990 elements as per the spec for Gaussian."""
    epsilon = 1.0
    delta = 1e-5
    res = hrs_cdf_dp_gauss(mock_lfs_data, epsilon, delta)
    
    # The output must be a vector of counts with 990 entries 
    assert len(res) == 990
    
    # Similar to the Laplace test, verify it is adding noise 
    # and isn't just the raw data
    raw_res = hrs_cdf(mock_lfs_data)
    assert not np.array_equal(res, raw_res)

def test_avg_wages_accuracy(mock_lfs_data):
    """
    Tests the accuracy of the DP average wage function by using a very high 
    privacy budget (epsilon = 10,000). At this level, the Laplace noise 
    becomes negligible (scale ~ 0), allowing us to verify that the underlying 
    calculation logic—clipping, summing, and dividing—correctly matches 
    the actual mean of the dataset.
    """
    # Use a very large epsilon to minimize noise
    epsilon = 10000 
    
    # Calculate expected value manually:
    # 1. Divide by 100 as per your code logic
    wages = mock_lfs_data['HRLYEARN'] / 100
    # 2. Your code uses a mean of these wages
    expected_mean = wages.mean()
    
    result = avg_wages(mock_lfs_data, epsilon)
    
    # The result should be very close to the actual mean
    assert result == pytest.approx(expected_mean, abs=1.0)

def test_hrs_cdf_v2_length(mock_lfs_data):
    epsilon = 1.0
    res = hrs_cdf_v2(mock_lfs_data)
    
    assert len(res) == 99

# Tests T3
def test_rdp_mech_shape():
    result = rdp_mech(5)

    assert isinstance(result, np.ndarray)
    assert len(result) == 990


def test_rdp_mech_not_equal_true():
    true_cdf = np.array(hrs_cdf(load_data()))
    noisy = rdp_mech(5)

    assert not np.array_equal(true_cdf, noisy)


def test_rdp_mech_seed():
    np.random.seed(42)
    r1 = rdp_mech(5)

    np.random.seed(42)
    r2 = rdp_mech(5)

    assert np.array_equal(r1, r2)


def test_rdp_mech_different_alpha():
    r_small = rdp_mech(2)
    r_large = rdp_mech(10)

    assert len(r_small) == len(r_large)


def test_convert_RDP_ED_large_alpha():
    alpha = 100
    epsilon_bar = 0.001
    delta = 1e-5

    epsilon, _ = convert_RDP_ED(alpha, epsilon_bar, delta)

    assert epsilon > epsilon_bar  # must increase


def test_convert_RDP_ED_small_delta():
    alpha = 5
    epsilon_bar = 0.001
    delta = 1e-10

    epsilon, _ = convert_RDP_ED(alpha, epsilon_bar, delta)

    assert epsilon > 0

# Tests T4
def test_encode_response_sales_no_flip():
    # alpha = 0 to prevent flipping bits
    alpha = 0
    one_hot = encode_response_sales(12, alpha)
    compare_one_hot = [0] * 44
    compare_one_hot[12] = 1
    assert ((one_hot == pd.Series(compare_one_hot)).all())


def test_encode_response_sales_flip():
    # alpha = 1 force bit flip
    alpha = 1
    one_hot = encode_response_sales(12, alpha)
    compare_one_hot = [1] * 44
    compare_one_hot[12] = 0
    assert ((one_hot == pd.Series(compare_one_hot)).all())


def test_encode_response_sales_nan():
    # check if nan is covered
    alpha = 0
    one_hot = encode_response_sales(np.nan, alpha)
    compare_one_hot = [0] * 44
    compare_one_hot[0] = 1
    assert ((one_hot == pd.Series(compare_one_hot)).all())


def test_decode_responses_sales():
    alpha = 0
    occ_data = pd.DataFrame({'NOC_43': [12, 12, 12, 1, 1, np.nan]})['NOC_43']
    encoded = occ_data.apply(encode_response_sales, alpha=alpha)
    decoded = decode_responses_sales(encoded, alpha)
    assert (decoded[0] == 1)
    assert (decoded[1] == 2)
    assert (decoded[12] == 3)
    assert (decoded[2] == 0)
    assert (decoded[43] == 0)
