import pytest
import numpy as np
import pandas as pd
from e2 import (
    laplace_mech, 
    hrs_cdf, 
    hrs_cdf_dp_laplace, 
    hrs_cdf_dp_gauss,
    avg_wages,
    hrs_cdf_v2
)

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
    """v2 should only return buckets for physically possible hours (0-168)."""
    epsilon = 1.0
    res = hrs_cdf_v2(mock_lfs_data, epsilon)
    
    # If you implemented it to stop at 168 hours, length should be 169
    assert len(res) == 169