# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import pytest


def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers
    qis. Returns false otherwise."""
    if df.empty: return False
    if df.groupby(qis).size().min() >= k:
        return True
    else:
        return False


def generalize_categorical(data):
    data = data.copy()
    below_hs = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']
    hs_and_above = ['HS-grad', 'Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors', 'Prof-school', 'Masters',
                    'Doctorate']
    married = ['Married-civ-spouse', 'Married-AF-spouse', 'Married-spouse-absent']
    not_married = ['Never-married', 'Separated', 'Divorced', 'Widowed']
    for index, row in data.iterrows():
        if row['Education'] in below_hs:
            data.at[index, 'Education'] = '< HS'
        elif row['Education'] in hs_and_above:
            data.at[index, 'Education'] = '>= HS'
        if row['Marital Status'] in married:
            data.at[index, 'Marital Status'] = 'Married'
        elif row['Marital Status'] in not_married:
            data.at[index, 'Marital Status'] = 'Not Married'

    group_sizes = data.groupby(['Education', 'Marital Status'])['Education'].transform('size')
    df = data[group_sizes >= 2]

    return df


def generalize_numeric(zip, n):
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return zip
    return zip - int(str(zip)[-n:])


def suppress_count(k, qis, df):
    group_sizes = df.groupby(qis)[qis[0]].transform('size')
    return (group_sizes < k).sum()


def is_l_diverse(l, qis, sens_col, df, type='probabilistic'):
    # The type parameter has two valid values: 'probabilistic' and 'entropy'
    groups = df.groupby(qis)
    for _, group in groups:
        values = group[sens_col]
        if type == 'probabilistic':
            probs = values.value_counts(normalize=True)
            if probs.max() > (1 / l):
                return False
        elif type == 'entropy':
            probs = values.value_counts(normalize=True)
            ent = stats.entropy(probs, base=2)
            if ent < np.log2(l):
                return False
    return True


def max_l(qis, sens_col, df, type='probabilistic'):
    # The variant parameter has two valid values: 'probabilistic' and 'entropy'
    l = 1
    while is_l_diverse(l, qis, sens_col, df, type):
        l += 1
    return l - 1


if __name__ == "__main__":
    # GENERAL
    adult = pd.read_csv('adult_with_pii.csv')
    adult_small = adult[['Education', 'Marital Status', 'Target']][:100]
    # TASK 1
    print(is_k_anonymous(1, ['Education', 'Marital Status', 'Target'], adult_small))
    print(is_k_anonymous(2, ['Education', 'Marital Status', 'Target'], adult_small))
    # TASK 2
    adult_small_gen = generalize_categorical(adult_small)
    print(adult_small_gen)
    print(adult_small_gen.groupby(['Education', 'Marital Status', 'Target']).size())
    # TASK 3
    # See test_generalize_numeric
    # TASK 4
    adult['Sex'] = "*"
    adult['Age'] = adult['Age'].apply(lambda x: generalize_numeric(x, 1))
    adult['Zip'] = adult['Zip'].apply(lambda x: generalize_numeric(x, 2))
    print(suppress_count(3, ['Sex', 'Age', 'Zip'], adult))
    # TASK 5
    print(is_l_diverse(1, ['Education', 'Marital Status', 'Sex'], 'DOB', adult, 'entropy'))
    # TASK 7
    print(max_l(['Education', 'Marital Status', 'Sex'], 'DOB', adult))
    adult_gen = generalize_categorical(adult)
    print(max_l(['Education', 'Marital Status', 'Sex'], 'DOB', adult_gen))


def test_is_k_anonymous():
    df = pd.DataFrame({
        "Education": ["Bachelors", "Bachelors", "HS-grad", "HS-grad"],
        "Marital Status": ["Single", "Single", "Married", "Married"]
    })
    assert is_k_anonymous(2, ["Education", "Marital Status"], df)
    assert not is_k_anonymous(3, ["Education", "Marital Status"], df)
    assert not is_k_anonymous(1, ["Education", "Marital Status"], df.iloc[0:0])


def test_generalize_numeric():
    assert generalize_numeric(47401, 0) == 47401
    assert generalize_numeric(47401, 2) == 47400
    assert generalize_numeric(47401, 4) == 40000
    with pytest.raises(ValueError):
        generalize_numeric(47401, -1)


def test_generalize_categorical_and_suppression():
    df = pd.DataFrame({
        "Education": ["Bachelors", "HS-grad", "Preschool"],
        "Marital Status": ["Never-married", "Divorced", "Married-civ-spouse"],
        "Target": ["<=50K", ">50K", "<=50K"]
    })
    out = generalize_categorical(df)
    assert len(out) == 2
    assert set(out["Education"]) == {">= HS"}
    assert set(out["Marital Status"]) == {"Not Married"}


def test_suppress_count():
    df = pd.DataFrame({
        "Sex": ["*", "*", "*", "*"],
        "Age": [20, 20, 30, 40],
        "Zip": [10000, 10000, 20000, 30000]
    })
    assert suppress_count(2, ["Sex", "Age", "Zip"], df) == 2
    assert suppress_count(3, ["Sex", "Age", "Zip"], df) == 4
