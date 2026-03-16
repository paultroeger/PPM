# Load the data and libraries
import pandas as pd
import numpy as np
from scipy import stats
import pytest

education_map = {
    "11th": "Below HS",
    "9th": "Below HS",
    "7th-8th": "Below HS",
    "5th-6th": "Below HS",
    "10th": "Below HS",

    "Bachelors": "Above HS",
    "HS-grad": "Above HS",
    "Masters": "Above HS",
    "Some-college": "Above HS",
    "Doctorate": "Above HS",
    "Assoc-acdm": "Above HS",
    "Assoc-voc": "Above HS",
    "Prof-school": "Above HS"
}

marital_status_map = {
    "Married-civ-spouse": "Married",
    "Married-AF-spouse": "Married",
    "Married-spouse-absent": "Married",

    "Never-married": "Not Married",
    "Divorced": "Not Married",
    "Separated": "Not Married"
}


def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers
    qis. Returns false otherwise."""
    if df.empty: return False
    if df.groupby(qis).size().min() >= k:
        return True
    else:
        return False


def generalize_categorical(data):
    df = data.copy()

    df["Education"] = df["Education"].map(education_map)
    df["Marital Status"] = df["Marital Status"].map(marital_status_map)

    # Suppression commented out for task 7
    # group_sizes = data.groupby(['Education', 'Marital Status'])['Education'].transform('size')
    # df = data[group_sizes >= 2]

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
    if type not in {'probabilistic', 'entropy'}:
        raise ValueError("type must be 'probabilistic' or 'entropy'")
    if l <= 0:
        raise ValueError("l must be > 0")
    groups = df.groupby(qis)
    for _, group in groups:
        values = group[sens_col].value_counts(normalize=True)
        if type == 'probabilistic':
            if values.max() > (1 / l):
                return False
        elif type == 'entropy':
            ent = stats.entropy(values, base=2)
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
    # print(is_k_anonymous(1, ['Education', 'Marital Status', 'Target'], adult_small))
    # print(is_k_anonymous(2, ['Education', 'Marital Status', 'Target'], adult_small))
    # TASK 2
    adult_small_gen = generalize_categorical(adult_small)
    # print(adult_small_gen)
    # print(adult_small_gen.groupby(['Education', 'Marital Status', 'Target']).size())
    # TASK 3
    # See test_generalize_numeric
    # TASK 4
    # adult['Sex'] = "*"
    # adult['Age'] = adult['Age'].apply(lambda x: generalize_numeric(x, 1))
    # adult['Zip'] = adult['Zip'].apply(lambda x: generalize_numeric(x, 2))
    # print(suppress_count(3, ['Sex', 'Age', 'Zip'], adult))
    # TASK 5
    print(is_l_diverse(1, ['Education', 'Marital Status', 'Sex'], 'DOB', adult, 'entropy'))
    # TASK 7
    print(max_l(['Education', 'Marital Status', 'Sex'], 'DOB', adult))
    adult_gen = generalize_categorical(adult)
    print(max_l(['Education', 'Marital Status', 'Sex'], 'DOB', adult_gen, 'entropy'))


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

# Tests both probabilistic and entropy versions of is_l_diverse.
# Test invalid type parameter and invalid l parameter.
def test_is_l_diverse():
    data = {
        'qis': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        'sens_col': ['HD', 'VI', 'C', 'C', 'C', 'HD', 'VI', 'VI', 'HD', 'VI', 'C', 'C']
    }
    data = pd.DataFrame(data)

    assert is_l_diverse(2, ['qis'], 'sens_col', data)
    assert not is_l_diverse(2.1, ['qis'], 'sens_col', data)

    assert is_l_diverse(2.8, ['qis'], 'sens_col', data, 'entropy')
    assert not is_l_diverse(3, ['qis'], 'sens_col', data, 'entropy')
    with pytest.raises(ValueError):
        is_l_diverse(2, ['qis'], 'sens_col', data, 'invalid')
    with pytest.raises(ValueError):
        is_l_diverse(0, ['qis'], 'sens_col', data)
    with pytest.raises(ValueError):
        is_l_diverse(-1, ['qis'], 'sens_col', data)

