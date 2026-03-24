#%%
import pandas as pd
import numpy as np
from scipy import stats
import pytest

ADULT_CSV_PATH = 'adult_with_pii.csv'

# my functions
def load_data():
    return pd.read_csv(ADULT_CSV_PATH)


def get_adult_small():
    return load_data().head(100)[['Education', 'Marital Status', 'Target']]


def get_homogeneity_attack(data):
    return print(f'Rows that can be attacked via homogeneity attack: {data.duplicated().sum()}')


def generalize_full_adult(n_zip, n_age, sup_sex=False):
    data = load_data()
    data = data.drop(columns=['Name', 'DOB', 'SSN'])
    data['Zip'] = data['Zip'].apply(lambda zip : generalize_numeric(zip, n_zip))
    data['Age'] = data['Age'].apply(lambda age : generalize_numeric(age, n_age))
    if(sup_sex):
        data['Age'] = '*'
    return data

def generalize_full_adult_task7():
    data = load_data()[['Education', 'Education-Num', 'Marital Status', 'Target']]
    hs_num = data[data['Education'] == 'HS-grad']['Education-Num'].iloc[0]
    data.loc[data['Education-Num'] < 9, 'Education'] = '<HS'
    data.loc[data['Education-Num'] >= 9, 'Education'] = '>=HS'
    data = data.drop(columns=['Education-Num'])
    data.loc[data['Marital Status'].isin(['Never-married', 'Divorced', 'Separated' ]), 'Marital Status'] = 'Not Married'
    data.loc[~data['Marital Status'].isin(['Not Married', 'Never-married', 'Divorced', 'Separated' ]), 'Marital Status'] = 'Married'
    return data

def print_info(k, n_zip, n_age, sup_sex,  qis, data):
    print(f'k: {k}, n_zip: {n_zip}, n_age: {n_age}, sup_sex: {sup_sex} - Entries to suppress: {suppress_count(k, qis, data)}/{len(data)}')


# predefined functions
def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers 
    qis. Returns false otherwise."""
    if(len(df[qis]) == 0):
        return True
    return df[qis].value_counts().min() >= k

def test_is_k_anonymous():
    qis = ['Education', 'Marital Status']
    data = {
        'Education': ['A', 'A', 'B'],
        'Marital Status': ['M', 'M', 'NM'],
        'Target': ['1', '1', '1']
    }
    data = pd.DataFrame(data)
    np.testing.assert_array_equal(is_k_anonymous(1, qis, data), True)
    np.testing.assert_array_equal(is_k_anonymous(2, qis, data), False)
    data = {
        'Education': ['A', 'A', 'B', 'B'],
        'Marital Status': ['M', 'M', 'NM', 'NM'],
        'Target': ['1', '1', '1', '1']
    }
    data = pd.DataFrame(data)
    np.testing.assert_array_equal(is_k_anonymous(2, qis, data), True)
    np.testing.assert_array_equal(is_k_anonymous(3, qis, data), False)

def test_k_anonymous_emptydf():
    adult = pd.read_csv(ADULT_CSV_PATH)
    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:0]
    qis = ['Education', 'Marital Status']

    assert is_k_anonymous(1, qis, adult_small) == True

def test_k_anonymous_onerowdf():
    adult = pd.read_csv(ADULT_CSV_PATH)
    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:1]
    qis = ['Education', 'Marital Status']
    
    assert is_k_anonymous(1, qis, adult_small) == True
    assert is_k_anonymous(2, qis, adult_small) == False

def generalize_categorical():
    data = load_data().head(100)[['Education', 'Education-Num', 'Marital Status', 'Target']]
    hs_num = data[data['Education'] == 'HS-grad']['Education-Num'].iloc[0]
    data.loc[data['Education-Num'] < 9, 'Education'] = '<HS'
    data.loc[data['Education-Num'] >= 9, 'Education'] = '>=HS'
    data = data.drop(columns=['Education-Num'])
    data.loc[data['Marital Status'].isin(['Never-married', 'Divorced', 'Separated' ]), 'Marital Status'] = 'Not Married'
    data.loc[~data['Marital Status'].isin(['Not Married', 'Never-married', 'Divorced', 'Separated' ]), 'Marital Status'] = 'Married'
    return data

def test_load_dataset():
    # Execute the function to trigger internal loading
    df = generalize_categorical()

    # Verify it returned a DataFrame
    assert isinstance(df, pd.DataFrame), "Result is not a pandas DataFrame"

    # Verify columns were correctly selected internally
    expected_cols = ['Education', 'Marital Status', 'Target']
    assert all(col in df.columns for col in expected_cols), "Column selection failed"

    # Verify the internal slice [0:100] was applied
    assert len(df) == 100, f"Expected 100 rows, but got {len(df)}"

    # Verify data integrity (checking if 'Target' exists and is populated)
    assert df['Target'].notnull().any(), "Data values are missing or null"

def test_generalize_categorical_emptydf():
    adult = pd.read_csv(ADULT_CSV_PATH)

    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:0]
    qis = ['Education', 'Marital Status']

    # check 1-anonymity which should be true
    if is_k_anonymous(1, qis, adult_small):
        flag = 'Dataset remains unchanged! I did nothing.' # save flag to assert later
    else:
        assert 1==2 # should the if clause fail the test must fail

    assert flag == 'Dataset remains unchanged! I did nothing.'

def test_generalize_categorical_achieves_anonymity():
    # 1. Run the actual function on the actual adult.csv
    result = generalize_categorical()

    # 2. Check: Did it actually achieve 2-anonymity?
    qis = ['Education', 'Marital Status']
    is_safe = is_k_anonymous(2, qis, result)
    
    assert is_safe == True, "The function failed to achieve 2-anonymity on the real data!"

def test_generalize_categorical_correct_column_entries():
    result = generalize_categorical()
    allowed_edu = {'<HS', '>=HS'}
    actual_edu = set(result['Education'].unique())
    assert actual_edu.issubset(allowed_edu), f"Found unexpected Education values: {actual_edu}"
    allowed_marital = {'Married', 'Not Married'}
    actual_marital = set(result['Marital Status'].unique())
    assert actual_marital.issubset(allowed_marital), f"Found unexpected Marital Status values: {actual_marital}"

def test_generalize_categorical_nothing_not_enough():
    adult = pd.read_csv(ADULT_CSV_PATH)

    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:100]
    qis = ['Education', 'Marital Status']

    assert (~ is_k_anonymous(2, qis, adult_small))

def test_generalize_categorical_education_not_enough():
    adult = pd.read_csv(ADULT_CSV_PATH)

    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:100]
    qis = ['Education', 'Marital Status']

    #generalizes Education status
    no_HSdegree = ['11th', '10th', '7th-8th', '9th', '12th', '5th-6th', '1st-4th', 'Preschool']
    adult_small.loc[ adult_small['Education'].isin(no_HSdegree), 'Education'] = '< HS'
    adult_small.loc[ ~adult_small['Education'].isin(['< HS']), 'Education'] = '>= HS'

    assert (~ is_k_anonymous(2, qis, adult_small))

def test_generalize_categorical_education_and_marital_enough():
    adult = pd.read_csv(ADULT_CSV_PATH)

    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:100]
    qis = ['Education', 'Marital Status']
    no_HSdegree = ['11th', '10th', '7th-8th', '9th', '12th', '5th-6th', '1st-4th', 'Preschool']
    adult_small.loc[ adult_small['Education'].isin(no_HSdegree), 'Education'] = '< HS'
    adult_small.loc[ ~adult_small['Education'].isin(['< HS']), 'Education'] = '>= HS'

    not_married = ['Never-married', 'Divorced', 'Widowed']
    adult_small.loc[ adult_small['Marital Status'].isin(not_married), 'Marital Status'] = 'Not Married'
    adult_small.loc[ ~adult_small['Marital Status'].isin(['Not Married']), 'Marital Status'] = 'Married'

    assert (is_k_anonymous(2, qis, adult_small))

def generalize_numeric(zip, n):
    if(n == 0):
        return zip
    zip = str(zip)
    return int(zip[:-n] + ('0' * min(n, len(zip))))


def test_generalize_numeric():
    assert generalize_numeric(47401, 0) == 47401
    assert generalize_numeric(47401, 2) == 47400
    assert generalize_numeric(47401, 4) == 40000
    assert generalize_numeric(47401, 10) == 00000


def suppress_count(k, qis, df):
    return (df[qis].value_counts() < k).sum()

def test_suppress_count_no_suppression():
    df = pd.DataFrame({
        "Zip": [1, 1, 2, 2],
        "Sex": ["M", "M", "F", "F"],
        "Age": [30, 30, 40, 40]
    })

    assert suppress_count(2, ["Zip", "Sex", "Age"], df) == 0


def test_suppress_count_some_suppression():
    df = pd.DataFrame({
        "Zip": [1, 1, 1, 2],
        "Sex": ["M", "M", "M", "F"],
        "Age": [30, 30, 30, 40]
    })

    # group (1,M,30) size 3
    # group (2,F,40) size 1 -> suppressed
    assert suppress_count(2, ["Zip", "Sex", "Age"], df) == 1


def test_suppress_count_multiple_small_groups():
    df = pd.DataFrame({
        "Zip": [1, 2, 3],
        "Sex": ["M", "M", "M"],
        "Age": [30, 30, 30]
    })

    # three groups of size 1
    assert suppress_count(2, ["Zip", "Sex", "Age"], df) == 3


def test_suppress_count_exactly_k():
    df = pd.DataFrame({
        "Zip": [1, 1],
        "Sex": ["M", "M"],
        "Age": [30, 30]
    })

    # size == k should NOT be suppressed
    assert suppress_count(2, ["Zip", "Sex", "Age"], df) == 0


def test_suppress_count_all_suppressed():
    df = pd.DataFrame({
        "Zip": [1, 2],
        "Sex": ["M", "F"],
        "Age": [30, 40]
    })

    # two groups of size 1
    assert suppress_count(2, ["Zip", "Sex", "Age"], df) == 2

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

def test_is_l_diverse():
    # data from lecture slides
    data = {
        'qis': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        'sens_col': ['HD', 'VI', 'C', 'C', 'C', 'HD', 'VI', 'VI', 'HD', 'VI', 'C', 'C']
    }
    data = pd.DataFrame(data)

    # probabilistic version
    assert is_l_diverse(2, ['qis'], 'sens_col', data)
    assert not is_l_diverse(2.1, ['qis'], 'sens_col', data)

    # entropy version
    assert is_l_diverse(2.8, ['qis'], 'sens_col', data, 'entropy')
    assert not is_l_diverse(3, ['qis'], 'sens_col', data, 'entropy')
    # test invalid type
    with pytest.raises(ValueError):
        is_l_diverse(2, ['qis'], 'sens_col', data, 'invalid')
    # test invalid l
    with pytest.raises(ValueError):
        is_l_diverse(0, ['qis'], 'sens_col', data)

def max_l(qis, sens_col, df, type='probabilistic'):
    if type not in {'probabilistic', 'entropy'}:
        raise ValueError("type must be 'probabilistic' or 'entropy'")
    group_probability = df.groupby(qis)[sens_col].value_counts(normalize=True)
    match type:
        case 'probabilistic':
            return 1/group_probability.max()
        case 'entropy':
            entropy = -(group_probability * np.log2(group_probability))
            entropy = entropy.groupby(qis).sum()
            return np.pow(2, entropy.min())

def test_max_l():
    # data from lecture slides
    data = {
        'qis': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        'sens_col': ['HD', 'VI', 'C', 'C', 'C', 'HD', 'VI', 'VI', 'HD', 'VI', 'C', 'C']
    }
    data = pd.DataFrame(data)
    l = max_l(['qis'], 'sens_col', data)
    assert pytest.approx(2, 0.05) == l
    l = max_l(['qis'], 'sens_col', data, 'entropy')
    assert pytest.approx(2.8, 0.05) == l
    # 1, 1
    data = {
        'qis': ['A', 'A'],
        'sens_col' : ['A', 'B']
    }
    data = pd.DataFrame(data)
    l = max_l(['qis'], 'sens_col', data)
    assert pytest.approx(2, 0.1) == l
    l = max_l(['qis'], 'sens_col', data, 'entropy')
    assert pytest.approx(2, 0.1) == l
    # 3, 2, 1
    data = {
        'qis': ['A', 'A', 'A', 'A', 'A', 'A'],
        'sens_col' : ['A', 'A', 'A', 'A', 'B', 'C']
    }
    data = pd.DataFrame(data)
    l = max_l(['qis'], 'sens_col', data)
    assert pytest.approx(1.5, 0.1) == l
    l = max_l(['qis'], 'sens_col', data, 'entropy')
    assert pytest.approx(2.4, 0.1) == l



# Tasks
def task1():
    print('-' * 6, 'Task 1', '-' * 6)
    data = get_adult_small()
    qis = ['Education', 'Marital Status']
    for k in range(1, 11):
        print(f'adult_small {k}-anonymous: {is_k_anonymous(k, qis, data)}')

    group_counts = data[qis].value_counts()
    print(group_counts)
    print()


def task2():
    print('-' * 6, 'Task 2', '-' * 6)
    data = generalize_categorical()
    print('Check for Homogenity attack:')
    print(data.groupby(['Education', 'Marital Status']).value_counts())
    print(data.groupby(['Education', 'Marital Status'])['Target'].nunique())
    print()


def task4():
    print('-' * 6, 'Task 4', '-' * 6)
    n_zip = 2
    n_age = 1
    sup_sex = False
    k = 3
    qis = ['Zip', 'Sex', 'Age']

    print_info(k, n_zip, n_age, sup_sex, qis, generalize_full_adult(n_zip, n_age, sup_sex))
    n_zip = 3
    print_info(k, n_zip, n_age, sup_sex, qis, generalize_full_adult(n_zip, n_age, sup_sex))
    
    k = 7
    n_zip = 2
    print_info(k, n_zip, n_age, sup_sex, qis, generalize_full_adult(n_zip, n_age, sup_sex))
    n_zip = 3
    print_info(k, n_zip, n_age, sup_sex, qis, generalize_full_adult(n_zip, n_age, sup_sex))
    print()


def task6():
    print('-' * 6, 'Task 6', '-' * 6)
    data = load_data()[['Education', 'Marital Status', 'Sex', 'DOB']]
    l = 2
    qis = ['Education', 'Marital Status', 'Sex']
    sens_col = 'DOB'
    print(f'Is {2}-div prob: {is_l_diverse(l, qis, sens_col, data)}')
    print(f'Is {2}-div entr: {is_l_diverse(l, qis, sens_col, data, 'entropy')}')

    qis = ['Education', 'Sex']
    print(f'With qis = {qis}')
    print(f'Is {2}-div prob: {is_l_diverse(l, qis, sens_col, data)}')
    print(f'Is {2}-div entr: {is_l_diverse(l, qis, sens_col, data, 'entropy')}')

    qis = ['Marital Status', 'Sex']
    print(f'With qis = {qis}')
    print(f'Is {2}-div prob: {is_l_diverse(l, qis, sens_col, data)}')
    print(f'Is {2}-div entr: {is_l_diverse(l, qis, sens_col, data, 'entropy')}')

    qis = ['Education', 'Sex']
    print(f'With qis = {qis}')
    print(f'Is {2}-div prob: {is_l_diverse(l, qis, sens_col, data)}')
    print(f'Is {2}-div entr: {is_l_diverse(l, qis, sens_col, data, 'entropy')}')
    print()


def task7():
    print('-' * 6, 'Task 7', '-' * 6)
    data = generalize_full_adult_task7()
    qis = ['Education', 'Marital Status']
    sens_col = 'Target'
    l = max_l(qis, sens_col, data)
    print(f"Max l with prob. {l}")
    l = max_l(qis, sens_col, data, 'entropy')
    print(f"Max l with entr. {l}")
    print(data.groupby(qis).value_counts())
    print()

# %%
if __name__ == "__main__":
    task1()
    task2()
    task4()
    task6()
    task7()

    
