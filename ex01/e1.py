import pandas as pd
import numpy as np
import scipy.stats
import pytest

# my functions
def load_data():
    return pd.read_csv('adult_with_pii.csv')


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
    print(f'k: {k}, n_zip: {n_zip}, n_age: {n_age}, sup_sex: {sup_sex} - Entries to supress: {suppress_count(k, qis, data)}/{len(data)}')


# predefined functions
def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers 
    qis. Returns false otherwise."""
    return df[qis].value_counts().min() >= k

def test_is_k_anonymous():
    data = {
        'Education': ['A', 'A', 'B'],
        'Marital Status': ['M', 'M', 'NM'],
        'Target': ['1', '1', '1']
    }
    data = pd.DataFrame(data)
    np.testing.assert_array_equal(is_k_anonymous(1, ['Education', 'Marital Status'], data), True)


def generalize_categorical():
    data = load_data().head(100)[['Education', 'Education-Num', 'Marital Status', 'Target']]
    hs_num = data[data['Education'] == 'HS-grad']['Education-Num'].iloc[0]
    data.loc[data['Education-Num'] < 9, 'Education'] = '<HS'
    data.loc[data['Education-Num'] >= 9, 'Education'] = '>=HS'
    data = data.drop(columns=['Education-Num'])
    data.loc[data['Marital Status'].isin(['Never-married', 'Divorced', 'Separated' ]), 'Marital Status'] = 'Not Married'
    data.loc[~data['Marital Status'].isin(['Not Married', 'Never-married', 'Divorced', 'Separated' ]), 'Marital Status'] = 'Married'
    return data


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


def is_l_diverse(l, qis, sens_col, df, type='probabilistic'):
    # The type parameter has two valid values: 'probabilistic' and 'entropy'
    group_probability = df.groupby(qis)[sens_col].value_counts(normalize=True)
    match type:
        case 'probabilistic':
            return group_probability.max() <= 1/l
        case 'entropy':
            entropy = -(group_probability * np.log2(group_probability))
            entropy = entropy.groupby(qis).sum()
            return entropy.min() >= np.log2(l)
        case _:
            raise Error()

def test_is_l_diverse():
    # data from lecture slides
    data = {
        'qis': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        'sens_col': ['HD', 'VI', 'C', 'C', 'C', 'HD', 'VI', 'VI', 'HD', 'VI', 'C', 'C']
    }
    data = pd.DataFrame(data)
    assert is_l_diverse(2, ['qis'], 'sens_col', data)
    assert ~is_l_diverse(2.1, ['qis'], 'sens_col', data)

    assert is_l_diverse(2.8, ['qis'], 'sens_col', data, 'entropy')
    assert ~is_l_diverse(3, ['qis'], 'sens_col', data, 'entropy')

def max_l(qis, sens_col, df, type='probabilistic'):
    l = 0.1
    while(is_l_diverse(l, qis, sens_col, df, type)):
        l += 0.1
    return l

def test_max_l():
    # data from lecture slides
    data = {
        'qis': ['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'C', 'C', 'C', 'C'],
        'sens_col': ['HD', 'VI', 'C', 'C', 'C', 'HD', 'VI', 'VI', 'HD', 'VI', 'C', 'C']
    }
    data = pd.DataFrame(data)
    l = max_l(['qis'], 'sens_col', data)
    assert pytest.approx(2, 0.1) == l
    l = max_l(['qis'], 'sens_col', data, 'entropy')
    assert pytest.approx(2.8, 0.1) == l
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
    for k in range(1, 11):
        print(f'adult_small {k}-anonymous: {is_k_anonymous(k, ['Education', 'Marital Status'], data)}')
    print()


def task2():
    print('-' * 6, 'Task 2', '-' * 6)
    data = generalize_categorical()
    print('Check for Homogenity attack:')
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
    sup_sex = True
    print_info(k, n_zip, n_age, sup_sex, qis, generalize_full_adult(n_zip, n_age, sup_sex))
    
    k = 7
    sup_sex = False
    print_info(k, n_zip, n_age, sup_sex, qis, generalize_full_adult(n_zip, n_age, sup_sex))
    sup_sex = True
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
    print()

if __name__ == "__main__":
    task1()
    task2()
    task4()
    task6()
    task7()

    
