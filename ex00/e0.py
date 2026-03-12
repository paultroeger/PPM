# Load needed libraries and the dataset
import pandas as pd
import matplotlib.pyplot as plt
import pytest


def load_adult():
    return pd.read_csv('adult_with_pii.csv')


def print_stats(data):
    print("--- Columns ---")
    print(data.columns, '\n')

    print("--- Head ---")
    pd.set_option('display.max_columns', None)
    print(data.head(), '\n')
    pd.reset_option('display.max_columns')
    
    print("--- Row Size ---")
    print(data.size, '\n')

    print("--- Missing Values ---")
    has_missing = data.isna().values.any()
    print(f"Contains missing values: {has_missing}")
    print(data.isna().sum(), '\n')


def deidentify_dataset(data):
    return data.drop(columns=['Name', 'SSN'])


def get_rosys_row(data):
    return data[data['Name'].str.contains('Rosy')].iloc[0]


def test_get_rosys_row():
    # TODO: your test cases here
    raise NotImplementedError()


def recover_rosys_row(data, data_deid):
    return data_deid[(data_deid['Age'] == data['Age']) & (data_deid['Zip'] == data['Zip'])].iloc[0]


def test_recover_rosys_row():
    # TODO: your test cases here
    raise NotImplementedError()


def group_one_count(data, col):
    return data.groupby(col).size().reset_index(name='count')


def test_group_one_count():
    # TODO: your test cases here
    raise NotImplementedError()


def group_one_age(data, col):
    return data.groupby(col)['Age'].mean().reset_index()


def group_two_count(data, col1, col2):
    return data.groupby([col1, col2]).size().reset_index(name='count')


def test_group_two_count():
    # TODO: your test cases here
    raise NotImplementedError()


def group_two_age(data, col1, col2):
    return data.groupby([col1, col2])['Age'].mean().reset_index()


def test_group_two_age():
    # TODO: your test cases here
    raise NotImplementedError()


def get_rosys_age(data):
    data_without_rosy = data[~data['Name'].str.contains('Rosy')]
    return data['Age'].sum() - data_without_rosy['Age'].sum()
    

def get_rosys_age_noname(data):
    data_without_rosy = data[~(data['Zip'] == 75436)]
    return (data['Age'].sum() - data_without_rosy['Age'].sum())
    

def get_rosys_age_mean(data):
    data_without_rosy = data[~data['Name'].str.contains('Rosy')]
    return data['Age'].mean() * len(data) - data_without_rosy['Age'].mean() * len(data_without_rosy)


def test_rosys_age():
    # TODO: your test cases here
    raise NotImplementedError()

def hist_unique_entries(data):
    print(f"Zip Uniqueness: {len(data['Zip'].unique())}")
    print(f"Age Uniqueness: {len(data['Age'].unique())}")
    print(f"DOB Uniqueness: {len(data['DOB'].unique())}")
    print(f"\t\t/{len(data)}")

    # histogram
    zip_age_counts = data.groupby(['Zip', 'Age']).size()
    zip_dob_counts = data.groupby(['Zip', 'DOB']).size()

    plt.figure(figsize=(6, 6))

    plt.hist([zip_age_counts, zip_dob_counts], bins=2, label=['Zip + Age', 'Zip + DOB'])

    plt.title('Uniqueness Comparison: (Zip, Age) vs (Zip, DOB)')
    plt.xlabel('Group Size')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    max_val = max(zip_age_counts.max(), zip_dob_counts.max())
    plt.xticks(range(1, max_val + 1))
    plt.show()

def hist_link_attack(data):
    data_deid = deidentify_dataset(data)
    
    data_dob = pd.merge(data[['Name', 'DOB']], data_deid, on='DOB', how='left')
    counts = data_dob['Name'].value_counts()
    unique_names = counts[counts == 1].index.tolist()
    data_dob_identified_n = len(data_dob[data_dob['Name'].isin(unique_names)])

    data_dob_zip = pd.merge(data[['Name', 'DOB', 'Zip']], data_deid, on=['DOB', 'Zip'], how='left')
    counts = data_dob_zip['Name'].value_counts()
    unique_names = counts[counts == 1].index.tolist()
    data_dob_zip_identified_n = len(data_dob_zip[data_dob_zip['Name'].isin(unique_names)])
    
    print(f"Identified by DOB: {data_dob_identified_n}")
    print(f"Identified by DOB and Zip: {data_dob_zip_identified_n}")
    print(f"Dataset size: {len(data)}")

def counts(data):
    print("--- Occupation ---")
    print(pd.merge(group_one_count(data, 'Occupation'), group_one_age(data, 'Occupation'), on='Occupation'), '\n')
    
    print("--- Education ---")
    print(pd.merge(group_one_count(data, 'Education'), group_one_age(data, 'Education'), on='Education'), '\n')

    print("--- Occupation & Education---")
    print(pd.merge(group_two_count(data, 'Occupation', 'Education'), group_two_age(data, 'Occupation', 'Education'), on=['Occupation', 'Education']), '\n')

if __name__ == "__main__":
    data = load_adult()
    print(get_rosys_age(data))
    print(get_rosys_age_noname(data))
    print(get_rosys_age_mean(data))