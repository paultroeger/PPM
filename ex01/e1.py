# Load the data and libraries
import pandas as pd
import numpy as np
import scipy.stats
import pytest


def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers 
    qis. Returns false otherwise."""
    # value_counts on a list of columns returns the size of every unique group
    group_counts = df[qis].value_counts()
    
    # Check if the smallest group is at least k
    return group_counts.min() >= k

def generalize_categorical():
    # TODO: your code here
    raise NotImplementedError()


def generalize_numeric(zip, n):
    # TODO: your code here
    raise NotImplementedError()


def suppress_count(k, qis, df):
    # TODO: your code here
    raise NotImplementedError()


def is_l_diverse(l, qis, sens_col, df, type='probabilistic'):
    # The type parameter has two valid values: 'probabilistic' and 'entropy'
    # TODO: your code here
    raise NotImplementedError()


def max_l(qis, sens_col, df, type='probabilistic'):
    # The type parameter has two valid values: 'probabilistic' and 'entropy'
    # TODO: your code here
    raise NotImplementedError()


if __name__ == "__main__":
    adult = pd.read_csv(r'C:\2.0\Privacy-Preserving Methods for Data Science and Distributed Systems\Privacy-Preserving-Methods-for-Data-Science-and-Distributed-Systems-\ex01\adult_with_pii.csv')

    #print(adult.columns)
    adult_small = adult[['Education','Marital Status', 'Target']]
    adult_small = adult_small[0:100]
    print(adult_small)


    #Exercise 1

    # there is no identifiers left in this table because we took all of them out (like Name, DOB and SSN).

    # quasi-identifiers: everything thats neither sensitive nor an identifier (so Education and Marital Status)
    # sensitive attributes: given the following exercises and re-identification attacks I'd say the only sensitive column is the Target column

    k_anonymity = np.zeros(10,dtype = bool)
    qis = ['Education','Marital Status']
    for k in range(1,10):
        if is_k_anonymous(k, qis, adult_small):
            k_anonymity[k-1] = True
        else: k_anonymity[k-1] = False

    print(k_anonymity) # -> from this it becomes clear that k = 2 is the problem

    # let's print grouped set to see what the reason is:

    group_counts = adult_small[qis].value_counts()
    print(group_counts)

    # which will show that 
    ''' 
    Education     Marital Status
    Assoc-voc     Never-married             1
    7th-8th       Married-spouse-absent     1
    5th-6th       Married-civ-spouse        1
    Prof-school   Married-civ-spouse        1
    Masters       Never-married             1
    9th           Married-spouse-absent     1
                  Never-married             1
    HS-grad       Married-AF-spouse         1
    Assoc-acdm    Divorced                  1
    Doctorate     Never-married             1
    Bachelors     Separated                 1
    10th          Married-civ-spouse        1
'''
    #is the reason.


    #Exercise 2

    group_counts = adult['Education'].value_counts()
    print(group_counts) # to see all possible Education attributes we print it and get the following variations of "no_HSdegreee"
    no_HSdegree = ['11th', '10th', '7th-8th', '9th', '12th', '5th-6th', '1st-4th', 'Preschool']

    












