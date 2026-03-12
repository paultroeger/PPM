# Load the data and libraries
import pandas as pd
import numpy as np
import scipy.stats
import pytest
ADULT_CSV_PATH = r'C:\2.0\Privacy-Preserving Methods for Data Science and Distributed Systems\Privacy-Preserving-Methods-for-Data-Science-and-Distributed-Systems-\ex01\adult_with_pii.csv'

def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers 
    qis. Returns false otherwise."""
    # value_counts on a list of columns returns the size of every unique group
    group_counts = df[qis].value_counts()
    
    # Check if the smallest group is at least k
    return group_counts.min() >= k

def generalize_categorical():
    # I wasn't sure if we can assume this function is run after the main script. Just wanna make sure the adult data set is loaded correctly.
    # This is the "Safety Net" for the grader's environment
    adult = pd.read_csv(ADULT_CSV_PATH)

    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:100]
    qis = ['Education', 'Marital Status']

    #check if the dataset already satisfies 2-anonymity. If yes, do nothing and return dataset.
    if is_k_anonymous(2, qis, adult_small):
        print('Dataset remains unchanged! I did nothing.')
        return adult_small
    
    #otherwise generalize "HS Grad"/ "No HS Grad"
    else:
        no_HSdegree = ['11th', '10th', '7th-8th', '9th', '12th', '5th-6th', '1st-4th', 'Preschool']
        adult_small.loc[ adult_small['Education'].isin(no_HSdegree), 'Education'] = '< HS'
        adult_small.loc[ ~adult_small['Education'].isin(['< HS']), 'Education'] = '>= HS'
    
    #check if the dataset already satisfies 2-anonymity. If yes, return dataset.
    if is_k_anonymous(2, qis, adult_small):
        print('Education was generalized! The rest is unchanged.')
        return adult_small
    
    #otherwise generalize "Married"/"Not Married"
    else:
        adult_small.loc[ adult_small['Marital Status'].isin(not_married), 'Marital Status'] = 'Not Married'
        adult_small.loc[ ~adult_small['Marital Status'].isin(['Not Married']), 'Marital Status'] = 'Married'

    #check if dataset already satisfies 2-anonymity. If yes, return dataset.
    if is_k_anonymous(2, qis, adult_small):
        print('Education and Marital Status was generalized! The rest is unchanged.')
        return adult_small
    
    #otherwise suppress the rows that break 2-anonymity
    else:
        #filters out groups that have fewer than 2 rows (i.e. deletes unique rows)
        adult_small = adult_small.groupby(qis).filter(lambda x: len(x) >= 2)
        print('Education and Marital Status was generalized! I also suppressed some rows.')
        return adult_small

    



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
    adult = pd.read_csv(ADULT_CSV_PATH)

    #print(adult.columns)
    adult_small = adult[['Education','Marital Status', 'Target']]
    adult_small = adult_small[0:100]
    print(adult_small)


    #Exercise 1

    '''
    there are no identifiers left in adult_small because we took all of them out (like Name, DOB and SSN).

    quasi-identifiers: everything thats neither sensitive nor an identifier is a quasi identifier (so Education and Marital Status)

    sensitive attributes: given the following exercises and re-identification attacks I'd say the only sensitive column is the Target column
    '''

    #checking for k-anonymity
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
    #is the reason we only have 1-anonymity.


    #Exercise 2

    # The Tests we need to know what to do in generalize_categorical()
    #-------------------------------------------Tests------------------------------------------------
    group_counts = adult['Education'].value_counts()
    print(group_counts) # We do this to see all possible Education attributes and get the following variations of "no_HSdegree":
    no_HSdegree = ['11th', '10th', '7th-8th', '9th', '12th', '5th-6th', '1st-4th', 'Preschool']

    group_counts = adult['Marital Status'].value_counts()
    print(group_counts) # we do the same thing for Marital Status and get:
    not_married = ['Never-married', 'Divorced', 'Widowed']
    #------------------------------------------- End -----------------------------------------------
    
    adult_small_modified = generalize_categorical()
    print(adult_small_modified[['Marital Status', 'Education']].value_counts())
    # which will give
    '''
    Marital Status  Education
    Married         >= HS        50
    Not Married     >= HS        36
    Married         < HS         11
    Not Married     < HS          3
    '''
    # so we achieved 3-anonymity


    # Group by qi and count the number of unique (.nunique()) rows in each "bucket" at the 'Target' column.
    homogeneity_check = adult_small_modified.groupby(qis)['Target'].nunique()
    print(homogeneity_check)
    #which will give us this output
    ''' 
    Marital Status  Education    nunique
    Married         < HS         1
                    >= HS        2
    Not Married     < HS         1
                    >= HS        2
    '''
    # thus homogeinity attacks are possible against the first and third row (Married         < HS         1 and Not Married     < HS         1) because these combinations of Education and Marital Status only have Target variant (either >50k or <=50k) in their "bucket".










