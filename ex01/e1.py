# Load the data and libraries
import pandas as pd
import numpy as np
import scipy.stats
import pytest

# Reminder: 

# k-anonymity: how many ppl with certain qis (married + PhD) combinations are 
# in the dataset (sensitive attributes are ignored for this). 
# -> i.e. how many ppl will land in the married + PhD bucket (ignoring if they
#    are <50k or >50k)

# l-diversity: for a certain qis combination ("bucket") how many different 
# variations of sensitive attributes exist (e.g. 2 for >50k and <50k) 
# -> protects against homogeneity attack

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
    # This is the "Safety Net" if the pytests only call my function
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

    



def generalize_numeric(val, n):
    if n == 0:
            return val
    
    else:
        # 1. Convert the number to a string so we can "see/parse" the digits
        s = str(val)

        # 2. Slice: Keep everything up to the last 'n' digits
        prefix = s[:-n]
        # 3. Overwrite: Fill up the rest with '0' by repeating it 'n' times
        suffix = '0' * n

        # 4. Join them and cast back to an integer for the test cases
        return int(prefix + suffix)


#returns the number of rows you need to suppress to achieve a given k
def suppress_count(k, qis, df):
    #gives the number of people in each bucket
    group_counts=df[qis].value_counts()

    #filters for all the qis combinations (buckets) that violate k-anonymity,
    #listing the number of people that are in this bucket. 
    unsafe_groups = group_counts[group_counts < k] 

    #sums over all the people in the buckets and tells me how many id need to
    #suppress to become k anonymous 
    total_unsafe_rows = unsafe_groups.sum() 
    return total_unsafe_rows


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
    Education  Marital Status
    < HS       Married           1
               Not Married       1
    >= HS      Married           2
               Not Married       2
    '''
    # thus homogeneity attacks are possible against the rows("Married         < HS         1" and "Not Married     < HS         1") because these combination "buckets" of "Education" and "Marital Status" only have one Target variant (either >50k or <=50k) in their "bucket".


    #Exercise 3
    assert generalize_numeric(47401, 0) == 47401
    assert generalize_numeric(47401, 2) == 47400
    assert generalize_numeric(47401, 4) == 40000
    

    #Exercise 4
    adult_ex4 = adult[['Zip', 'Sex', 'Age','Target']].copy()
    
    gen_zip = 2 #how many digits of zip to generalize
    gen_age = 1 #how many digits of age to generalize
    adult_ex4['Zip'] = adult_ex4['Zip'].apply(lambda zip: generalize_numeric(zip, gen_zip))
    adult_ex4['Age'] = adult_ex4['Age'].apply(lambda age: generalize_numeric(age, gen_age))
    #print(adult_ex4['Sex'].value_counts()) # Sex cant really be generalized further

    qis_ex4 = ['Zip', 'Sex', 'Age']
    print("For k = 3 we need to suppress " + str(suppress_count(3, qis_ex4, adult_ex4)) + " rows." ) 
    print("For k = 7 we need to suppress " + str(suppress_count(7, qis_ex4, adult_ex4)) + " rows." ) 

    '''
    gen_zip = 2, gen_age = 1 
    For k = 3 we need to suppress 7245 rows.
    For k = 7 we need to suppress 24808 rows.

    gen_zip = 3, gen_age = 1 
    For k = 3 we need to suppress 255 rows.
    For k = 7 we need to suppress 1143 rows.

    gen_zip = 4, gen_age = 1 
    For k = 3 we need to suppress 20 rows.
    For k = 7 we need to suppress 79 rows.
    '''
    #So I would say generalizing gen_zip = 4 and gen_age = 1 makes sense to get the best
    #tradeoff between utility and security


    








