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



# asserts that loading the dataset works
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

#assert that True is returned on the empty dataframe
def test_k_anonymous_emptydf():
    adult = pd.read_csv(ADULT_CSV_PATH)
    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:0]
    qis = ['Education', 'Marital Status']

    assert is_k_anonymous(1, qis, adult_small) == True

#assert single entry dataframe is returned ruled as 1-anonymous
def test_k_anonymous_onerowdf():
    adult = pd.read_csv(ADULT_CSV_PATH)
    adult_small = adult[['Education', 'Marital Status', 'Target']]
    adult_small = adult_small[0:1]
    qis = ['Education', 'Marital Status']
    
    assert is_k_anonymous(1, qis, adult_small) == True
    assert is_k_anonymous(2, qis, adult_small) == False

# asserts that generalize_categorical actually terminates immediately on the empty dataset
# since I can't overwrite the dataset and am not allowed to change the funciton signature
# of generalize_categorical() I have to use code duplicates as workaround.
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

#asserts that it actually achieves the 2-anonymity for our adult_small dataset.
def test_generalize_categorical_achieves_anonymity():
    # 1. Run the actual function on the actual adult.csv
    result = generalize_categorical()

    # 2. Check: Did it actually achieve 2-anonymity?
    qis = ['Education', 'Marital Status']
    is_safe = is_k_anonymous(2, qis, result)
    
    assert is_safe == True, "The function failed to achieve 2-anonymity on the real data!"

#asserts that the categories "Marital Status" and "EducatioN" have the correct
#generalized entries and no entries were forgotten
def test_generalize_categorical_correct_column_entries():

    result = generalize_categorical()
    allowed_edu = {'< HS', '>= HS'}
    actual_edu = set(result['Education'].unique())
    assert actual_edu.issubset(allowed_edu), f"Found unexpected Education values: {actual_edu}"
    allowed_marital = {'Married', 'Not Married'}
    actual_marital = set(result['Marital Status'].unique())
    assert actual_marital.issubset(allowed_marital), f"Found unexpected Marital Status values: {actual_marital}"

'''
We know that our function generalize_categorical() generalizes Education and Marital Status in order to achieve 2-anonymity. We will now test if these generalizations were really necessary by asserting that the dataset was indeed not 2-anonymous after each step. Testing this is important because we want the generalization to be minimal in order to maximize utility (=data is more statistically meaningful to an analyst if it is less generalized)
'''

#asserts generalizing nothing is not enough to achieve 2-anonymity. 
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



def is_k_anonymous(k, qis, df):
    """Returns true if df satisfies k-Anonymity for the quasi-identifiers 
    qis. Returns false otherwise."""

    if (df.empty):
        return True
    else:
        # value_counts on a list of columns returns the size of every unique group
        group_counts = df[qis].value_counts()
        
        # Check if the smallest group is at least k
        return group_counts.min() >= k

def generalize_categorical():
    # I wasn't sure if we can assume this function is run after the main script. Just wanna make sure the adult data set is loaded correctly.
    # This is a "Safety Net" if the handin teams pytests only call my function
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
        not_married = ['Never-married', 'Divorced', 'Widowed']
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
    # Group by QIs and look at the sensitive column
    groups = df.groupby(qis)[sens_col]

    for name, group in groups:
        # value_counts(normalize=True) gives us the percentage of each value
        counts = group.value_counts(normalize=True)
        
        if type == 'probabilistic':
            # Check the highest probability (the most frequent value)
            p_max = counts.max()
            if p_max > (1/l):
                return False
                
        elif type == 'entropy':
            # Calculate entropy: -sum(p * log2(p))
            probs = counts.values
            entropy = -np.sum(probs * np.log2(probs))
            if entropy < np.log2(l):
                return False
                
    return True


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

    #Exercise 6



    








