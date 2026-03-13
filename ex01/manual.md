# Task 1
## Which columns of adult_small are identifiers, which are quasi-identifiers, and which are sensitive attributes?
There are no identifiers, Quasi-Identifiers are Education and Marital Status and the Target is sensitive.
## For which k \in [1, 10] does adult_small satisfy k-anonymity? For the cases that do not satisfy k-anonymity, why not?
adult_small only satisfies k = 1 because there are multiple rows with one unique quasi-identifier. E.g. Doctorate, Never-married