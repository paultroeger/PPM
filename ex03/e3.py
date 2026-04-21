# %% Imports and definitions
import numpy as np
from collections import defaultdict
import galois
import pytest

# %%

# arbitrary
MAX_INT = 100
GLOBAL_SEED = 1234
field=galois.GF(2 ** 13 - 1)

def shamir_share(x, t, n):
 
  np.random.seed(GLOBAL_SEED) 
  p = field(np.random.choice(np.arange(1, field.order), size=n, replace=False))
  
  random_poly = field.Random(t, low=1)
  
  poly = galois.Poly(np.append(random_poly, x), field=field)

  secrets = []
  for i in range(n):
      secrets.append([p[i], poly(p[i])])
  return field(secrets)


def add_shares(shares1, shares2):
   secrets = []
   for i in range(len(shares1)):
     if shares1[i][0] == shares2[i][0]:
       secrets.append([shares1[i][0], shares1[i][1] + shares2[i][1]])
     else:
       raise Exception("Not matching x")
   return field(secrets)


def add_const(shares, k):
    secrets = []
    for s in shares:
      secrets.append([s[0], s[1] + field(k)])
    return field(secrets)


def mult_const(shares, k):
    secrets = []
    for s in shares:
      secrets.append([s[0], s[1] * k])
    return field(secrets)


def reconstruct(shares):
    return galois.lagrange_poly(shares[:,0], shares[:,1])(0)


class Party:
    """A participant in a multiparty computation protocol."""
    def __init__(self):
        # TODO: your code here
        raise NotImplementedError()
    
    def send(self, other, round, msg):
        # TODO: your code here
        raise NotImplementedError()

    def get_view(self):
        # TODO: your code here
        raise NotImplementedError()


class BGW(Party):
    def round1(self, parties, a_shr, b_shr, t):
        self.input = (a_shr, b_shr)
        self.parties = parties
        n = len(parties)
        assert t <= n/2

        # TODO: your code here
        raise NotImplementedError()

    def round2(self):
        n = len(self.parties)

        # TODO: your code here
        raise NotImplementedError()


def run_bgw(t, n, a, b):
        # TODO: your code here
        raise NotImplementedError()

# Tasks

def task1():
  secret1 = shamir_share(field(12), 1, 4)
  secret2 = shamir_share(field(5), 1, 4)
  print(secret1)
  print(secret2)
  secrets1 = add_const(secret1, 2)
  reconstruction = reconstruct(secret1[0:2])
  print("re:", reconstruction)

  secret3 = add_shares(secret1, secret2)
  reconstruction = reconstruct(secret3[0:2])
  print("re:", reconstruction)

if __name__ == "__main__":
    #field = galois.GF(2 ** 13 - 1)

    task1()


# %%
