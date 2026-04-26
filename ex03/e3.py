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
        self.id = 0
        self.input = []
        self.shares = []
        self.output = 0
    
    def send(self, other, round, msg):
        match round:
           case 1:
              self.shares[other] = msg

    def get_view(self):
        return self.output


class BGW(Party):
    def round1(self, parties, a_shr, b_shr, t):
        self.input = (a_shr, b_shr)
        self.parties = parties
        n = len(parties)
        # see page 44 in the book
        assert 2*t+1 <= n

        m_share = self.input[0][1] * self.input[1][1]
        shares = shamir_share(m_share, t, n)
       
        for i in range(n):
          self.parties[i].send(self.id, 1, shares[i])

    def round2(self):
        n = len(self.parties)

        np.random.seed(GLOBAL_SEED) 
        p = field(np.random.choice(np.arange(1, field.order), size=n, replace=False))

        # page 43 from book coeff 0
        coeffs = get_lagrange_coeffs(p)
        y = field(0)
        for i in range(n):
          y += coeffs[i](0) * self.shares[i][1]
        self.output = [self.shares[0][0], y]


def run_bgw(t, n, a, b):
    secret1 = shamir_share(field(a), t, n)
    secret2 = shamir_share(field(b), t, n)

    # setup parties
    parties = [BGW() for _ in range(n)]
    for i in range(n):
        parties[i].id = i
        parties[i].shares = [0] * n
    # round 1
    for i in range(n):
      parties[i].round1(parties, secret1[i], secret2[i], t)
    # round 2
    for i in range(n):
      parties[i].round2()
    # collect outputs and reconstruct
    outputs = []
    for i in range(n):
      outputs.append(parties[i].get_view())
    outputs = field(outputs)
    print(reconstruct(outputs))


def get_lagrange_coeffs(x_values):
    n = len(x_values)
    p = []

    for i in range(n):
        y_values = field.Zeros(n)
        y_values[i] = 1
     
        p.append(galois.lagrange_poly(x_values, y_values))
        
    return p

# Tasks

def task1():
    print('-' * 6, 'Task 1', '-' * 6)
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
    print()

def task2():
    print('-' * 6, 'Task 2', '-' * 6)
    t, n, a, b = 2, 5, 5, 5
    print(f"t:{t}, n:{n}, a:{a}, b:{b}")
    run_bgw(t, n, a, b)
    print()

if __name__ == "__main__":
    #field = galois.GF(2 ** 13 - 1)
    task2()


# %%
