# Imports and definitions
import numpy as np
from collections import defaultdict
import galois
import pytest


# x -> secret
# t -> degree of polynomial
# n -> number of parties
def shamir_share(x, t, n):
    field = type(x)  # find out field class (e.g., GF(2**13-1))
    elements = field.Random(t)  # generate random elements between 0 and 8191 (for given GF)
    poly = galois.Poly(np.concatenate([elements, [x]]), field)  # generate polynomial with secret at the end
    shares = [(field(i), poly(i)) for i in range(1, n + 1)]  # start at 1 to exclude secret
    return shares


def add_shares(shares1, shares2):
    # TODO: your code here
    raise NotImplementedError()


def add_const(shares, k):
    return [(share[0], share[1] + k) for share in shares]


def mult_const(shares, k):
    return [(share[0], share[1] * k) for share in shares]


def reconstruct(shares):
    field = type(shares[0][0])  # find out field class
    x = field([share[0] for share in shares])
    y = field([share[1] for share in shares])
    poly = galois.lagrange_poly(x, y)  # reconstruct polynomial
    return poly(0)


class Party:
    """A participant in a multiparty computation protocol."""

    def __init__(self):
        self.mailbox = defaultdict(list)

    def send(self, other, round, msg):
        other.mailbox[round].append(msg)

    def get_view(self):
        return self.mailbox


class BGW(Party):
    def round1(self, parties, a_shr, b_shr, t):
        self.input = (a_shr, b_shr)
        self.parties = parties
        n = len(parties)
        assert t <= n / 2

        h = a_shr[1] * b_shr[1]
        h_shares = shamir_share(h, t, n)
        for j in range(n):
            self.send(parties[j], 1, h_shares[j])

    def round2(self):
        n = len(self.parties)

        received = self.mailbox[1]
        field = type(received[0][1])
        points = field([i for i in range(1, n + 1)])
        values = field([r[1] for r in received])
        poly = galois.lagrange_poly(points, values)
        self.output = poly(0)


def run_bgw(t, n, a, b):
    field = type(a)
    parties = [BGW() for _ in range(n)]
    a_shares = shamir_share(a, t, n)
    b_shares = shamir_share(b, t, n)

    for i in range(n):
        parties[i].round1(parties, a_shares[i], b_shares[i], t)

    for i in range(n):
        parties[i].round2()

    result_shares = [(field(i + 1), parties[i].output) for i in range(n)]
    return reconstruct(result_shares)


def task1():
    print("Task 1: \n")
    field = galois.GF(2 ** 13 - 1)
    secret = field(7)
    shares = shamir_share(secret, 2, 5)
    print("Secret:", secret)
    print("Shares:", shares)

    recovered = reconstruct(shares[:3])
    print("Reconstructed from 3 shares (t+1):", recovered)

    k = field(10)
    new_shares = add_const(shares, k)
    recovered_add = reconstruct(new_shares[:3])
    print("Reconstructed after adding 10:", recovered_add)

    new_shares = mult_const(shares, k)
    recovered_mult = reconstruct(new_shares[:3])
    print("Reconstructed after multiplying by 10:", recovered_mult)


# Protocol
#
# Setup: Secrets a and b are shared among n parties with threshold t. Party i holds shares a_i and b_i
# Round 1: - Each party computes a_i * b_i = c_i locally
#          - Each party then re-shares c_i with the other parties
# Round 2: - Each party received one share from every party
#          - Each party computes the Lagrange coefficients for points 1 to n
#          - Each party computes their output share d_i
# Result: Each party has d_i, which is a share of a * b with threshold t
def task2():
    print("Task 2: \n")
    field = galois.GF(2 ** 13 - 1)
    a = field(4)
    b = field(7)
    result = run_bgw(2, 5, a, b)
    print(f"{a} * {b} = {result}")

if __name__ == "__main__":
    task1()
    print()
    task2()
