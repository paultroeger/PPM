# Imports and definitions
import numpy as np
from collections import defaultdict
import galois
import pytest


def shamir_share(x, t, n):
    # TODO: your code here
    raise NotImplementedError()


def add_shares(shares1, shares2):
    # TODO: your code here
    raise NotImplementedError()


def add_const(shares, k):
    # TODO: your code here
    raise NotImplementedError()


def mult_const(shares, k):
    # TODO: your code here
    raise NotImplementedError()


def reconstruct(shares):
    # TODO: your code here
    raise NotImplementedError()


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


if __name__ == "__main__":
    field = galois.GF(2 ** 13 - 1)

    # TODO: your code here
    raise NotImplementedError()

