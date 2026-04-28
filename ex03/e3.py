# %% Imports and definitions
import numpy as np
from collections import defaultdict
import galois
import pytest

# %%

# arbitrary
GLOBAL_SEED = 1234
field = galois.GF(2 ** 13 - 1)


def shamir_share(x, t, n):
    np.random.seed(GLOBAL_SEED)
    p = field(np.random.choice(np.arange(1, field.order), size=n, replace=False))

    random_poly = field.Random(t, low=1)

    poly = galois.Poly(np.append(random_poly, x), field=field)

    secrets = []
    for i in range(n):
        secrets.append([p[i], poly(p[i])])
    return field(secrets)


def test_shamir_share_reconstruction():
    """Test if t+1 shares correctly reconstruct the secret."""
    t, n = 2, 5
    secret_value = field(42)
    shares = shamir_share(secret_value, t, n)

    # Use exactly t+1 shares (the minimum required)
    reconstructed = reconstruct(shares[:t + 1])
    assert reconstructed == secret_value


def test_shamir_share_insufficient_shares():
    """Test that t shares are not enough to reconstruct the secret."""
    t, n = 2, 5
    secret_value = field(100)
    shares = shamir_share(secret_value, t, n)

    # Try to reconstruct with only t shares
    # This should result in a different value (or fail depending on implementation)
    reconstructed = reconstruct(shares[:t])
    assert reconstructed != secret_value


def test_shamir_share_uniqueness():
    """Checks that participants get distinct and unique data points."""
    t, n, secret = 2, 5, field(123)
    shares = shamir_share(secret, t, n)

    x_coords = [int(x) for x in shares[:, 0]]
    y_shares = [int(y) for y in shares[:, 1]]

    # Check: No two people have the same ID (x)
    assert len(set(x_coords)) == n

    # Check: No two people have the same share value (y)
    # (In a large field, it is statistically impossible for two shares to be identical)
    assert len(set(y_shares)) == n


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
    return galois.lagrange_poly(shares[:, 0], shares[:, 1])(0)


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
        assert 2 * t + 1 <= n

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
            y += coeffs[i] * self.shares[i][1]
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
    return reconstruct(outputs)


def get_lagrange_coeffs(x_values):
    n = len(x_values)
    p = []

    for i in range(n):
        y_values = field.Zeros(n)
        y_values[i] = 1

        p.append(galois.lagrange_poly(x_values, y_values)(0))

    return p


# Tests

## Task 2

def test_run_bgw_large_t():
    t, n, a, b = 3, 5, 5, 5
    with pytest.raises(Exception):
        run_bgw(t, n, a, b)


def test_run_bgw_correctness():
    t = [2, 2, 4]
    n = [5, 5, 10]
    a = [5, 6, 59]
    b = [5, 1, 42]

    for i in range(len(n)):
        assert a[i] * b[i] == run_bgw(t[i], n[i], a[i], b[i])


def test_get_lagrange_coeffs_two_points():
    """
    x = [1, 2]
    L0(0) = (0 - x1) / (x0 - x1) = (0 - 2) / (1 - 2) = (0-2)/-1 = 2
    L1(0) = (0 - x0) / (x1 - x0) = (0 - 1) / (2 - 1) = (0-1) = -1
    """
    x_values = field([1, 2])

    expected = [field(2), field(-1 % field.order)]

    result = get_lagrange_coeffs(x_values)

    assert result == expected

    # sum of lagrange coeffs must be 1
    assert field(np.sum(result) % field.order) == field(1)

    # check if weighted sum is 0
    sum = field(0)
    for i in range(len(x_values)):
        sum += result[i] * x_values[i]
    assert sum == field(0)


def test_get_lagrange_coeffs_three_points():
    """
    x = [1, 2, 3]
    L0(0) = (0-2)(0-3) / (1-2)(1-3) = 6 / 2 = 3
    L1(0) = (0-1)(0-3) / (2-1)(2-3) = 3 / -1 = -3
    L2(0) = (0-1)(0-2) / (3-1)(3-2) = 2 / 2 = 1
    """
    x_values = field([1, 2, 3])
    expected = [field(3), field(-3 % field.order), field(1)]

    result = get_lagrange_coeffs(x_values)

    assert result == expected
    assert field(np.sum(result) % field.order) == field(1)

    # check if weighted sum is 0
    sum = field(0)
    for i in range(len(x_values)):
        sum += result[i] * x_values[i]
    assert sum == field(0)


# %% Tasks
def task1():
    print('-' * 6, 'Task 1', '-' * 6)
    t, n, a, b = 1, 4, 12, 5
    print(f"t:{t}, n:{n}, a:{a}, b:{b}")
    secret1 = shamir_share(field(a), t, n)
    secret2 = shamir_share(field(b), t, n)
    print("Secret1\n", secret1)
    print("Secret2\n", secret2)
    c = 2
    print("Adding constant", c, "to shares1")
    secret1 = add_const(secret1, c)
    reconstruction = reconstruct(secret1[0:2])
    print("re:", reconstruction)

    print("Adding shares together..")
    secret3 = add_shares(secret1, secret2)
    reconstruction = reconstruct(secret3[0:2])
    print("re:", reconstruction)
    print()


def task2():
    print('-' * 6, 'Task 2', '-' * 6)
    t, n, a, b = 2, 5, 5, 5
    print(f"t:{t}, n:{n}, a:{a}, b:{b}")
    print("re:", run_bgw(t, n, a, b))
    print()


# %%
if __name__ == "__main__":
    task1()
    task2()
    pass

def test_reconstruct_basic():
    shares = shamir_share(field(42), 2, 4)
    assert reconstruct(shares[:3]) == field(42)

def test_reconstruct_different_subsets_with_same_content():
    shares = shamir_share(field(77), 2, 5)
    r1 = reconstruct(shares[0:3])
    r2 = reconstruct(shares[1:4])
    r3 = reconstruct(shares[2:5])
    assert r1 == r2 == r3 == field(77)

def test_reconstruct_t1():
    shares = shamir_share(field(99), 1, 4)
    assert reconstruct(shares[:2]) == field(99)

def test_reconstruct_after_add_shares():
    s1 = shamir_share(field(30), 2, 4)
    s2 = shamir_share(field(20), 2, 4)
    s3 = add_shares(s1, s2)
    assert reconstruct(s3[:3]) == field(50)

def test_reconstruct_after_mult():
    shares = shamir_share(field(10), 2, 4)
    shares = mult_const(shares, 3)
    assert reconstruct(shares[:3]) == field(30)
