
"""
Count Min Sketch for approximating counts of distinct elements.
"""

import numpy as np
from heapdict import heapdict

# Small list of "large" prime numbers for convenience, from (http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php)
PRIMES = [157189, 164387, 171793, 178987, 186107]

class CountMin():
    def __init__(self, num_hash, length_table, heavy_hitters=0):
        """
        Initialize the count-min sketch datastructure.
        This data structure is used to approximate the counts of distint items.
        Parameters:
            num_hash: Number of tables or hash functions
            length_table: Number of counters per table, assumed <= min(PRIMES)
            heavy_hitters: Number of most frequent elements being tracked
        """
        self.hh_count = heavy_hitters
        self.min_heavy_count = 0
        self.heavy_hitters = heapdict()
        self.n = 0
        self.tables = np.zeros((num_hash, length_table), dtype=np.int32)
        self.hashfs = np.array([self.get_uhash(PRIMES[p % len(PRIMES)], length_table) for p in range(num_hash)])

    @staticmethod
    def get_uhash(prime, length_table, a=None, b=None):
        """
        Assuming a is distributed uniformly over [1,prime) and b is distributed
        uniformly over [0, prime), Pr[h(x)=h(y)] <= 1/length_table.
        This form of hash function is based on work by Carter, Wegman (https://doi.org/10.1016%2F0022-0000%2879%2990044-8)

        Parameters:
            prime: A prime number > length_table
            length_table: Values will be hashed to the range [0,length_table)
            a, b: Random integers
        Returns a 2-universal hash function, of the form: ((ax+b) % prime) % length_table
        """
        if a is None or b is None:
            a = np.random.randint(1,prime)
            b = np.random.randint(prime)
        def h(x):
            if isinstance(x, tuple):
                return ((a*int.from_bytes("_".join(x).encode(), "big")+b) % prime) % length_table
            elif isinstance(x, str):
                return ((a*int.from_bytes(x.encode(), "big")+b) % prime) % length_table
            return ((a*x+b) % prime) % length_table
        return h

    def update(self, elem, sub_error=False):
        """
        Update counters in each table based on a new element.
        elem is assumed to be an integer.
        """
        self.n += 1
        for i in range(len(self.tables)):
            self.tables[i][self.hashfs[i](elem)] += 1
        if self.hh_count > 0:
            # Update heavy hitters
            self.heavy_hitters[elem] = self.get_count(elem, sub_error)
            if len(self.heavy_hitters) > self.hh_count:
                # Remove least count
                self.heavy_hitters.popitem()

    def get_count(self, elem, sub_error=False):
        """
        Return the minimum counter that elem hashes to in all of the tables.
        """
        if sub_error:
            count = np.average([self.tables[i][self.hashfs[i](elem)] for i in range(len(self.tables))])
            return max((len(self.tables[0])*count - self.n) / (len(self.tables[0]) - 1), 0)
        return np.min([self.tables[i][self.hashfs[i](elem)] for i in range(len(self.tables))])

    @staticmethod
    def evaluate_params(num_hash, length_table, n):
        """
        Given parameters for the number of hash functions (and tables),
        and the length of a table, evaluate a bound on the expected error for any count
        with a probability that all frequencies are less than this bound. 
        """
        error_bound = (2*n) / length_table
        confidence = 1 - ((1 / (2 ** num_hash)) * n)
        return error_bound, confidence

    @staticmethod
    def get_params(error_bound, confidence, n):
        """
        Given a bound on the expected error for any count, and a probability that all frequencies
        are less than this bound, evaluate the table length and number of tables (and hash functions)
        in order to reach these metrics.
        """
        length_table = (2*n) / error_bound
        num_hash = np.log2(n / (1 - confidence))
        return num_hash, length_table