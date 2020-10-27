
"""
Count Min Sketch for approximating counts of distinct elements.
"""

import numpy as np

# Small list of large prime numbers for convenience, from (http://compoasso.free.fr/primelistweb/page/prime/liste_online_en.php)
PRIMES = [157189, 164387, 171793, 178987, 186107]

class CountMin():
    def __init__(self, num_hash, length_table):
        """
        Initialize the count-min sketch datastructure.
        This data structure is used to approximate the counts of distint items.
        length_table is assumed to be <= min(PRIMES).
        """
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
        return lambda x: ((a*x+b) % prime) % length_table

    def update(self, elem):
        """
        Update counters in each table based on a new element.
        elem is assumed to be an integer.
        """
        for i in range(len(self.tables)):
            self.tables[i][self.hashfs[i](elem)] += 1

    def get_count(self, elem):
        """
        Return the minimum counter that elem hashes to in all of the tables.
        """
        return min([self.tables[i][self.hashfs[i](elem)] for i in range(len(self.tables))])

