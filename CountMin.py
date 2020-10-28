
"""
Count Min Sketch for approximating counts of distinct elements.
"""

import numpy as np
import heapq

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
        self.heavy_hitters = {}
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
        return lambda x: ((a*x+b) % prime) % length_table

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
            if len(self.heavy_hitters) < self.hh_count and elem not in self.heavy_hitters:
                self.heavy_hitters[elem] = self.get_count(elem, sub_error)
                return
            new_count = self.get_count(elem, sub_error)
            if new_count > self.min_heavy_count:
                self.min_heavy_count = min(self.heavy_hitters.values())
                if elem not in self.heavy_hitters:
                    for key in self.heavy_hitters:
                        if self.heavy_hitters[key] == self.min_heavy_count:
                            self.heavy_hitters.pop(key)
                            break
                self.heavy_hitters[elem] = new_count
            



    def get_count(self, elem, sub_error=False):
        """
        Return the minimum counter that elem hashes to in all of the tables.
        """
        if sub_error:
            count = np.average([self.tables[i][self.hashfs[i](elem)] for i in range(len(self.tables))])
            return (len(self.tables[0])*count - self.n) / (len(self.tables[0]) - 1)
        return np.min([self.tables[i][self.hashfs[i](elem)] for i in range(len(self.tables))])
