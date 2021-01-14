from matplotlib import pyplot as plt
import numpy as np
import heapq
from math import inf, nan
from math import log, sqrt
from collections import Counter

class Heap:
    def __init__(self, arr=None, key=lambda x: x, max_len=inf):
        self.key = key
        self.max_len = max_len
        if not arr:
            self.h = []
        else:
            self.h = [(self.key(i), i) for i in arr]
        heapq.heapify(self.h)
        self.i = 0

    def __len__(self):
        return len(self.h)

    def __bool__(self):
        return len(self.h) != 0

    def __iter__(self):
        while self:
            yield self.pop()

    def push(self, x):
        # insert an number to the middle so that `x` will be never compared
        # because maybe `x` doesn't have comparing operator defined
        heapq.heappush(self.h, (self.key(x), self.i, x))
        self.i += 1
        if len(self.h) > self.max_len:
            self.pop()

    def top(self):
        return self.h[0][-1]

    def pop(self):
        return heapq.heappop(self.h)[-1]

def euc_dis(a, b):
    return np.linalg.norm(a - b, axis=-1)