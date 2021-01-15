from matplotlib import pyplot as plt
import numpy as np
import heapq
from math import inf, nan
from math import log, sqrt
from collections import Counter
from scipy import sparse

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

def safe_sparse_dot(a, b, *, dense_output=False):
    """Dot product that handle the sparse matrix case correctly

    Parameters
    ----------
    a : array or sparse matrix
    b : array or sparse matrix
    dense_output : boolean, (default=False)
        When False, ``a`` and ``b`` both being sparse will yield sparse output.
        When True, output will always be a dense array.

    Returns
    -------
    dot_product : array or sparse matrix
        sparse if ``a`` and ``b`` are sparse and ``dense_output=False``.
    """
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sparse.issparse(a) and sparse.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret