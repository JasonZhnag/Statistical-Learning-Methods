import numpy as np
from utils import Heap, euc_dis
from math import inf
import matplotlib.pyplot as plt

class KDTree:
    class Node:
        def __init__(self, X, y, axis):
            self.X = X
            self.y = y
            self.axis = axis
            self.left = None
            self.right = None

    def build(self, X, y, split_axis=0):
        if not len(X):
            return None
        # Wait for test
        median_index = np.argpartition(X[:, split_axis], len(X) // 2, axis=0)[len(X) // 2]
        split_point = float(X[median_index, split_axis])
        equal_x = X[X[:, split_axis] == split_point]
        equal_y = y[X[:, split_axis] == split_point]
        less_x = X[X[:, split_axis] < split_point]
        less_y = y[X[:, split_axis] < split_point]
        greater_x = X[X[:, split_axis] > split_point]
        greater_y = y[X[:, split_axis] > split_point]
        node = self.Node(equal_x, equal_y, split_axis)
        node.left = self.build(less_x, less_y, 1 - split_axis)
        node.right = self.build(greater_x, greater_y, 1 - split_axis)
        return node

    def _query(self, root, x, k):
        if not root:
            return Heap(max_len=k, key=lambda xy: -euc_dis(x, xy[0])), inf
        min_dis = inf
        if x[root.axis] <= root.X[0][root.axis]:
            ans, l_min_dis = self._query(root.left, x, k)
            min_dis = min(min_dis, l_min_dis)
            sibling = root.right
        else:
            ans, r_min_dis = self._query(root.right, x, k)
            min_dis = min(min_dis, r_min_dis)
            sibling = root.left

        for cur_x, cur_y in zip(root.X, root.y):
            min_dis = min(euc_dis(cur_x, x), min_dis)
            ans.push((cur_x, cur_y))

        if min_dis > abs(x[root.axis] - root.X[0][root.axis]):
            other_ans, other_min_dis = self._query(sibling, x, k)
            min_dis = min(min_dis, other_min_dis)
            while other_ans:
                other_x, other_y = other_ans.pop()
                ans.push((other_x, other_y))

        return ans, min_dis

    def query(self, X, k):
        return self._query(self.root, X, k)[0]

    def __init__(self, X, y):
        self.root = self.build(X, y)

class KNN:
    def __init__(self, k=1, distance_func="l2"):
        self.k = k
        if distance_func == 'l2':
            self.distance_func = lambda x, y: np.linalg.norm(x - y)
        else:
            self.distance_func = distance_func

    def _predict(self, x):
        top_k = self.tree.query(x, self.k)
        top_k_y = [y for x, y in top_k]
        return np.argmax(np.bincount(top_k_y))

    def fit(self, X, y):
        self.tree = KDTree(X, y)
        self.k = min(self.k, len(X))

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)

if __name__ == "__main__":
    def evaluate(X_train, y_train, X_test, k, desc):
        knn = KNN(k=k)
        knn.fit(X_train, y_train)
        pred_test = knn.predict(X_test)

        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=20)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=pred_test, marker=".", s=1)
        plt.title(desc)
        plt.show()

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    y_train = np.array([1, 2, 3, 4, 5])
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    evaluate(X_train, y_train, X_test, 1, "Example 1")

    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [.5, .5]])
    y_train = np.array([1, 1, 2, 3, 4])
    # generate grid-shaped test data
    X_test = np.concatenate(np.stack(np.meshgrid(np.linspace(-1, 2, 100), np.linspace(-1, 2, 100)), axis=-1))
    evaluate(X_train, y_train, X_test, 1, "Example 2")
