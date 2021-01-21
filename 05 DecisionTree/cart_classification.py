from collections import Counter
from utils import argmax, gini
from math import inf, nan
import numpy as np

class DecisionTree:
    class Node:
        def __init__(self, attr, y):
            self.attr = attr
            self.left, self.right = None, None
            self.val = None
            self.label = Counter(y).most_common(1)[0][0]
            self.alpha = None

    def __init__(self, verbose=False, is_prune=False):
        self.verbose = verbose
        if is_prune:
            self.possible_alpha = set()

    def gini_sum(self, y1, y2):
        size = len(y1) + len(y2)
        return len(y1) / size * gini(y1) + len(y2) / size * gini(y2)

    def build(self, X, y):
        nowNode = self.Node(None, y)
        if self.verbose:
            print("Current data: ")
            print(X)
            print(y)

        best_gini = inf
        best_idx, best_val = -1, nan

        if len(set(y)) > 1:
            for feature_idx in range(len(X[0])):
                val_set = set(X[:, feature_idx])
                if len(val_set) != 1:
                    for val in val_set:
                        mask = X[:, feature_idx] == val
                        op_mask = X[:, feature_idx] != val
                        y_left = y[mask]
                        y_right = y[op_mask]
                        cur_gini = self.gini_sum(y_left, y_right)
                        if cur_gini < best_gini:
                            best_gini, best_idx, best_val = cur_gini, feature_idx, val
            print(f"Split by value {best_val} of {best_idx}th column")
            mask = X[:, best_idx] == best_val
            op_mask = X[:, best_idx] != best_val
            y_left = y[mask]
            y_right = y[op_mask]
            X_left = X[mask]
            X_right = X[op_mask]

            nowNode.attr = best_idx
            nowNode.val = best_val
            nowNode.left = self.build(X_left, y_left)
            nowNode.right = self.build(X_right, y_right)
        elif self.verbose:
            print("No split")
        return nowNode

    def generate_alpha(self, root: Node, X, y):
        pruned_gini = len(X)*gini(Counter(y).values())
        pruned_loss = pruned_gini
        if root.attr is None:
            return pruned_loss, 1

        cur_loss = 0
        cur_size = 1
        mask = X[:, root.attr] == root.val
        op_mask = X[:, root.attr] != root.val
        y_left = y[mask]
        y_right = y[op_mask]
        X_left = X[mask]
        X_right = X[op_mask]

        child_loss, child_size = self.generate_alpha(root.left, X_left, y_left)
        cur_loss += child_loss
        cur_size += child_size

        child_loss, child_size = self.generate_alpha(root.right, X_right, y_right)
        cur_loss += child_loss
        cur_size += child_size

        alpha = (pruned_loss - cur_loss) / (cur_size - 1)
        root.alpha = alpha
        self.possible_alpha.add(alpha)
        return cur_loss, cur_size

    def fit(self, X, y):
        self.root = self.build(X, y)

    def query(self, root: Node, x, alpha=None):
        # If given alpha is greater than the threshold, then trim.
        if root.attr is None or (alpha and root.alpha and root.alpha < alpha):
            return root.label
        return self.query(root.left, x, alpha) if x[root.attr] == root.val else self.query(root.right, x, alpha)

    def _predict(self, x, alpha=None):
        return self.query(self.root, x, alpha)

    def predict(self, X, alpha=None):
        alpha = alpha if alpha else self.alpha
        return np.array([self._predict(x, alpha) for x in X])

    def validate(self, val_X, val_y, alpha):
        pred_y = self.predict(val_X, alpha)
        return (pred_y == val_y).mean()

    def choose_alpha(self, val_X, val_y):
        best_acc = -1
        best_alpha = 0
        for alpha in self.possible_alpha:
            cur_acc = self.validate(val_X, val_y, alpha)
            if self.verbose:
                print(f"When alpha = {alpha}, accuracy is {cur_acc}")
            if cur_acc > best_acc:
                best_acc = cur_acc
                best_alpha = alpha

        return best_alpha

    def prune(self, X, y, val_X, val_y):
        self.generate_alpha(self.root, X, y)
        self.alpha = self.choose_alpha(val_X, val_y)

if __name__ == "__main__":
    cart = DecisionTree(verbose=True, is_prune=True)
    # -------------------------- Example 1 ----------------------------------------
    # unpruned decision tree predict correctly for all training data
    print("Example 1:")
    X = [
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ]
    X = np.array(X)
    Y = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])
    cart.fit(X, Y)
    cart.prune(X, Y, X, Y)

    # show in table
    pred = cart.predict(X)
    print(pred)

    # -------------------------- Example 2 ----------------------------------------
    # but unpruned decision tree doesn't generalize well for test data
    print("Example 2:")
    X = [
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '否', '否', '一般'],
    ]
    X = np.array(X)
    Y = np.array(['否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否'])
    cart.fit(X, Y)

    testX = [
        ['青年', '否', '否', '一般'],
        ['青年', '否', '否', '好'],
        ['青年', '是', '否', '好'],
        ['青年', '是', '是', '一般'],
        ['青年', '否', '否', '一般'],
        ['老年', '否', '否', '一般'],
        ['老年', '否', '否', '好'],
        ['老年', '是', '是', '好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '非常好'],
        ['老年', '否', '是', '好'],
        ['老年', '是', '否', '好'],
        ['老年', '是', '否', '非常好'],
        ['老年', '否', '否', '一般'],
    ]
    testX = np.array(testX)
    testY = np.array(['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否'])

    # show in table
    pred = cart.predict(testX)
    print(pred)

