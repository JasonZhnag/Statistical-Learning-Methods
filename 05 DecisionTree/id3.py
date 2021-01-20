from collections import Counter
from utils import argmax, info_gain, info_gain_ratio

class DecitionTree:
    class Node:
        def __init__(self, attr, y):
            self.attr = attr
            self.children = {}
            self.prob = Counter(y)
            s = sum(self.prob.values())
            for idx in self.prob:
                self.prob[idx] /= s
            label_idx, self.label_prob = argmax(self.prob.keys(),
                                                key=self.prob.__getitem__)
            self.label = y[label_idx]

    def __init__(self, info_gain_threshold=0, verbose=False, id3=True):
        self.info_gain_threshold = info_gain_threshold
        self.verbose = verbose
        self.id3 = id3
        self.feature_size = 0

    def build(self, X, y, selected_features):
        nowNode = self.Node(None, y)
        if self.verbose:
            print("Current selected features: ", selected_features)
            print("Current data: ")
            print(X)
            print(y)

        split = False

        if len(selected_features) != self.feature_size and len(set(y)) > 1:
            left_features = list(set(range(self.feature_size)) - selected_features)
            if self.id3:
                left_feature_idx, max_info_gain = argmax(left_features,
                                                    key=lambda idx: info_gain(X, y, idx))
            else:
                left_feature_idx, max_info_gain = argmax(left_features,
                                                    key=lambda idx: info_gain_ratio(X, y, idx))
            feature_idx = left_features[left_feature_idx]
            if max_info_gain >= self.info_gain_threshold and self.verbose:
                print(f"Spilit by {feature_idx}th feature")
                split = True
                nowNode.attr = feature_idx
                for val in set(x[feature_idx] for x in X):
                    ind_mask = [x[feature_idx] == val for x in X]
                    child_X = [x for i, x in zip(ind_mask, X) if i]
                    child_y = [yi for i, yi in zip(ind_mask, y) if i]
                    nowNode.children[val] = self.build(child_X, child_y, selected_features | {feature_idx})

        if not split and self.verbose:
            print("No split")
        return nowNode

    def fit(self, X, y):
        self.feature_size = len(X[0])
        self.root = self.build(X, y, set())

    def query(self, root: Node, x):
        if root.attr is None or x[root.attr] not in root.children:
            return root.label

        return self.query(root.children[x[root.attr]], x)

    def _predict(self, x):
        return self.query(self.root, x)

    def predict(self, X):
        return [self._predict(x) for x in X]

if __name__ == "__main__":
    c45 = DecitionTree(verbose=True, id3=False)
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
    Y = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    c45.fit(X, Y)

    # show in table
    pred = c45.predict(X)
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
    Y = ['否', '否', '是', '否', '否', '否', '是', '是', '是', '是', '是', '否']
    c45.fit(X, Y)

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
    testY = ['否', '否', '是', '是', '否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']

    # show in table
    pred = c45.predict(testX)
    print(pred)

