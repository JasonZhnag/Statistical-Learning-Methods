from collections import defaultdict, Counter
import numpy as np

class NativeBayes:
    def __init__(self, alpha=1, verbose=False):
        # p(a|y), the probability of an attribute a when the data is of label y
        # its a three-layer dict
        # the first-layer key is y, the value label
        # the second-layer key is n, which means the nth attribute
        # the thrid-layer key is the value of the nth attribute
        self.pay = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        self.py = defaultdict(lambda: 0)
        self.verbose = verbose
        self.alpha = alpha

    def fit(self, X, y):
        y_cnt = Counter(y)
        for col in range(len(X[0])):
            col_values = set(x[col] for x in X)
            for x, y_ in zip(X, y):
                self.pay[y_][col][x[col]] += 1
            for y_ in y_cnt:
                for a in self.pay[y_][col]:
                    self.pay[y_][col][a] += self.alpha
                    self.pay[y_][col][a] /= y_cnt[y_] + self.alpha * len(col_values)
        for y_ in y_cnt:
            self.py[y_] = (y_cnt[y_] + self.alpha) / (len(X) + self.alpha * len(y_cnt))

        if self.verbose:
            for y_ in self.pay:
                print(f'The prior probability of label {y_} is', self.py[y_])
                for nth in self.pay[y_]:
                    prob = self.pay[y_][nth]
                    for a in prob:
                        print(f'When the label is {y_}, the probability that {nth}th attribute be {a} is {prob[a]}')

    def _predict(self, x):
        labels = list(self.pay.keys())
        probs = []
        for y in labels:
            prob = self.py[y]
            for i, a in enumerate(x):
                prob *= self.pay[y][i][a]
            probs.append(prob)
        if self.verbose:
            for y, p in zip(labels, probs):
                print(f'The likelihood {x} belongs to {y} is {p}')
        return labels[np.argmax(probs)]

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)

if __name__ == "__main__":
    native_bayes = NativeBayes(verbose=True)
    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = [
        [1, 'S'],
        [1, 'M'],
        [1, 'M'],
        [1, 'S'],
        [1, 'S'],
        [2, 'S'],
        [2, 'M'],
        [2, 'M'],
        [2, 'L'],
        [2, 'L'],
        [3, 'L'],
        [3, 'M'],
        [3, 'M'],
        [3, 'L'],
        [3, 'L'],
    ]
    y = [-1, -1, 1, 1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1]
    native_bayes.fit(X, y)
