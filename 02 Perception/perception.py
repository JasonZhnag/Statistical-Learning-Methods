import numpy as np
import matplotlib.pyplot as plt

class Perception:
    def __init__(self, lr=1e-1, max_iter=2000, verbose=False):
        self.lr = lr
        self.mat_iter = max_iter
        self.verbose = verbose

    def _calculate(self, X):
        return self.w @ X + self.b

    def _predict(self, X):
        return 1 if self._calculate(X) >= 0. else -1

    def fit(self, X, y):
        self.feature_size = X.shape[-1]
        self.w = np.random.rand(self.feature_size)
        self.b = np.random.rand(1)

        for epoch in range(1, self.mat_iter + 1):
            if self.verbose:
                print(f"epoch {epoch} started....")

            update_cnt = 0
            perm = np.random.permutation(len(X))
            for i in perm:
                xi, yi = X[i], y[i]
                if self._predict(xi) != yi:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    update_cnt += 1

            if self.verbose:
                print(f"epoch {epoch} finished, {update_cnt} pieces of mis-classifying")

            if update_cnt == 0:
                break

    def predict(self, X):
        return np.apply_along_axis(self._predict, axis=-1, arr=X)

if __name__ == '__main__':
    def evaluate(X, y, desc):
        perception = Perception(verbose=True)
        perception.fit(X, y)

        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.title(desc)
        plt.show()

    print("Example 1:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 1, -1, -1])
    evaluate(X, y, "Example 1")

    print("Example 2: Perception cannot solve a simple XOR problem")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, -1, -1, 1])
    evaluate(X, y, "Example 2: Perception cannot solve a simple XOR problem")