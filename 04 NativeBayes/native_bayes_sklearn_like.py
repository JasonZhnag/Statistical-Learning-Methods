# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelBinarizer
from utils import safe_sparse_dot

_ALPHA_MIN = 1e-10

class _BaseNB(object):
    """
    class_count (c, 1): 记录每一类的文档数量
    feature_count (c, n): 记录每一类的对应特征值和
    feature_log_prob (c, n): P(x=x_{c_in_i})
    class_log_prior (c, 1): 类的先验概率
    log_likelihood (c, 1):
    feature_all (1, n)
    """
    def _check_alpha(self, alpha):
        assert ((alpha <= 1.0) and (
                alpha > 0.0)), "ERROR: smoothing parameter alpha should have value within [0.0, 1.0]!"

    def _init_counters(self, n_effective_classes, n_features):
        self.class_count_ = np.zeros(n_effective_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_effective_classes, n_features),
                                       dtype=np.float64)
    def _update_class_log_prior(self):
        self.class_log_prior_ = (np.log(self.class_count_) -
                                 np.log(self.class_count_.sum()))

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        label_bin = LabelBinarizer()
        Y = label_bin.fit_transform(y)
        self.classes_ = label_bin.classes_
        n_classes = Y.shape[1]

        self._init_counters(n_classes, self.n_features_)
        self._count(X, Y)
        self._update_feature_log_prob(self.alpha_)
        self._update_class_log_prior()
        return self

    def predict(self, X):
        log_likelihood = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(log_likelihood, axis=1)]

class MultinomialNB(_BaseNB):
    def __init__(self, alpha=1.0, fit_prior=True):
        self._check_alpha(alpha)
        self.alpha_ = alpha
        self.fit_prior_ = fit_prior

    def _count(self, X, Y):
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        smoothed_fc = self.feature_count_ + alpha   # (c, n)
        smoothed_cc = smoothed_fc.sum(axis=1)   # (c, 1)

        self.feature_log_prob_ = (np.log(smoothed_fc) -
                                  np.log(smoothed_cc.reshape(-1, 1)))

    def _joint_log_likelihood(self, X):
        return (safe_sparse_dot(X, self.feature_log_prob_.T) +
                self.class_log_prior_)

class ComplementNB(_BaseNB):
    def __init__(self, alpha=1.0, fit_prior=True):
        self._check_alpha(alpha)
        self.alpha_ = alpha
        self.fit_prior_ = fit_prior

    def _count(self, X, Y):
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)
        self.feature_all_ = self.feature_count_.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        comp_count = self.feature_all_ + alpha - self.feature_count_
        logged = np.log(comp_count / comp_count.sum(axis=1, keepdims=True))
        self.feature_log_prob_ = -logged

    def _joint_log_likelihood(self, X):
        return safe_sparse_dot(X, self.feature_log_prob_.T)
