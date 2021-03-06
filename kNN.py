# @Time    : 2019/11/24 12:26
# @Author  : Seven
# @Email   : ysq96@126.com
# @File    : kNN.py
# @Software: PyCharm Community Edition

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from math import sqrt
from .metrics import accuracy_score

class KNNClassifier:
    def __init__(self, k):
        """初始化knn分类器"""
        assert k >=1 ,"k must be valid"

        self.k = k
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练knn分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
        "the size of X_train must be equal to y_train."
        assert self.k <= X_train.shape[0], \
        "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待预测X_predict 返回预测结果"""
        assert self._X_train is not None and self._y_train is not None, \
        "must fit before predict"
        assert self._X_train.shape[1] == X_predict.shape[1], \
        "the feature number of X_predict must equal to X_train"

        y_predict = [self._predict(x) for x in X_predict]

        return np.array(y_predict)

    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1], \
        "the feature number of x must equal to X_train"

        distances = [sqrt(sum((x-x_train)**2)) for x_train in self._X_train]
        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """根据测试数据集 X_test 和 y_test 确定当前模型的准确度"""

        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)

    def __repr__(self):
        return "KNN=%d" % self.k