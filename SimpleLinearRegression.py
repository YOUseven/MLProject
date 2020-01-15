# @Time    : 2019/11/27 13:09
# @Author  : Seven
# @Email   : ysq96@126.com
# @File    : SimpleLinearRegression.py
# @Software: PyCharm Community Edition

import numpy as np
from .metrics import r2_score

class SimpleLinearRegression1:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
        "simple linear regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
        "the size of x_train must be qual to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x_i, y_i in zip(x_train, y_train):
            num += (x_i - x_mean) * (y_i - y_mean)
            d += (x_i - x_mean) **2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
        "simple linear regression can only solve single feature of training data"
        assert self.a_ is not None and self.b_ is not None, \
        "must fit before predict"

        return [self._predict(x) for x in x_predict]

    def _predict(self, x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果值"""
        return x_single * self.a_ + self.b_

    def __repr__(self):
        return "SimpleLinearRegression()"

class SimpleLinearRegression2:

    def __init__(self):
        """初始化Simple Linear Regression 模型"""
        self.a_ = None
        self.b_ = None

    def fit(self, x_train, y_train):
        """根据训练数据集x_train,y_train训练Simple Linear Regression模型"""
        assert x_train.ndim == 1, \
        "simple linear regression can only solve single feature training data"
        assert len(x_train) == len(y_train), \
        "the size of x_train must be qual to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean
        return self

    def predict(self, x_predict):
        """给定待预测数据集x_predict,返回表示x_predict的结果向量"""
        assert x_predict.ndim == 1, \
        "simple linear regression can only solve single feature of training data"
        assert self.a_ is not None and self.b_ is not None, \
        "must fit before predict"

        return [self._predict(x) for x in x_predict]

    def _predict(self, x_single):
        """给定单个待预测数据x_single,返回x_single的预测结果值"""
        return x_single * self.a_ + self.b_

    def score(self, x_test, y_test):

        y_predic = self.y_predic(x_test)

        return r2_score(y_test, y_predic)

    def __repr__(self):
        return "SimpleLinearRegression()"