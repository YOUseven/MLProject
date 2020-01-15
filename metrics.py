# @Time    : 2019/11/24 17:14
# @Author  : Seven
# @Email   : ysq96@126.com
# @File    : metrics.py
# @Software: PyCharm Community Edition

import numpy as np
from math import sqrt

def accuracy_score(y_true, y_predict):
    """计算 y_true 和 y_predict 之间的准确率"""

    assert y_true[0] == y_predict[0], \
    "the size of y_true must equal to y_predict"

    return sum(y_true == y_predict) / len(y_true)

def mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的mse"""
    assert len(y_predict) == len(y_true), \
    "the size of y_true must equal to the size of y_predict"

    return np.sum((y_true - y_predict)**2) / len(y_true)

def root_mean_squared_error(y_true, y_predict):
    """计算y_true和y_predict之间的rmse"""

    return sqrt(mean_squared_error(y_true, y_predict))

def mean_absolute_error(y_true, y_predict):
    """计算y_true和y_predict之间的mae"""
    assert len(y_true) == len(y_predict), \
    "the size of y_true must equal to the size of y_predict"

    return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

def r2_score(y_true, y_predict):
    return 1 - (mean_squared_error(y_true, y_predict) / np.var(y_true))