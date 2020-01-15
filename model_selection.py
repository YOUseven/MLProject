# @Time    : 2019/11/24 16:08
# @Author  : Seven
# @Email   : ysq96@126.com
# @File    : model_selection.py
# @Software: PyCharm Community Edition

import numpy as np

def train_test_split(X, y, test_ratio=0.2, seed=None):
    """将数据X，y按照test_ratio分成X_train,y_train,X_test,y_test"""

    assert X.shape[0] == y.shape[0], \
    "the number of X must equal to y"
    assert 0 <= test_ratio <= 1, \
    "the test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffle_indexs = np.random.permutation(len(X))

    train_size = len(X) - int(test_ratio*len(X))
    train_indexs = shuffle_indexs[:train_size]
    test_indexs = shuffle_indexs[train_size:]

    X_train = X[train_indexs]
    X_test = X[test_indexs]

    y_train = y[train_indexs]
    y_test = y[test_indexs]

    return X_train, X_test, y_train, y_test