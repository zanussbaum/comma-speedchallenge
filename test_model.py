import numpy as np

from model import Model
from sklearn.metrics import mean_squared_error


def test_error():
    model = Model()

    y = np.array([1, 2, 3, 4, 5])
    yhat = np.array([0, 1, 2, 3, 4])

    assert model.error(y, yhat) == 1

    y = np.array([1.0, 2.5, 3.5])
    yhat = np.array([.5, 2.0, 3.0])

    assert model.error(y, yhat) == .25

    y = np.random.randint(0, 10, size=50)
    yhat = np.random.randint(0, 10, size=50)

    assert model.error(y, yhat) == mean_squared_error(y, yhat)
