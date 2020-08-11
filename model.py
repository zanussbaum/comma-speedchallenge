import numpy as np

from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, x, y):
        self.model.fit(x, y)

        pred = self.model.predict(x)

        return self.error(y, pred)

    def error(self, y, yhat):
        err = np.square(y-yhat)
        return np.mean(err)

    def predict(self, x):
        return self.model.predict(x)
