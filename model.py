import pickle

import numpy as np

from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self):
        self.model = LinearRegression(normalize=True)

    def fit(self, x, y):
        self.model.fit(x, y)

        pred = self.model.predict(x)

        return self.error(y, pred)

    def error(self, y, yhat):
        err = np.square(y-yhat)
        return np.mean(err)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, save_path='model'):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model
