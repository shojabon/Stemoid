import numpy as np

from Stemoid.Model.LossFunctions.CrossEntropyError import CrossEntropyError


class SoftLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def softmax(self, x):
        if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T

        x = x - np.max(x)
        return np.exp(x) / np.sum(np.exp(x))

    def forward(self, x, t):
        self.t = t
        self.y = self.softmax(x)
        self.loss = CrossEntropyError().get_loss(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx