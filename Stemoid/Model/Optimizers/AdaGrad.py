import numpy as np


class AdaGrad:

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, model, grads, gradient_model):
        if self.h is None:
            self.h = {}
            i = 0
            for x in gradient_model:
                self.h[i] = np.zeros_like(model[x].weights)
                i += 1
                self.h[i] = np.zeros_like(model[x].bias)
                i += 1
        ii = 0
        for x in gradient_model:
            self.h[ii] += grads[x]['weights'] * grads[x]['weights']
            model[x].weights -= self.lr * grads[x]['weights'] / (np.sqrt(self.h[ii]) + 1e-7)
            ii += 1
            self.h[ii] += grads[x]['bias'] * grads[x]['bias']
            model[x].bias -= self.lr * grads[x]['bias'] / (np.sqrt(self.h[ii]) + 1e-7)
            ii += 1
