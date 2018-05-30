import numpy as np


class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, model, grads, gradient_model):
        if self.m is None:
            self.m, self.v = {}, {}
            i = 0
            for x in gradient_model:
                self.m[i] = np.zeros_like(model[x].weights, dtype=float)
                self.v[i] = np.zeros_like(model[x].weights, dtype=float)
                i += 1
                self.m[i] = np.zeros_like(model[x].bias, dtype=float)
                self.v[i] = np.zeros_like(model[x].bias, dtype=float)
                i += 1

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)
        ii = 0
        for x in gradient_model:
            self.m[ii] += (1 - self.beta1) * (grads[x]['weights'] - self.m[ii])
            self.v[ii] += (1 - self.beta2) * (grads[x]['weights'] ** 2 - self.v[ii])

            new_weights = model[x].weights - lr_t * self.m[ii] / (np.sqrt(self.v[ii]) + 1e-7)

            ii += 1

            self.m[ii] += (1 - self.beta1) * (grads[x]['bias'] - self.m[ii])
            self.v[ii] += (1 - self.beta2) * (grads[x]['bias'] ** 2 - self.v[ii])

            new_bias = model[x].bias - lr_t * self.m[ii] / (np.sqrt(self.v[ii]) + 1e-7)

            ii += 1
            model[x].set_weights(new_weights, new_bias)