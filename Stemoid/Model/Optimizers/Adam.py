import numpy as np

#Adam最適化関数
#Adam Optimizer

class Adam:

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        #パラメーター
        #Variables
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    #勾配を更新
    #Update Gradients
    def update(self, model, grads, gradient_model):
        if self.m is None:
            # 逆転伝播法で勾配を計算
            # Calculate Gradiant With Back Propagation
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
        #計算した勾配を保存されてるレイヤーに適応する
        #Apply Calculated Gradient To Saved Layers
        for x in gradient_model:
            self.m[ii] += (1 - self.beta1) * (grads[x]['weights'] - self.m[ii])
            self.v[ii] += (1 - self.beta2) * (grads[x]['weights'] ** 2 - self.v[ii])

            model[x].weights -= lr_t * self.m[ii] / (np.sqrt(self.v[ii]) + 1e-7)

            ii += 1

            self.m[ii] += (1 - self.beta1) * (grads[x]['bias'] - self.m[ii])
            self.v[ii] += (1 - self.beta2) * (grads[x]['bias'] ** 2 - self.v[ii])

            model[x].bias -= lr_t * self.m[ii] / (np.sqrt(self.v[ii]) + 1e-7)

            ii += 1