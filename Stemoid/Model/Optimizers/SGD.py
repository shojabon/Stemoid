#SGD最適化関数
#SGD Optimizer
class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    #勾配を更新
    #Update Gradients
    def update(self, model, grads, gradient_model):
        #計算した勾配を保存されてるレイヤーに適応する
        #Apply Calculated Gradient To Saved Layers
        for x in gradient_model:
            model[x].weights -= (self.lr * grads[x]['weights'])
            model[x].bias -= (self.lr * grads[x]['bias'])