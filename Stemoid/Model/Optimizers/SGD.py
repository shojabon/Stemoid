class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, model, grads, gradient_model):
        for x in gradient_model:
            model[x].weights -= (self.lr * grads[x]['weights'])
            model[x].bias -= (self.lr * grads[x]['bias'])