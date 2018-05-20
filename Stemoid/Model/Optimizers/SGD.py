class SGD:
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, model, grads):
        for x in range(len(model)):
            update_wei = model[x].weights - (self.lr * grads[x]['weights'])
            update_bia = model[x].bias - (self.lr * grads[x]['bias'])
            model[x].set_weights(update_wei, update_bia)