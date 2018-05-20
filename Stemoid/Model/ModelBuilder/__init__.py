import numpy as np

from Stemoid.Model.Layers.Softloss import SoftLoss
from Stemoid.Model.Optimizers.SGD import SGD


class ModelBuilder:

    def __init__(self):
        self.model = []
        self.model_shape = []
        self.input = None
        self.first_done = False
        self.loss = None
        self.optimizer = None

    def add(self, layer):
        if self.first_done:
            self.model.append(layer)
        if not self.first_done:
            self.input = layer
            self.first_done = True
        return self

    def compile(self, optimizer=SGD(), loss=SoftLoss()):
        self.model_shape.append(self.input.get_size())
        for x in self.model:
            self.model_shape.append(x.get_size())
        for x in range(len(self.model_shape) - 1):
            self.model[x].compile(self.model_shape[x], self.model_shape[x + 1])
        self.loss = loss
        self.optimizer = optimizer

    def predict(self, input_data):
        input_data = self.input.forward(input_data)
        for x in range(len(self.model_shape) - 1):
            input_data = self.model[x].forward(input_data)
        return input_data

    def get_loss(self, input_data, label):
        out = self.predict(input_data)
        return self.loss.forward(out, label)

    def get_gradient(self, input_data, label):
        self.get_loss(input_data, label)
        dout = self.loss.backward()
        lista = list(self.model)
        lista.reverse()
        out = []
        for x in lista:
            dout = x.backward(dout)
        for x in lista:
            out.append({'weights': x.doutWeights,
                        'bias': x.doutBias})
        out.reverse()
        return out

    def get_accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def learn(self, input_data, label):
        gradient = self.get_gradient(input_data, label)
        self.optimizer.update(self.model, gradient)