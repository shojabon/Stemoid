import numpy as np

from Stemoid.Model.Layers.Softloss import SoftLoss
from Stemoid.Model.Optimizers.SGD import SGD


class ModelBuilder:

    def __init__(self):
        self.model = []
        self.model_shape = []
        self.model_output_shape = []
        self.gradient_model = []
        self.input = None
        self.first_done = False
        self.loss = None
        self.optimizer = None

        self.train_data = None
        self.train_label = None
        self.train_label_label = None

        self.epoch = 0
        self.iteration = 0
        self.hidden_iteration = 0

    def add(self, layer):
        if self.first_done:
            self.model.append(layer)
        if not self.first_done:
            self.input = layer
            self.first_done = True
        return self

    def compile(self, optimizer=SGD(), loss=SoftLoss()):
        input_shape = self.input.get_output_shape()
        self.model_output_shape.append(input_shape)
        for x in range(len(self.model)):
            input_shape = self.model[x].get_output_shape(input_shape)
            self.model_output_shape.append(input_shape)


        self.model_shape.append(self.input.get_size())
        for x in self.model:
            self.model_shape.append(x.get_size())

        for x in range(len(self.model)):
            if hasattr(self.model[x], 'weights'):
                self.gradient_model.append(x)

        for x in range(len(self.model_shape) - 1):
            self.model[x].compile()
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
        out = {}
        for x in lista:
            dout = x.backward(dout)
        lista.reverse()
        for x in self.gradient_model:
            out[x] = {'weights': lista[x].doutWeights,
                        'bias': lista[x].doutBias}
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
        self.optimizer.update(self.model, gradient, self.gradient_model)

    def learn_epoch(self, batch_size=100):
        if self.train_label_label.size < batch_size:
            self.epoch += 1
            self.train_label_label = np.arange(0, self.train_label.shape[0])
        mask = np.random.choice(self.train_label_label.size, batch_size)
        self.iteration += batch_size
        self.learn(self.train_data[mask], self.train_label[mask])
        self.train_label_label = np.delete(self.train_label_label, mask)

    def learn_random(self, batch_size=100):
        if self.hidden_iteration >= self.train_label.size:
            self.epoch += 1
            self.hidden_iteration = 0
        mask = np.random.choice(self.train_label_label.size, batch_size)
        self.iteration += batch_size
        self.hidden_iteration += batch_size
        self.learn(self.train_data[mask], self.train_label[mask])

    def set_training_data(self, data, label):
        self.train_data = data
        self.train_label = label
        self.train_label_label = np.arange(0, label.shape[0])