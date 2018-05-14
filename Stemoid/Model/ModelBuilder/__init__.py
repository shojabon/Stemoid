import numpy as np

from Stemoid.Model.Layers.Softloss import SoftLoss
from Stemoid.Model.LossFunctions.CrossEntropyError import CrossEntropyError
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

    def compile(self, optimizer=SGD(),loss=SoftLoss(),starting_weight_multiplier=0.01,):
        self.model_shape.append(self.input.get_size())
        for x in self.model:
            self.model_shape.append(x.get_size())
        for x in range(len(self.model_shape) - 1):
            self.model[x].compile(starting_weight_multiplier * np.random.rand(self.model_shape[x], self.model_shape[x+1]),starting_weight_multiplier * np.random.rand(self.model_shape[x+1]))
        self.loss = loss
        self.optimizer = optimizer


    def predict(self, input_data):
        input_data = self.input.forward(input_data)
        for x in range(len(self.model_shape) - 1):
            input_data = self.model[x].forward(input_data)
        return input_data

    def get_loss(self, input_data, label):
        return self.loss.forward(self.predict(input_data), label)

    def get_gradient(self, input_data, label):
        self.get_loss(input_data, label)




