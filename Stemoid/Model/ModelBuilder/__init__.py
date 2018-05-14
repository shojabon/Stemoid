import numpy as np


class ModelBuilder:
    
    def __init__(self):
        self.model = []
        self.model_shape = []
        self.input = None
        self.first_done = False

    def add(self, layer):
        if self.first_done:
            self.model.append(layer)
        if not self.first_done:
            self.input = layer
            self.first_done = True
        return self

    def compile(self, weight_multiplier=0.01):
        self.model_shape.append(self.input.get_size())
        for x in self.model:
            self.model_shape.append(x.get_size())
        for x in range(len(self.model_shape) - 1):
            self.model[x].compile(weight_multiplier * np.random.rand(self.model_shape[x], self.model_shape[x+1]),weight_multiplier * np.random.rand(self.model_shape[x+1]))

    def p(self, input_data):
        input_data = self.input.forward(input_data)
        for x in range(len(self.model_shape) - 1):
            input_data = self.model[x].forward(input_data)
        return input_data

