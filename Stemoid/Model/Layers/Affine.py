import numpy as np


class Affine:

    def __init__(self, size, activation=None):
        self.ac = activation
        self.size = size
        self.compiled = False

        self.weights = None
        self.bias = None

        self.doutWeights = None
        self.doutBias = None

    def forward(self, input_data):
        if self.ac is not None:
            input_data = self.ac.forward(np.dot(input_data, self.weights) + self.bias)
            return input_data
        return np.dot(input_data, self.weights) + self.bias

    def get_size(self):
        return self.size

    def compile(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


