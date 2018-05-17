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

        self.original_shape = None
        self.input = None

    def forward(self, input_data):
        self.original_shape = input_data.shape
        self.input = input_data
        if self.ac is not None:
            x = self.ac.forward(np.dot(input_data, self.weights) + self.bias)
            return x
        x = np.dot(input_data, self.weights) + self.bias
        return x

    def backward(self, dout):
        if self.ac is not None:
            dout = self.ac.backward(dout)
        dx = np.dot(dout, self.weights.T)
        self.doutWeights = np.dot(self.input.T, dout)
        self.doutBias = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_shape)
        return dx

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_size(self):
        return self.size

    def compile(self, first_layer, next_layer):
        self.weights = 0.01 * np.random.rand(first_layer, next_layer)
        self.bias = 0.01 * np.random.rand(next_layer)
