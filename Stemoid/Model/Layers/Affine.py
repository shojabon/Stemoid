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
        self.output = None

    def forward(self, input_data):
        self.original_shape = input_data.shape
        if self.ac is not None:
            input_data = self.ac.forward(np.dot(input_data, self.weights) + self.bias)
            self.output = input_data
            return input_data
        input_data = np.dot(input_data, self.weights) + self.bias
        self.output = input_data
        return input_data

    def backward(self, dout):
        if self.ac is not None:
            dout = self.ac.backward(dout)
        dx = np.dot(dout, self.weights.T)
        self.doutWeights = np.dot(self.output.T, dout)
        self.doutBias = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_shape)
        return dx


    def get_size(self):
        return self.size

    def compile(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias


