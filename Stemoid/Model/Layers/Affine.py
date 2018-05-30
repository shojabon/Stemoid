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
        self.input_shape = None

    def forward(self, input_data):
        self.original_shape = input_data.shape
        input_data = input_data.reshape(input_data.shape[0], -1)
        self.input = input_data
        if self.ac is not None:
            a = np.dot(input_data, self.weights) + self.bias
            x = self.ac.forward(a)
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

    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        return self.size

    def get_size(self):
        return {'type':'Affine',
                'size':self.size,
                'activation':self.ac}

    def compile(self):
        self.weights = 0.01 * np.random.rand(self.input_shape, self.size)
        self.bias = np.zeros(self.size)
