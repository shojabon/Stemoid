import numpy as np


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.input_shape = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

    def get_output_shape(self, input_shape):
        self.input_shape = input_shape
        return input_shape


    def get_size(self):
        return {'type':'Dropout',
                'size':(self.dropout_ratio, self.input_shape),}

    def compile(self):
        pass