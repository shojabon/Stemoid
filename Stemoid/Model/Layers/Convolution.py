import numpy as np


class Error(Exception):
    pass
class Convolution:

    def __init__(self, filter_num, filter_shape, stride=1, padding=0, activation=None):
        self.filter_num = filter_num
        if len(filter_shape) is 2:
            self.filter_shape = filter_shape
        if len(filter_shape) is 1:
            filter_shape = (filter_shape[0], filter_shape[0])
            self.filter_shape = filter_shape
        if len(filter_shape) is not 2:
            raise Error('filter shape does not fit requirements')
        self.stride = stride
        self.padding = padding
        self.activation = activation
        self.input_shape = None

        self.weights = None
        self.bias = None

        self.input_data = None
        self.col = None
        self.col_weights = None

        self.doutWeights = None
        self.doutBias = None

    def set_weights(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def compile(self):
        self.weights = 0.01 * np.random.randn(self.filter_num, self.input_shape[0], self.filter_shape[0], self.filter_shape[0])
        self.bias = 0.01 * np.random.random(self.filter_num)

    def forward(self, input_data):
        FNumber, C, FHight, FWith = self.weights.shape
        N, C, H, W = input_data.shape
        out_h = 1 + int((H + 2*self.padding - FHight) / self.stride)
        out_w = 1 + int((W + 2*self.padding - FWith) / self.stride)
        col = self.im2col(input_data, FHight, FWith, self.stride, self.padding)
        col_weights = self.weights.reshape(FNumber, -1).T
        out = np.dot(col, col_weights) + self.bias
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        self.input_data = input_data
        self.col = col
        self.col_weights=col_weights
        if self.activation is not None:
            return self.activation.forward(out)
        return out

    def backward(self, dout):
        if self.activation is not None:
            dout = self.activation.backward(dout)
        FNumber, C, FHight, FWith = self.weights.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FNumber)

        self.doutBias = np.sum(dout, axis=0)
        self.doutWeights = np.dot(self.col.T, dout)
        self.doutWeights = self.doutWeights.T.reshape(FNumber, C, FHight, FWith)

        dcol = np.dot(dout, self.col_weights.T)
        dx = self.col2im(dcol, self.input_data.shape, FHight, FWith, self.stride, self.padding)

        return dx

    def im2col(self, input_data, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_data.shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1

        img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col

    def col2im(self, col, input_shape, filter_h, filter_w, stride=1, pad=0):
        N, C, H, W = input_shape
        out_h = (H + 2 * pad - filter_h) // stride + 1
        out_w = (W + 2 * pad - filter_w) // stride + 1
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

        img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        return img[:, :, pad:H + pad, pad:W + pad]

    def get_output_shape(self, input_shape):
        C, H, W = input_shape
        self.input_shape = input_shape
        if self.padding == -1:
            npadding = (W * self.stride - W + self.filter_shape[1] - self.stride)//2
            self.padding = npadding
        out_h = (H + 2 * self.padding - self.filter_shape[0]) // self.stride + 1
        out_w = (W + 2 * self.padding - self.filter_shape[1]) // self.stride + 1
        return self.filter_num, out_h, out_w


    def get_size(self):
        return {'type':'Convolution',
                'size':(self.filter_num, self.filter_shape[0], self.filter_shape[1]),
                'activation':self.activation}


